"""Loading and assembly of the widget front-end sources.

The JavaScript lives in :mod:`radrs.viz` ``static/`` as real ``.js`` files, one
per widget plus a ``shared.js`` of common helpers. anywidget loads ``_esm`` from
a blob URL, so a relative ``import "./shared.js"`` inside a widget file has no
base to resolve against, and we do not want a bundler in the install path.
Instead :func:`assemble_esm` concatenates ``shared.js`` with one widget file at
import time to make a single self-contained module.

``shared.js`` therefore holds only plain declarations, never imports or exports.
A widget file may still import from a CDN: ES modules require every ``import``
to precede other statements, so :func:`assemble_esm` hoists a widget's leading
imports above the shared prelude.

Set ``RADRS_VIZ_DEV=1`` to re-read the files on every widget instantiation, so
editing a ``.js`` file and re-running a notebook cell picks up the change.
"""

from __future__ import annotations

import base64
import json
import os
import re
from functools import lru_cache
from pathlib import Path

#: Directory holding the widget JavaScript and CSS.
STATIC_DIR = Path(__file__).parent / "static"

#: Environment variable that turns off source caching.
DEV_ENV_VAR = "RADRS_VIZ_DEV"

_SHARED_JS = "shared.js"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: An import statement, as opposed to an identifier that merely starts "import".
_IMPORT_RE = re.compile(r"^import\b")


def dev_mode() -> bool:
    """Is the reload-on-every-use development switch set?"""

    return os.environ.get(DEV_ENV_VAR, "").strip().lower() in _TRUTHY


@lru_cache(maxsize=None)
def _read_cached(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def read_static(name: str) -> str:
    """Read one file from ``static/``, cached unless :func:`dev_mode` is on."""

    if dev_mode():
        return (STATIC_DIR / name).read_text(encoding="utf-8")
    return _read_cached(name)


def _split_leading_imports(source: str) -> tuple[str, str]:
    """Split a widget module into its leading ``import`` block and the rest.

    Leading blank lines and ``//`` comments travel with the imports, which keeps
    a file's header comment at the top of the assembled module.
    """

    lines = source.splitlines(keepends=True)
    head: list[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("//"):
            head.append(lines[index])
            index += 1
            continue
        if _IMPORT_RE.match(stripped):
            # An import may wrap over several lines; it ends at the semicolon.
            while index < len(lines):
                head.append(lines[index])
                complete = lines[index].rstrip().endswith(";")
                index += 1
                if complete:
                    break
            continue
        break
    return "".join(head), "".join(lines[index:])


def assemble_esm(widget_file: str) -> str:
    """Build one widget's ES module: its imports, ``shared.js``, then its body."""

    imports, body = _split_leading_imports(read_static(widget_file))
    shared = read_static(_SHARED_JS)
    return f"{imports}{shared}\n{body}"


def widget_css() -> str:
    """The stylesheet every widget shares."""

    return read_static("widget.css")


def payload_to_html(
    state: dict[str, object],
    esm_str: str,
    css_str: str,
    width: int,
    height: int,
    extra_state: dict[str, object] | None = None,
) -> str:
    """Render a payload's widget state as a self-contained HTML document."""

    # Build JS state entries: base64-encode bytes, JSON-encode everything else.
    js_entries: list[str] = []
    for key, val in state.items():
        if isinstance(val, (bytes, bytearray, memoryview)):
            raw = bytes(val) if not isinstance(val, bytes) else val
            b64 = base64.b64encode(raw).decode("ascii")
            js_entries.append(f"{json.dumps(key)}: _b64ToAB({json.dumps(b64)})")
        else:
            js_entries.append(f"{json.dumps(key)}: {json.dumps(val)}")

    # Merge width, height, and any extra_state scalars.
    js_entries.append(f'"width": {json.dumps(width)}')
    js_entries.append(f'"height": {json.dumps(height)}')
    if extra_state:
        for key, val in extra_state.items():
            js_entries.append(f"{json.dumps(key)}: {json.dumps(val)}")

    js_state_body = ", ".join(js_entries)

    return (
        "<!DOCTYPE html>\n"
        '<html><head><meta charset="utf-8">\n'
        f"<style>{css_str}\n"
        f"#widget-root {{ width: {width}px; height: {height}px; }}\n"
        "</style></head>\n"
        '<body><div id="widget-root"></div>\n'
        '<script type="module">\n'
        "function _b64ToAB(b64) {\n"
        "  const bin = atob(b64);\n"
        "  const u8 = new Uint8Array(bin.length);\n"
        "  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);\n"
        "  return u8.buffer;\n"
        "}\n"
        f"const _S = {{{js_state_body}}};\n"
        # A static page has no kernel to sync to, so the model is read-only and
        # the write side is a no-op. `save_changes` and `off` are stubbed too:
        # the widgets call them on hover and teardown, and without them every
        # hover threw a TypeError into the console.
        "const model = {\n"
        "  get(k) { return _S[k]; },\n"
        "  set() {},\n"
        "  save_changes() {},\n"
        "  on() {},\n"
        "  off() {},\n"
        "};\n"
        f"const _esm = {json.dumps(esm_str)};\n"
        'const _blob = new Blob([_esm], {type:"text/javascript"});\n'
        "const _url = URL.createObjectURL(_blob);\n"
        "const _mod = await import(_url);\n"
        "URL.revokeObjectURL(_url);\n"
        '_mod.default.render({ model, el: document.getElementById("widget-root") });\n'
        "</script></body></html>"
    )
