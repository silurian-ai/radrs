"""anywidget renderers for the payload types.

Each widget class pairs one payload with one file under ``static/``. The JS is
assembled by :mod:`radrs.viz.assets`; with ``RADRS_VIZ_DEV=1`` set, every
instantiation re-reads the sources so an edited ``.js`` file shows up on the
next cell run.
"""

from __future__ import annotations

import xarray as xr

from . import assets
from .payloads import (
    DEFAULT_BYTE_BUDGET,
    FoldedWaterfallPayload,
    GridPayload,
    PolarPayload,
    VolumePayload,
    WaterfallPayload,
    get_returns_and_sweeps,
    prepare_cappi_payload,
    prepare_folded_waterfall_payload,
    prepare_polar_payload,
    prepare_ray_payload,
    prepare_volume_payload,
    prepare_waterfall_payload,
    prepare_xsec_payload,
)

try:
    import anywidget as _anywidget
    import traitlets as _traitlets
except ImportError:
    _anywidget = None
    _traitlets = None


def _state_bytes(state: dict[str, object], key: str) -> bytes:
    raw = state.get(key)
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, bytearray):
        return bytes(raw)
    if isinstance(raw, memoryview):
        return raw.tobytes()
    raise TypeError(f"widget state key '{key}' must be bytes-like")


def _reload_sources(widget: object) -> None:
    """Re-read this widget's JS and CSS when ``RADRS_VIZ_DEV`` is set.

    The class attributes are assembled once at import; anywidget turns them into
    per-instance traits, so re-assigning here picks up an edited ``.js`` file on
    the next widget instantiation without restarting the kernel.
    """

    if not assets.dev_mode():
        return
    widget._esm = assets.assemble_esm(widget._widget_js)
    widget._css = assets.widget_css()


if _anywidget is not None and _traitlets is not None:

    class PolarWidget(_anywidget.AnyWidget):
        """Binary anywidget renderer for quick polar inspection."""

        _widget_js = "polar.js"
        _esm = assets.assemble_esm("polar.js")
        _css = assets.widget_css()

        width = _traitlets.Int(760).tag(sync=True)
        height = _traitlets.Int(760).tag(sync=True)

        value_bytes = _traitlets.Bytes(b"").tag(sync=True)
        azimuth_bytes = _traitlets.Bytes(b"").tag(sync=True)
        elevation_bytes = _traitlets.Bytes(b"").tag(sync=True)
        base_range_bytes = _traitlets.Bytes(b"").tag(sync=True)
        range_step_bytes = _traitlets.Bytes(b"").tag(sync=True)
        return_index_bytes = _traitlets.Bytes(b"").tag(sync=True)
        return_time_ms_bytes = _traitlets.Bytes(b"").tag(sync=True)

        meta = _traitlets.Dict(default_value={}).tag(sync=True)
        hover = _traitlets.Dict(default_value={}).tag(sync=True)

        _payload_traits = (
            "value_bytes",
            "azimuth_bytes",
            "elevation_bytes",
            "base_range_bytes",
            "range_step_bytes",
            "return_index_bytes",
            "return_time_ms_bytes",
        )

        def __init__(self, width: int = 760, height: int = 760):
            super().__init__()
            _reload_sources(self)
            self.width = int(width)
            self.height = int(height)

        def set_payload(self, payload: PolarPayload) -> None:
            state = payload.to_widget_state()
            for trait in self._payload_traits:
                setattr(self, trait, _state_bytes(state, trait))

            meta = state.get("meta")
            if not isinstance(meta, dict):
                raise TypeError("widget state meta must be a dict")
            self.meta = meta

        def clear(self) -> None:
            for trait in self._payload_traits:
                setattr(self, trait, b"")
            self.meta = {
                "n_returns": 0,
                "n_gates": 0,
                "point_count": 0,
                "moment": "",
                "sweep_number": -1,
            }
            self.hover = {}

        @classmethod
        def from_datatree(
            cls,
            dt: xr.DataTree,
            moment: str,
            *,
            sweep_index: int = 0,
            max_points: int | None = None,
            byte_budget: int | None = DEFAULT_BYTE_BUDGET,
            width: int = 760,
            height: int = 760,
        ) -> "PolarWidget":
            returns, sweeps = get_returns_and_sweeps(dt)
            payload = prepare_polar_payload(
                returns,
                sweeps,
                sweep_index,
                moment,
                max_points=max_points,
                byte_budget=byte_budget,
            )
            w = cls(width=width, height=height)
            w.set_payload(payload)
            return w

    class VolumeWidget(_anywidget.AnyWidget):
        """Binary anywidget renderer for volume-wide ray points."""

        _widget_js = "volume.js"
        _esm = assets.assemble_esm("volume.js")
        _css = assets.widget_css()

        width = _traitlets.Int(760).tag(sync=True)
        height = _traitlets.Int(760).tag(sync=True)
        yaw_deg = _traitlets.Float(35.0).tag(sync=True)
        pitch_deg = _traitlets.Float(30.0).tag(sync=True)

        value_bytes = _traitlets.Bytes(b"").tag(sync=True)
        return_slot_bytes = _traitlets.Bytes(b"").tag(sync=True)
        gate_index_bytes = _traitlets.Bytes(b"").tag(sync=True)
        return_index_bytes = _traitlets.Bytes(b"").tag(sync=True)
        azimuth_bytes = _traitlets.Bytes(b"").tag(sync=True)
        elevation_bytes = _traitlets.Bytes(b"").tag(sync=True)
        base_range_bytes = _traitlets.Bytes(b"").tag(sync=True)
        range_step_bytes = _traitlets.Bytes(b"").tag(sync=True)

        meta = _traitlets.Dict(default_value={}).tag(sync=True)
        hover = _traitlets.Dict(default_value={}).tag(sync=True)

        _payload_traits = (
            "value_bytes",
            "return_slot_bytes",
            "gate_index_bytes",
            "return_index_bytes",
            "azimuth_bytes",
            "elevation_bytes",
            "base_range_bytes",
            "range_step_bytes",
        )

        def __init__(
            self,
            width: int = 760,
            height: int = 760,
            yaw_deg: float = 35.0,
            pitch_deg: float = 30.0,
        ):
            super().__init__()
            _reload_sources(self)
            self.width = int(width)
            self.height = int(height)
            self.yaw_deg = float(yaw_deg)
            self.pitch_deg = float(pitch_deg)

        def set_payload(self, payload: VolumePayload) -> None:
            state = payload.to_widget_state()
            for trait in self._payload_traits:
                setattr(self, trait, _state_bytes(state, trait))

            meta = state.get("meta")
            if not isinstance(meta, dict):
                raise TypeError("widget state meta must be a dict")
            self.meta = meta

        def clear(self) -> None:
            for trait in self._payload_traits:
                setattr(self, trait, b"")
            self.meta = {
                "point_count": 0,
                "return_count": 0,
                "moment": "",
                "render_mode": "points",
            }
            self.hover = {}

        @classmethod
        def from_datatree(
            cls,
            dt: xr.DataTree,
            moment: str,
            *,
            max_points: int | None = None,
            byte_budget: int | None = DEFAULT_BYTE_BUDGET,
            render_mode: str = "points",
            width: int = 760,
            height: int = 760,
        ) -> "VolumeWidget":
            returns, _sweeps = get_returns_and_sweeps(dt)
            prepare = prepare_ray_payload if render_mode == "rays" else prepare_volume_payload
            payload = prepare(
                returns, moment, max_points=max_points, byte_budget=byte_budget
            )
            w = cls(width=width, height=height)
            w.set_payload(payload)
            return w

    class GridWidget(_anywidget.AnyWidget):
        """Binary anywidget renderer for 2D gridded CAPPI / cross-section views."""

        _widget_js = "grid.js"
        _esm = assets.assemble_esm("grid.js")
        _css = assets.widget_css()

        width = _traitlets.Int(760).tag(sync=True)
        height = _traitlets.Int(760).tag(sync=True)
        grid_bytes = _traitlets.Bytes(b"").tag(sync=True)
        meta = _traitlets.Dict(default_value={}).tag(sync=True)
        hover = _traitlets.Dict(default_value={}).tag(sync=True)

        def __init__(self, width: int = 760, height: int = 760):
            super().__init__()
            _reload_sources(self)
            self.width = int(width)
            self.height = int(height)

        def set_payload(self, payload: GridPayload) -> None:
            state = payload.to_widget_state()
            self.grid_bytes = _state_bytes(state, "grid_bytes")

            meta = state.get("meta")
            if not isinstance(meta, dict):
                raise TypeError("widget state meta must be a dict")
            self.meta = meta

        def clear(self) -> None:
            self.grid_bytes = b""
            self.meta = {"grid_mode": "cappi", "n_rows": 0, "n_cols": 0, "moment": ""}
            self.hover = {}

        @classmethod
        def from_cappi(
            cls,
            dt: xr.DataTree,
            moment: str,
            *,
            altitude_m: float = 2000.0,
            tolerance_m: float = 500.0,
            grid_size: int = 500,
            byte_budget: int | None = DEFAULT_BYTE_BUDGET,
            width: int = 760,
            height: int = 760,
        ) -> "GridWidget":
            returns, sweeps = get_returns_and_sweeps(dt)
            payload = prepare_cappi_payload(
                returns, sweeps, moment,
                altitude_m=altitude_m, tolerance_m=tolerance_m,
                grid_size=grid_size, byte_budget=byte_budget,
            )
            w = cls(width=width, height=height)
            w.set_payload(payload)
            return w

        @classmethod
        def from_xsec(
            cls,
            dt: xr.DataTree,
            moment: str,
            *,
            azimuth_deg: float = 0.0,
            azimuth_tolerance_deg: float = 2.0,
            grid_size: int = 500,
            byte_budget: int | None = DEFAULT_BYTE_BUDGET,
            width: int = 760,
            height: int = 760,
        ) -> "GridWidget":
            returns, sweeps = get_returns_and_sweeps(dt)
            payload = prepare_xsec_payload(
                returns, sweeps, moment,
                azimuth_deg=azimuth_deg,
                azimuth_tolerance_deg=azimuth_tolerance_deg,
                grid_size=grid_size, byte_budget=byte_budget,
            )
            w = cls(width=width, height=height)
            w.set_payload(payload)
            return w

    class WaterfallWidget(_anywidget.AnyWidget):
        """Binary anywidget renderer for waterfall (return_time x range) heatmaps."""

        _widget_js = "waterfall.js"
        _esm = assets.assemble_esm("waterfall.js")
        _css = assets.widget_css()

        width = _traitlets.Int(760).tag(sync=True)
        height = _traitlets.Int(760).tag(sync=True)
        grid_bytes = _traitlets.Bytes(b"").tag(sync=True)
        azimuth_bytes = _traitlets.Bytes(b"").tag(sync=True)
        elevation_bytes = _traitlets.Bytes(b"").tag(sync=True)
        return_time_ms_bytes = _traitlets.Bytes(b"").tag(sync=True)
        sweep_number_bytes = _traitlets.Bytes(b"").tag(sync=True)
        base_range_bytes = _traitlets.Bytes(b"").tag(sync=True)
        range_step_bytes = _traitlets.Bytes(b"").tag(sync=True)
        sweep_boundary_bytes = _traitlets.Bytes(b"").tag(sync=True)
        meta = _traitlets.Dict(default_value={}).tag(sync=True)
        hover = _traitlets.Dict(default_value={}).tag(sync=True)

        _payload_traits = (
            "grid_bytes",
            "azimuth_bytes",
            "elevation_bytes",
            "return_time_ms_bytes",
            "sweep_number_bytes",
            "base_range_bytes",
            "range_step_bytes",
            "sweep_boundary_bytes",
        )

        def __init__(self, width: int = 760, height: int = 760):
            super().__init__()
            _reload_sources(self)
            self.width = int(width)
            self.height = int(height)

        def set_payload(self, payload: WaterfallPayload) -> None:
            state = payload.to_widget_state()
            for trait in self._payload_traits:
                setattr(self, trait, _state_bytes(state, trait))

            meta = state.get("meta")
            if not isinstance(meta, dict):
                raise TypeError("widget state meta must be a dict")
            self.meta = meta

        def clear(self) -> None:
            for trait in self._payload_traits:
                setattr(self, trait, b"")
            self.meta = {"n_returns": 0, "n_range": 0, "moment": ""}
            self.hover = {}

        @classmethod
        def from_datatree(
            cls,
            dt: xr.DataTree,
            moment: str,
            *,
            max_returns: int = 2048,
            max_range: int = 1024,
            byte_budget: int | None = DEFAULT_BYTE_BUDGET,
            width: int = 760,
            height: int = 760,
        ) -> "WaterfallWidget":
            returns, _sweeps = get_returns_and_sweeps(dt)
            payload = prepare_waterfall_payload(
                returns,
                moment,
                max_returns=max_returns,
                max_range=max_range,
                byte_budget=byte_budget,
            )
            w = cls(width=width, height=height)
            w.set_payload(payload)
            return w

    class FoldedWaterfallWidget(_anywidget.AnyWidget):
        """Binary anywidget renderer for the row-per-return folded view."""

        _widget_js = "folded_waterfall.js"
        _esm = assets.assemble_esm("folded_waterfall.js")
        _css = assets.widget_css()

        width = _traitlets.Int(900).tag(sync=True)
        height = _traitlets.Int(900).tag(sync=True)
        grid_bytes = _traitlets.Bytes(b"").tag(sync=True)
        vcp_index_bytes = _traitlets.Bytes(b"").tag(sync=True)
        vcp_time_ms_bytes = _traitlets.Bytes(b"").tag(sync=True)
        sweep_number_bytes = _traitlets.Bytes(b"").tag(sync=True)
        azimuth_bytes = _traitlets.Bytes(b"").tag(sync=True)
        elevation_bytes = _traitlets.Bytes(b"").tag(sync=True)
        return_time_ms_bytes = _traitlets.Bytes(b"").tag(sync=True)
        base_range_bytes = _traitlets.Bytes(b"").tag(sync=True)
        range_step_bytes = _traitlets.Bytes(b"").tag(sync=True)
        fold_index_bytes = _traitlets.Bytes(b"").tag(sync=True)
        group_start_bytes = _traitlets.Bytes(b"").tag(sync=True)
        meta = _traitlets.Dict(default_value={}).tag(sync=True)
        hover = _traitlets.Dict(default_value={}).tag(sync=True)

        def __init__(self, width: int = 900, height: int = 900):
            super().__init__()
            _reload_sources(self)
            self.width = int(width)
            self.height = int(height)

        _payload_traits = (
            "grid_bytes",
            "vcp_index_bytes",
            "vcp_time_ms_bytes",
            "sweep_number_bytes",
            "azimuth_bytes",
            "elevation_bytes",
            "return_time_ms_bytes",
            "base_range_bytes",
            "range_step_bytes",
            "fold_index_bytes",
            "group_start_bytes",
        )

        def set_payload(self, payload: FoldedWaterfallPayload) -> None:
            state = payload.to_widget_state()
            for trait in self._payload_traits:
                setattr(self, trait, _state_bytes(state, trait))

            meta = state.get("meta")
            if not isinstance(meta, dict):
                raise TypeError("widget state meta must be a dict")
            self.meta = meta

        def clear(self) -> None:
            for trait in self._payload_traits:
                setattr(self, trait, b"")
            self.meta = {"n_rows": 0, "n_range": 0, "moment": ""}
            self.hover = {}

        @classmethod
        def from_datatree(
            cls,
            dt: xr.DataTree,
            moment: str,
            *,
            row_offset: int = 0,
            row_count: int | None = None,
            max_range: int | None = None,
            byte_budget: int | None = DEFAULT_BYTE_BUDGET,
            width: int = 900,
            height: int = 900,
        ) -> "FoldedWaterfallWidget":
            returns, _sweeps = get_returns_and_sweeps(dt)
            payload = prepare_folded_waterfall_payload(
                returns,
                moment,
                row_offset=row_offset,
                row_count=row_count,
                max_range=max_range,
                byte_budget=byte_budget,
            )
            w = cls(width=width, height=height)
            w.set_payload(payload)
            return w


else:

    class PolarWidget:  # pragma: no cover - runtime guard for optional deps
        def __init__(self, *args: object, **kwargs: object):
            raise ImportError(
                "PolarWidget requires optional dependencies: anywidget and traitlets"
            )

        @classmethod
        def from_datatree(cls, *args: object, **kwargs: object) -> "PolarWidget":
            raise ImportError(
                "PolarWidget requires optional dependencies: anywidget and traitlets"
            )

    class VolumeWidget:  # pragma: no cover - runtime guard for optional deps
        def __init__(self, *args: object, **kwargs: object):
            raise ImportError(
                "VolumeWidget requires optional dependencies: anywidget and traitlets"
            )

        @classmethod
        def from_datatree(cls, *args: object, **kwargs: object) -> "VolumeWidget":
            raise ImportError(
                "VolumeWidget requires optional dependencies: anywidget and traitlets"
            )

    class GridWidget:  # pragma: no cover - runtime guard for optional deps
        def __init__(self, *args: object, **kwargs: object):
            raise ImportError(
                "GridWidget requires optional dependencies: anywidget and traitlets"
            )

        @classmethod
        def from_cappi(cls, *args: object, **kwargs: object) -> "GridWidget":
            raise ImportError(
                "GridWidget requires optional dependencies: anywidget and traitlets"
            )

        @classmethod
        def from_xsec(cls, *args: object, **kwargs: object) -> "GridWidget":
            raise ImportError(
                "GridWidget requires optional dependencies: anywidget and traitlets"
            )

    class WaterfallWidget:  # pragma: no cover - runtime guard for optional deps
        def __init__(self, *args: object, **kwargs: object):
            raise ImportError(
                "WaterfallWidget requires optional dependencies: anywidget and traitlets"
            )

        @classmethod
        def from_datatree(cls, *args: object, **kwargs: object) -> "WaterfallWidget":
            raise ImportError(
                "WaterfallWidget requires optional dependencies: anywidget and traitlets"
            )

    class FoldedWaterfallWidget:  # pragma: no cover - runtime guard for optional deps
        def __init__(self, *args: object, **kwargs: object):
            raise ImportError(
                "FoldedWaterfallWidget requires optional dependencies: "
                "anywidget and traitlets"
            )

        @classmethod
        def from_datatree(
            cls, *args: object, **kwargs: object
        ) -> "FoldedWaterfallWidget":
            raise ImportError(
                "FoldedWaterfallWidget requires optional dependencies: "
                "anywidget and traitlets"
            )
