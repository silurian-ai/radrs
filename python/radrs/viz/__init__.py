"""Visualization helpers for interactive raystack visualization.

This package provides:

- sweep and moment selectors over raystack DataTree nodes
- a vectorized adapter from sweep data to polar point buffers
- an optional anywidget polar renderer for marimo notebooks

The public surface is re-exported here, so ``import radrs.viz as viz`` and
``from radrs.viz import prepare_polar_payload`` work as they always have. The
implementation is split across :mod:`radrs.viz.scales` (colour tables),
:mod:`radrs.viz.geometry` (beam geometry), :mod:`radrs.viz.payloads` (payload
dataclasses and adapters), :mod:`radrs.viz.widgets` (anywidget classes) and
:mod:`radrs.viz.assets` (the widget JavaScript under ``static/``).
"""

from __future__ import annotations

# Re-exported for contributors and tests working on the widget front-end. They
# stay out of ``__all__`` so the rendered reference page keeps to the data API.
from . import assets
from .assets import DEV_ENV_VAR, STATIC_DIR, assemble_esm, dev_mode, widget_css
from .geometry import EARTH_RADIUS_M, EFFECTIVE_EARTH_RADIUS_M, beam_geometry
from .payloads import (
    DEFAULT_BYTE_BUDGET,
    QUANT_MAX,
    QUANT_NAN,
    FoldedWaterfallPayload,
    GridPayload,
    Payload,
    PolarPayload,
    SweepInfo,
    VolumePayload,
    WaterfallPayload,
    available_moments,
    dequantize,
    get_returns_and_sweeps,
    quantize,
    prepare_cappi_payload,
    prepare_folded_waterfall_payload,
    prepare_polar_payload,
    prepare_ray_payload,
    prepare_volume_payload,
    prepare_waterfall_payload,
    prepare_xsec_payload,
    sweep_infos,
    sweep_offsets,
    widget_state_nbytes,
)
from .scales import (
    COLOR_SCALES,
    FALLBACK_COLORMAP,
    MOMENT_NAMES,
    ColorScale,
    _value_bounds,
    color_scale_for,
)
from .widgets import (
    FoldedWaterfallWidget,
    GridWidget,
    PolarWidget,
    VolumeWidget,
    WaterfallWidget,
)

__all__ = [
    "DEFAULT_BYTE_BUDGET",
    "EARTH_RADIUS_M",
    "EFFECTIVE_EARTH_RADIUS_M",
    "QUANT_MAX",
    "QUANT_NAN",
    "MOMENT_NAMES",
    "COLOR_SCALES",
    "FALLBACK_COLORMAP",
    "ColorScale",
    "color_scale_for",
    "SweepInfo",
    "PolarPayload",
    "VolumePayload",
    "GridPayload",
    "WaterfallPayload",
    "FoldedWaterfallPayload",
    "Payload",
    "available_moments",
    "beam_geometry",
    "quantize",
    "dequantize",
    "widget_state_nbytes",
    "get_returns_and_sweeps",
    "sweep_offsets",
    "sweep_infos",
    "prepare_polar_payload",
    "prepare_volume_payload",
    "prepare_ray_payload",
    "prepare_cappi_payload",
    "prepare_xsec_payload",
    "prepare_waterfall_payload",
    "prepare_folded_waterfall_payload",
    "PolarWidget",
    "VolumeWidget",
    "GridWidget",
    "WaterfallWidget",
    "FoldedWaterfallWidget",
]
