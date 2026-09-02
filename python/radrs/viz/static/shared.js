// Helpers shared by every radrs widget module.
//
// This file is not imported: `radrs.viz.assets.assemble_esm` concatenates it
// ahead of one widget file to make a single ES module, because anywidget loads
// `_esm` from a blob URL where a relative `import "./shared.js"` cannot resolve.
// Keep it to plain `function`/`const` declarations at the top level, with no
// imports and no exports, so the concatenation stays a valid module.

// --- Binary transport ----------------------------------------------------

function toArrayBuffer(raw) {
  if (!raw) {
    return new ArrayBuffer(0);
  }
  if (raw instanceof ArrayBuffer) {
    return raw;
  }
  if (ArrayBuffer.isView(raw)) {
    return raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  }
  return new Uint8Array(raw).buffer;
}

/** Decode a synced bytes trait into a typed array, dropping any partial tail. */
function decodeArray(raw, ctor) {
  const buffer = toArrayBuffer(raw);
  if (buffer.byteLength === 0) {
    return new ctor(0);
  }
  const bytesPerElement = ctor.BYTES_PER_ELEMENT;
  const trimmed = buffer.byteLength - (buffer.byteLength % bytesPerElement);
  return new ctor(buffer.slice(0, trimmed));
}

// --- Quantized values ----------------------------------------------------
// Every payload ships its values as uint16 codes over [vmin, vmax]. Keep these
// in step with QUANT_MAX / QUANT_NAN in `radrs.viz.payloads`.

/** Highest finite code; 0..QUANT_MAX spans [vmin, vmax]. */
const QUANT_MAX = 65534;

/** Reserved code for NaN and missing data. */
const QUANT_NAN = 65535;

/** Normalized 0..1 position of a code, for indexing a palette LUT. */
function quantFraction(code) {
  return code / QUANT_MAX;
}

/** Recover a physical value from its code; QUANT_NAN comes back as NaN. */
function dequantize(code, vmin, vmax) {
  if (code === QUANT_NAN) return NaN;
  const lo = Number(vmin);
  const hi = Number(vmax);
  const span = Number.isFinite(lo) && Number.isFinite(hi) && hi > lo ? hi - lo : 1.0;
  return lo + (code / QUANT_MAX) * span;
}

// --- Numeric helpers -----------------------------------------------------

function clamp01(x) {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

function clamp(v, lo, hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

// --- Beam geometry -------------------------------------------------------
// Mirrors `radrs.viz.geometry.beam_geometry`: the 4/3 effective earth radius
// model, which folds mean refraction into an inflated earth so the beam can be
// treated as a straight line. Keep the constants in step with the Python side.

const EARTH_RADIUS_M = 6371000.0;
const EFFECTIVE_EARTH_RADIUS_M = (4.0 / 3.0) * EARTH_RADIUS_M;

/** Height above the antenna of a gate at `rangeM` along a beam at `elevationDeg`. */
function beamHeightM(rangeM, elevationDeg) {
  const r = Number(rangeM);
  if (!Number.isFinite(r)) return NaN;
  const el = (Number(elevationDeg) * Math.PI) / 180.0;
  const re = EFFECTIVE_EARTH_RADIUS_M;
  return Math.sqrt(r * r + re * re + 2.0 * r * re * Math.sin(el)) - re;
}

/** Great-circle distance along the surface to that same gate. */
function beamGroundRangeM(rangeM, elevationDeg) {
  const r = Number(rangeM);
  if (!Number.isFinite(r)) return NaN;
  const el = (Number(elevationDeg) * Math.PI) / 180.0;
  const re = EFFECTIVE_EARTH_RADIUS_M;
  const h = beamHeightM(r, elevationDeg);
  // Clamp guards against float error pushing the ratio past 1 at extreme ranges.
  return re * Math.asin(clamp((r * Math.cos(el)) / (re + h), -1.0, 1.0));
}

// --- Time formatting -----------------------------------------------------

function formatTimeUTC(ms) {
  const d = new Date(ms);
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  const ss = String(d.getUTCSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

/** Same as `formatTimeUTC`, with milliseconds. Folds within one radial share a second. */
function formatTimeUTCms(ms) {
  const d = new Date(ms);
  const ms3 = String(d.getUTCMilliseconds()).padStart(3, "0");
  return `${formatTimeUTC(ms)}.${ms3}`;
}

// --- Colour scales -------------------------------------------------------
// Named palettes as [position, r, g, b] stops; position runs 0..1 across
// [vmin, vmax]. One definition, shared by every widget module.
const PALETTES = {
  viridis: [
    [0.0, 68, 1, 84], [0.25, 59, 82, 139], [0.5, 33, 145, 140],
    [0.75, 94, 201, 98], [1.0, 253, 231, 37],
  ],
  nws_reflectivity: [
    [0.0, 40, 40, 48], [0.190476, 96, 96, 108], [0.333333, 0, 236, 236],
    [0.380952, 1, 160, 246], [0.428571, 0, 0, 246], [0.476190, 0, 255, 0],
    [0.523810, 0, 200, 0], [0.571429, 0, 144, 0], [0.619048, 255, 255, 0],
    [0.666667, 231, 192, 0], [0.714286, 255, 144, 0], [0.761905, 255, 0, 0],
    [0.809524, 214, 0, 0], [0.857143, 192, 0, 0], [0.904762, 255, 0, 255],
    [0.952381, 153, 85, 201], [1.0, 255, 255, 255],
  ],
  nws_velocity: [
    [0.0, 12, 68, 12], [0.25, 0, 200, 0], [0.45, 120, 235, 120],
    [0.5, 235, 235, 235], [0.55, 245, 150, 150], [0.75, 225, 30, 30],
    [1.0, 110, 10, 10],
  ],
  magma: [
    [0.0, 0, 0, 4], [0.25, 81, 18, 124], [0.5, 183, 55, 121],
    [0.75, 252, 137, 97], [1.0, 252, 253, 191],
  ],
  nws_zdr: [
    [0.0, 60, 60, 90], [0.25, 160, 170, 185], [0.4375, 40, 170, 90],
    [0.625, 235, 210, 60], [0.8125, 235, 120, 40], [1.0, 200, 30, 30],
  ],
  nws_rhohv: [
    [0.0, 30, 40, 120], [0.428571, 40, 160, 190], [0.714286, 90, 200, 90],
    [0.857143, 240, 220, 70], [1.0, 200, 40, 40],
  ],
  cyclic: [
    [0.0, 230, 90, 90], [0.166667, 220, 200, 70], [0.333333, 90, 200, 90],
    [0.5, 70, 200, 210], [0.666667, 90, 110, 220], [0.833333, 210, 100, 210],
    [1.0, 230, 90, 90],
  ],
};

/** Build a 256-entry RGB lookup table for a named palette (once per render). */
function buildLut(name) {
  const stops = PALETTES[name] || PALETTES.viridis;
  const lut = new Uint8Array(768);
  let seg = 0;
  for (let i = 0; i < 256; i += 1) {
    const t = i / 255;
    while (seg < stops.length - 2 && t > stops[seg + 1][0]) seg += 1;
    const a = stops[seg];
    const b = stops[seg + 1];
    const span = b[0] - a[0];
    const local = span > 0 ? clamp01((t - a[0]) / span) : 0;
    lut[i * 3] = Math.round(lerp(a[1], b[1], local));
    lut[i * 3 + 1] = Math.round(lerp(a[2], b[2], local));
    lut[i * 3 + 2] = Math.round(lerp(a[3], b[3], local));
  }
  return lut;
}

function lutColor(lut, t) {
  const off = Math.round(clamp01(t) * 255) * 3;
  return [lut[off], lut[off + 1], lut[off + 2]];
}

// --- Axis and colorbar drawing -------------------------------------------

function formatScaleTick(v) {
  if (!Number.isFinite(v)) return "?";
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(1);
  return v.toFixed(2);
}

/** Draw a vertical colorbar panel: strip, ticks, moment name and units. */
function drawColorbar(ctx, lut, meta, x, y, w, h) {
  const vmin = Number(meta.vmin);
  const vmax = Number(meta.vmax);
  if (!Number.isFinite(vmin) || !Number.isFinite(vmax)) return;
  if (w < 40 || h < 60) return;

  ctx.save();
  ctx.fillStyle = "rgba(255, 255, 255, 0.88)";
  ctx.strokeStyle = "rgba(110, 118, 129, 0.6)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 4);
  ctx.fill();
  ctx.stroke();

  const stripX = x + 8;
  const stripY = y + 30;
  const stripW = 12;
  const stripH = h - 40;
  for (let i = 0; i < stripH; i += 1) {
    const t = 1 - i / Math.max(1, stripH - 1);
    const rgb = lutColor(lut, t);
    ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    ctx.fillRect(stripX, stripY + i, stripW, 1);
  }
  ctx.strokeStyle = "rgba(110, 118, 129, 0.8)";
  ctx.strokeRect(stripX, stripY, stripW, stripH);

  ctx.fillStyle = "rgba(17, 24, 39, 0.9)";
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
  ctx.fillText(String(meta.moment || ""), x + 6, y + 13);
  const units = String(meta.units || "");
  if (units) {
    ctx.fillText(`[${units}]`, x + 6, y + 24);
  }

  ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
  for (let i = 0; i < 5; i += 1) {
    const frac = i / 4;
    const py = stripY + frac * stripH;
    ctx.beginPath();
    ctx.moveTo(stripX + stripW, py);
    ctx.lineTo(stripX + stripW + 3, py);
    ctx.stroke();
    ctx.fillText(
      formatScaleTick(vmax - frac * (vmax - vmin)),
      stripX + stripW + 6,
      py + 3,
    );
  }
  ctx.restore();
}

/** Compute nice round tick positions within [lo, hi] in world units. */
function niceTicks(lo, hi, maxTicks) {
  const range = hi - lo;
  if (range <= 0) return [];
  const rough = range / maxTicks;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  let step = mag;
  if (rough / mag >= 5) step = mag * 5;
  else if (rough / mag >= 2) step = mag * 2;
  const ticks = [];
  const start = Math.ceil(lo / step) * step;
  for (let v = start; v <= hi; v += step) {
    ticks.push(v);
  }
  return ticks;
}

// --- Hover picking -------------------------------------------------------

/** Nearest non-empty entry in a per-pixel index buffer, searching outward. */
function nearestPick(pick, width, height, x, y) {
  const ix = Math.round(x);
  const iy = Math.round(y);
  for (let radius = 0; radius <= 3; radius += 1) {
    for (let dy = -radius; dy <= radius; dy += 1) {
      const py = iy + dy;
      if (py < 0 || py >= height) continue;
      for (let dx = -radius; dx <= radius; dx += 1) {
        const px = ix + dx;
        if (px < 0 || px >= width) continue;
        const idx = pick[py * width + px];
        if (idx >= 0) {
          return idx;
        }
      }
    }
  }
  return -1;
}
