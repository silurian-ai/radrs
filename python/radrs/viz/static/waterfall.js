/** Map elevation to blue-red colour. */
function elevationColor(t) {
  const x = clamp01(t);
  return [
    Math.round(lerp(50, 220, x)),
    Math.round(lerp(50, 50, x)),
    Math.round(lerp(220, 50, 1 - x)),
  ];
}

export default {
  render({ model, el }) {
    const root = document.createElement("div");
    root.className = "radrs-viz-root";
    const canvas = document.createElement("canvas");
    canvas.className = "radrs-viz-canvas";
    const tooltip = document.createElement("div");
    tooltip.className = "radrs-viz-tooltip";
    tooltip.style.display = "none";

    root.appendChild(canvas);
    root.appendChild(tooltip);
    el.appendChild(root);

    let grid = new Uint16Array(0);
    let azimuth = new Float32Array(0);
    let elevation = new Float32Array(0);
    let returnTimeMs = new Float64Array(0);
    let sweepNumber = new Uint16Array(0);
    let baseRange = new Float32Array(0);
    let rangeStep = new Float32Array(0);
    let sweepBoundaries = new Int32Array(0);

    let lastHoverKey = "";

    const marginLeft = 60;
    const marginTop = 28;
    const colorbarW = 66;
    // Right margin reserves the colorbar gutter beyond the elevation strip.
    const marginRight = 40 + colorbarW;
    const marginBottom = 40;
    const elevStripW = 30;

    function redraw() {
      const width = Number(model.get("width")) || 760;
      const height = Number(model.get("height")) || 760;
      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, width, height);

      const meta = model.get("meta") || {};
      grid = decodeArray(model.get("grid_bytes"), Uint16Array);
      azimuth = decodeArray(model.get("azimuth_bytes"), Float32Array);
      elevation = decodeArray(model.get("elevation_bytes"), Float32Array);
      returnTimeMs = decodeArray(model.get("return_time_ms_bytes"), Float64Array);
      sweepNumber = decodeArray(model.get("sweep_number_bytes"), Uint16Array);
      baseRange = decodeArray(model.get("base_range_bytes"), Float32Array);
      rangeStep = decodeArray(model.get("range_step_bytes"), Float32Array);
      sweepBoundaries = decodeArray(model.get("sweep_boundary_bytes"), Int32Array);

      const nReturns = Number(meta.n_returns) || 0;
      const nRange = Number(meta.n_range) || 0;
      const lut = buildLut(String(meta.colormap || "viridis"));

      const plotW = width - marginLeft - marginRight - elevStripW;
      const plotH = height - marginTop - marginBottom;

      // --- Main heatmap ---
      if (nReturns > 0 && nRange > 0 && grid.length >= nReturns * nRange) {
        const image = ctx.createImageData(plotW, plotH);
        const pixels = image.data;

        for (let py = 0; py < plotH; py++) {
          const row = Math.floor((py / plotH) * nReturns);
          for (let px = 0; px < plotW; px++) {
            const col = Math.floor((px / plotW) * nRange);
            const code = grid[row * nRange + col];
            const off = (py * plotW + px) * 4;
            if (code === QUANT_NAN) {
              pixels[off + 3] = 0;
              continue;
            }
            const [r, g, b] = lutColor(lut, quantFraction(code));
            pixels[off] = r;
            pixels[off + 1] = g;
            pixels[off + 2] = b;
            pixels[off + 3] = 255;
          }
        }
        ctx.putImageData(image, marginLeft, marginTop);
      }

      // --- Sweep boundary lines ---
      if (nReturns > 0) {
        ctx.save();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
        ctx.lineWidth = 1;
        for (let i = 0; i < sweepBoundaries.length; i++) {
          const bndIdx = sweepBoundaries[i];
          const py = marginTop + (bndIdx / nReturns) * plotH;
          ctx.beginPath();
          ctx.moveTo(marginLeft, py);
          ctx.lineTo(marginLeft + plotW, py);
          ctx.stroke();
        }
        ctx.restore();
      }

      // --- Elevation side strip ---
      if (nReturns > 0 && elevation.length >= nReturns) {
        let elMin = Infinity, elMax = -Infinity;
        for (let i = 0; i < nReturns; i++) {
          const e = elevation[i];
          if (Number.isFinite(e)) {
            if (e < elMin) elMin = e;
            if (e > elMax) elMax = e;
          }
        }
        const elRange = elMax > elMin ? elMax - elMin : 1.0;
        const stripX = marginLeft + plotW + 2;
        const stripImage = ctx.createImageData(elevStripW - 4, plotH);
        const sp = stripImage.data;
        const sw = elevStripW - 4;
        for (let py = 0; py < plotH; py++) {
          const row = Math.floor((py / plotH) * nReturns);
          const t = (elevation[row] - elMin) / elRange;
          const [r, g, b] = elevationColor(t);
          for (let px = 0; px < sw; px++) {
            const off = (py * sw + px) * 4;
            sp[off] = r;
            sp[off + 1] = g;
            sp[off + 2] = b;
            sp[off + 3] = 255;
          }
        }
        ctx.putImageData(stripImage, stripX, marginTop);

        // Elevation ticks
        ctx.save();
        ctx.fillStyle = "rgba(17, 24, 39, 0.85)";
        ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
        ctx.fillText(`${elMax.toFixed(1)}°`, stripX, marginTop - 2);
        ctx.fillText(`${elMin.toFixed(1)}°`, stripX, marginTop + plotH + 10);
        ctx.fillText("El", stripX + sw / 2 - 4, marginTop + plotH + 20);
        ctx.restore();
      }

      // --- X-axis: Range in km ---
      // The axis is the median range ramp; rows whose own base or step differ
      // still report their exact range on hover.
      const axisStartM = Number(meta.axis_start_m) || 0;
      const axisStepM = Number(meta.axis_step_m) || 1;
      const nRangeOrig = Number(meta.n_range_orig) || nRange;
      const rangeStride = Number(meta.range_stride) || 1;

      ctx.save();
      ctx.strokeStyle = "rgba(110, 118, 129, 0.5)";
      ctx.fillStyle = "rgba(17, 24, 39, 0.85)";
      ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
      ctx.lineWidth = 1;

      const rangeMinKm = axisStartM / 1000;
      const rangeMaxKm = (axisStartM + (nRange - 1) * axisStepM * rangeStride) / 1000;
      const xTicks = niceTicks(rangeMinKm, rangeMaxKm, 8);
      for (const km of xTicks) {
        const col = ((km * 1000 - axisStartM) / (axisStepM * rangeStride)) / nRange;
        const px = marginLeft + col * plotW;
        if (px < marginLeft || px > marginLeft + plotW) continue;
        ctx.beginPath();
        ctx.moveTo(px, marginTop + plotH);
        ctx.lineTo(px, marginTop + plotH + 4);
        ctx.stroke();
        ctx.fillText(km.toFixed(0), px - 8, marginTop + plotH + 14);
      }
      ctx.fillText("Range (km)", marginLeft + plotW / 2 - 30, height - 4);

      // --- Y-axis: Time with sweep labels ---
      const nReturnsOrig = Number(meta.n_returns_orig) || nReturns;

      // Tick at each sweep boundary
      for (let i = 0; i < sweepBoundaries.length; i++) {
        const bndIdx = sweepBoundaries[i];
        const py = marginTop + (bndIdx / nReturns) * plotH;
        ctx.beginPath();
        ctx.moveTo(marginLeft - 4, py);
        ctx.lineTo(marginLeft, py);
        ctx.stroke();
        const swpNum = bndIdx < sweepNumber.length ? sweepNumber[bndIdx] : "?";
        ctx.fillText(`S${swpNum}`, 4, py + 4);
      }
      // First sweep label
      if (sweepNumber.length > 0) {
        ctx.fillText(`S${sweepNumber[0]}`, 4, marginTop + 10);
      }

      // Time ticks on y-axis
      if (returnTimeMs.length >= nReturns && nReturns > 0) {
        const t0 = returnTimeMs[0];
        const t1 = returnTimeMs[nReturns - 1];
        const tRange = t1 - t0;
        if (tRange > 0) {
          // Pick nice tick interval in seconds
          const tRangeSec = tRange / 1000;
          const niceIntervals = [1, 2, 5, 10, 15, 30, 60, 120, 300];
          let stepSec = 60;
          for (const s of niceIntervals) {
            if (tRangeSec / s <= 8) { stepSec = s; break; }
          }
          const stepMs = stepSec * 1000;
          const startTick = Math.ceil(t0 / stepMs) * stepMs;
          for (let t = startTick; t <= t1; t += stepMs) {
            const frac = (t - t0) / tRange;
            const py = marginTop + frac * plotH;
            if (py < marginTop || py > marginTop + plotH) continue;
            ctx.beginPath();
            ctx.moveTo(marginLeft - 4, py);
            ctx.lineTo(marginLeft, py);
            ctx.stroke();
            ctx.fillText(formatTimeUTC(t), 14, py + 4);
          }
        }
        // Always label first and last time at edges
        ctx.fillText(formatTimeUTC(t0), 14, marginTop + 10);
        if (t1 !== t0) {
          ctx.fillText(formatTimeUTC(t1), 14, marginTop + plotH - 2);
        }
      }

      // Border around plot area
      ctx.strokeStyle = "rgba(110, 118, 129, 0.7)";
      ctx.strokeRect(marginLeft, marginTop, plotW, plotH);

      ctx.restore();

      // --- Overlay text ---
      const moment = String(meta.moment || "");
      const label = `Waterfall | moment=${moment} | ${nReturns}\u00d7${nRange} (from ${nReturnsOrig}\u00d7${nRangeOrig})`;
      ctx.save();
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(17, 24, 39, 0.75)";
      ctx.fillRect(marginLeft, 4, tw + 12, 20);
      ctx.fillStyle = "#f9fafb";
      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
      ctx.fillText(label, marginLeft + 6, 18);
      ctx.restore();

      // --- Colorbar in the reserved right gutter ---
      const cbH = Math.max(120, Math.min(300, plotH));
      drawColorbar(ctx, lut, meta, width - colorbarW - 10, marginTop, colorbarW, cbH);
    }

    function hideTooltip() {
      tooltip.style.display = "none";
      if (lastHoverKey !== "") {
        lastHoverKey = "";
        model.set("hover", {});
        model.save_changes();
      }
    }

    function onMove(event) {
      const meta = model.get("meta") || {};
      const nReturns = Number(meta.n_returns) || 0;
      const nRange = Number(meta.n_range) || 0;
      if (nReturns === 0 || nRange === 0 || grid.length === 0) {
        hideTooltip();
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const px = event.clientX - rect.left;
      const py = event.clientY - rect.top;
      const width = canvas.width;
      const height = canvas.height;
      const plotW = width - marginLeft - marginRight - elevStripW;
      const plotH = height - marginTop - marginBottom;

      if (px < marginLeft || px > marginLeft + plotW || py < marginTop || py > marginTop + plotH) {
        hideTooltip();
        return;
      }

      const col = Math.floor(((px - marginLeft) / plotW) * nRange);
      const row = Math.floor(((py - marginTop) / plotH) * nReturns);
      if (row < 0 || row >= nReturns || col < 0 || col >= nRange) {
        hideTooltip();
        return;
      }

      const val = dequantize(grid[row * nRange + col], meta.vmin, meta.vmax);
      const az = row < azimuth.length ? azimuth[row] : NaN;
      const el = row < elevation.length ? elevation[row] : NaN;
      const swp = row < sweepNumber.length ? sweepNumber[row] : -1;

      const rangeStride = Number(meta.range_stride) || 1;
      const gateIdx = col * rangeStride;
      // This row's own range ramp, not the median one the axis is drawn on.
      const rowBase = row < baseRange.length ? baseRange[row] : Number(meta.axis_start_m) || 0;
      const rowStep = row < rangeStep.length ? rangeStep[row] : Number(meta.axis_step_m) || 1;
      const rangeKm = (rowBase + gateIdx * rowStep) / 1000;

      const returnStride = Number(meta.return_stride) || 1;
      const origRetIdx = row * returnStride;

      const tMs = row < returnTimeMs.length ? returnTimeMs[row] : NaN;
      const timeStr = Number.isFinite(tMs) ? formatTimeUTC(tMs) : "?";
      const valStr = Number.isFinite(val) ? val.toFixed(2) : "NaN";
      tooltip.style.left = `${Math.max(8, Math.round(px + 10))}px`;
      tooltip.style.top = `${Math.max(8, Math.round(py + 10))}px`;
      tooltip.style.display = "block";
      tooltip.textContent =
        `value=${valStr} time=${timeStr}Z ` +
        `az=${az.toFixed(2)}° el=${el.toFixed(2)}° sweep=${swp} range=${rangeKm.toFixed(2)}km`;

      const hoverKey = `${row},${col}`;
      if (hoverKey !== lastHoverKey) {
        lastHoverKey = hoverKey;
        model.set("hover", {
          row,
          col,
          value: Number(val),
          return_index: origRetIdx,
          gate_index: gateIdx,
          azimuth_deg: Number(az),
          elevation_deg: Number(el),
          sweep_number: Number(swp),
          range_km: rangeKm,
          time_utc: timeStr,
        });
        model.save_changes();
      }
    }

    function onLeave() {
      hideTooltip();
    }

    const watched = [
      "width", "height", "meta", "grid_bytes",
      "azimuth_bytes", "elevation_bytes", "return_time_ms_bytes",
      "sweep_number_bytes", "base_range_bytes", "range_step_bytes",
      "sweep_boundary_bytes",
    ];
    for (const key of watched) {
      model.on(`change:${key}`, redraw);
    }

    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", onLeave);

    redraw();

    return () => {
      for (const key of watched) {
        model.off(`change:${key}`, redraw);
      }
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mouseleave", onLeave);
    };
  },
};
