function drawGuides(ctx, width, height, maxR) {
  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.max(8, Math.min(width, height) / 2 - 24);

  ctx.save();
  ctx.strokeStyle = "rgba(110, 118, 129, 0.5)";
  ctx.lineWidth = 1;
  for (let ring = 1; ring <= 4; ring += 1) {
    const rr = radius * (ring / 4.0);
    ctx.beginPath();
    ctx.arc(cx, cy, rr, 0, Math.PI * 2);
    ctx.stroke();
  }

  ctx.beginPath();
  ctx.moveTo(cx - radius, cy);
  ctx.lineTo(cx + radius, cy);
  ctx.moveTo(cx, cy - radius);
  ctx.lineTo(cx, cy + radius);
  ctx.stroke();

  ctx.fillStyle = "rgba(17, 24, 39, 0.8)";
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
  ctx.fillText("N", cx + 4, cy - radius - 6);
  ctx.fillText("E", cx + radius + 6, cy + 4);
  ctx.fillText("S", cx + 4, cy + radius + 14);
  ctx.fillText("W", cx - radius - 14, cy + 4);

  if (Number.isFinite(maxR) && maxR > 0) {
    const km = (maxR / 1000.0).toFixed(1);
    ctx.fillText(`max range: ${km} km`, 12, 18);
  }
  ctx.restore();
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

    let pick = new Int32Array(0);
    let codes = new Uint16Array(0);
    let azimuth = new Float32Array(0);
    let elevation = new Float32Array(0);
    let baseRange = new Float32Array(0);
    let rangeStep = new Float32Array(0);
    let returnIndex = new Uint32Array(0);
    let returnTime = new Float64Array(0);
    let nReturns = 0;
    let nGates = 0;
    let gateStride = 1;
    let lastHover = -1;

    /** Slant range of the gate in `row`, column `col` of the shipped grid. */
    function cellRangeM(row, col) {
      return baseRange[row] + col * gateStride * rangeStep[row];
    }

    function redraw() {
      const width = Number(model.get("width")) || 760;
      const height = Number(model.get("height")) || 760;
      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, width, height);

      codes = decodeArray(model.get("value_bytes"), Uint16Array);
      azimuth = decodeArray(model.get("azimuth_bytes"), Float32Array);
      elevation = decodeArray(model.get("elevation_bytes"), Float32Array);
      baseRange = decodeArray(model.get("base_range_bytes"), Float32Array);
      rangeStep = decodeArray(model.get("range_step_bytes"), Float32Array);
      returnIndex = decodeArray(model.get("return_index_bytes"), Uint32Array);
      returnTime = decodeArray(model.get("return_time_ms_bytes"), Float64Array);

      const meta = model.get("meta") || {};
      const lut = buildLut(String(meta.colormap || "viridis"));
      // Colorbar hugs the right edge, clear of the compass labels.
      const cbW = 66;
      const cbH = Math.max(120, Math.min(260, height - 120));
      const cbX = width - cbW - 10;
      const cbY = 32;

      gateStride = Number(meta.gate_stride) || 1;
      nGates = Number(meta.n_gates) || 0;
      // Every per-return buffer must cover the rows the grid claims, or the
      // geometry for a row would be read out of an array that ended earlier.
      nReturns = Math.min(
        Number(meta.n_returns) || 0,
        azimuth.length,
        elevation.length,
        baseRange.length,
        rangeStep.length,
        returnIndex.length,
        returnTime.length,
      );
      if (nGates > 0) {
        nReturns = Math.min(nReturns, Math.floor(codes.length / nGates));
      }

      if (nReturns === 0 || nGates === 0) {
        // Drop the previous frame's pick buffer, otherwise hover keeps
        // resolving indices into arrays that are now empty.
        pick = new Int32Array(0);
        nReturns = 0;
        nGates = 0;
        lastHover = -1;
        drawGuides(ctx, width, height, Number(meta.max_range_m) || 0);
        drawColorbar(ctx, lut, meta, cbX, cbY, cbW, cbH);
        return;
      }

      const maxRange = Math.max(1.0, Number(meta.max_range_m) || 1.0);

      const cx = width / 2;
      const cy = height / 2;
      const radiusPx = Math.max(8, Math.min(width, height) / 2 - 24);

      const image = ctx.createImageData(width, height);
      const pixels = image.data;
      pick = new Int32Array(width * height);
      pick.fill(-1);

      // Coordinates are rebuilt per row rather than shipped per gate: the whole
      // row shares one azimuth, and range walks the row in equal steps.
      for (let row = 0; row < nReturns; row += 1) {
        const theta = ((90.0 - azimuth[row]) * Math.PI) / 180.0;
        const cosT = Math.cos(theta);
        const sinT = Math.sin(theta);
        const base = baseRange[row];
        const step = rangeStep[row] * gateStride;
        const rowOffset = row * nGates;
        for (let col = 0; col < nGates; col += 1) {
          const code = codes[rowOffset + col];
          if (code === QUANT_NAN) {
            continue;
          }
          const rr = ((base + col * step) / maxRange) * radiusPx;
          const x = Math.round(cx + rr * cosT);
          const y = Math.round(cy - rr * sinT);
          if (x < 0 || x >= width || y < 0 || y >= height) {
            continue;
          }

          const [r, g, b] = lutColor(lut, quantFraction(code));
          const pxOffset = (y * width + x) * 4;
          pixels[pxOffset] = r;
          pixels[pxOffset + 1] = g;
          pixels[pxOffset + 2] = b;
          pixels[pxOffset + 3] = 255;
          pick[y * width + x] = rowOffset + col;
        }
      }

      ctx.putImageData(image, 0, 0);
      drawGuides(ctx, width, height, maxRange);
      drawColorbar(ctx, lut, meta, cbX, cbY, cbW, cbH);
    }

    function hideTooltip() {
      tooltip.style.display = "none";
      if (lastHover !== -1) {
        lastHover = -1;
        model.set("hover", {});
        model.save_changes();
      }
    }

    function onMove(event) {
      if (pick.length === 0) {
        hideTooltip();
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const width = canvas.width;
      const height = canvas.height;
      const idx = nearestPick(pick, width, height, x, y);
      if (idx < 0) {
        hideTooltip();
        return;
      }

      const meta = model.get("meta") || {};
      const row = Math.floor(idx / nGates);
      const col = idx - row * nGates;
      const value = dequantize(codes[idx], meta.vmin, meta.vmax);
      const rangeM = cellRangeM(row, col);
      const gate = col * gateStride;
      const ret = returnIndex[row];
      const timeMs = returnTime[row];
      const iso = Number.isFinite(timeMs) ? new Date(timeMs).toISOString() : "n/a";
      tooltip.style.left = `${Math.max(8, Math.round(x + 10))}px`;
      tooltip.style.top = `${Math.max(8, Math.round(y + 10))}px`;
      tooltip.style.display = "block";
      tooltip.textContent =
        `value=${value.toFixed(2)} az=${azimuth[row].toFixed(2)}deg ` +
        `r=${(rangeM / 1000.0).toFixed(2)}km gate=${gate} ` +
        `ret=${ret} el=${elevation[row].toFixed(2)}deg t=${iso}`;

      if (idx !== lastHover) {
        lastHover = idx;
        model.set("hover", {
          index: idx,
          value: Number(value),
          azimuth_deg: Number(azimuth[row]),
          range_m: Number(rangeM),
          elevation_deg: Number(elevation[row]),
          gate_index: Number(gate),
          return_index: Number(ret),
          return_time_ms: Number(timeMs),
          return_time_iso: iso,
        });
        model.save_changes();
      }
    }

    function onLeave() {
      hideTooltip();
    }

    const watched = [
      "width",
      "height",
      "meta",
      "value_bytes",
      "azimuth_bytes",
      "elevation_bytes",
      "base_range_bytes",
      "range_step_bytes",
      "return_index_bytes",
      "return_time_ms_bytes",
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
