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
    let vcpIndex = new Uint16Array(0);
    let vcpTimeMs = new Float64Array(0);
    let sweepNumber = new Uint16Array(0);
    let azimuth = new Float32Array(0);
    let elevation = new Float32Array(0);
    let returnTimeMs = new Float64Array(0);
    let baseRange = new Float32Array(0);
    let rangeStep = new Float32Array(0);
    let foldIndex = new Uint16Array(0);
    let groupStarts = new Int32Array(0);

    let lastHoverKey = "";

    // Label gutter columns: VCP | SWP | TIME. Range is per-gate, not per-row, so it
    // would be misleading as a row label — it lives in the tooltip instead.
    const colVcp = 6;
    const colSwp = 34;
    const colTime = 62;
    const marginLeft = 148;
    const marginTop = 34;
    const colorbarW = 66;
    // Right margin reserves the colorbar gutter so the plot never runs under it.
    const marginRight = 16 + colorbarW;
    const marginBottom = 34;
    const rowFont = "10px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";

    /** Physical range of a gate: base_range + gate * range_step. */
    function gateRangeM(row, gateIdx) {
      return baseRange[row] + gateIdx * rangeStep[row];
    }

    function redraw() {
      const width = Number(model.get("width")) || 900;
      const height = Number(model.get("height")) || 900;
      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, width, height);

      const meta = model.get("meta") || {};
      grid = decodeArray(model.get("grid_bytes"), Uint16Array);
      vcpIndex = decodeArray(model.get("vcp_index_bytes"), Uint16Array);
      vcpTimeMs = decodeArray(model.get("vcp_time_ms_bytes"), Float64Array);
      sweepNumber = decodeArray(model.get("sweep_number_bytes"), Uint16Array);
      azimuth = decodeArray(model.get("azimuth_bytes"), Float32Array);
      elevation = decodeArray(model.get("elevation_bytes"), Float32Array);
      returnTimeMs = decodeArray(model.get("return_time_ms_bytes"), Float64Array);
      baseRange = decodeArray(model.get("base_range_bytes"), Float32Array);
      rangeStep = decodeArray(model.get("range_step_bytes"), Float32Array);
      foldIndex = decodeArray(model.get("fold_index_bytes"), Uint16Array);
      groupStarts = decodeArray(model.get("group_start_bytes"), Int32Array);

      const nRows = Number(meta.n_rows) || 0;
      const nRange = Number(meta.n_range) || 0;
      const lut = buildLut(String(meta.colormap || "viridis"));
      const rangeStride = Number(meta.range_stride) || 1;
      const rowOffset = Number(meta.row_offset) || 0;
      const rowTotal = Number(meta.row_total) || nRows;

      const plotW = width - marginLeft - marginRight;
      const plotH = height - marginTop - marginBottom;
      if (plotH > 0) {
        const cbH = Math.max(120, Math.min(300, plotH));
        drawColorbar(ctx, lut, meta, width - colorbarW - 10, marginTop, colorbarW, cbH);
      }
      if (nRows === 0 || nRange === 0 || plotW <= 0 || plotH <= 0) return;

      const rowH = plotH / nRows;

      // --- Heatmap: one grid row per screen band, no row resampling loss ---
      if (grid.length >= nRows * nRange) {
        const image = ctx.createImageData(plotW, plotH);
        const pixels = image.data;
        for (let py = 0; py < plotH; py++) {
          const row = Math.min(nRows - 1, Math.floor(py / rowH));
          for (let px = 0; px < plotW; px++) {
            const col = Math.min(nRange - 1, Math.floor((px / plotW) * nRange));
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

      // --- Row labels: one per row when there is room, else on a period ---
      // Greedy 11 px spacing (a 10 px glyph plus leading). At 1 px per row this
      // degrades to every 11th row rather than overprinting.
      const labelPeriod = Math.max(1, Math.ceil(11 / Math.max(rowH, 0.001)));
      let lastLabelY = -1e9;
      let labelsDrawn = 0;

      ctx.save();
      ctx.font = rowFont;
      ctx.fillStyle = "rgba(17, 24, 39, 0.85)";

      for (let row = 0; row < nRows; row++) {
        const py = marginTop + row * rowH + Math.min(rowH, 9) * 0.85;
        if (py - lastLabelY < 11) continue;
        if (py < marginTop || py > marginTop + plotH) continue;

        ctx.fillText(String(vcpIndex[row]), colVcp, py);
        ctx.fillText(String(sweepNumber[row]), colSwp, py);
        ctx.fillText(formatTimeUTCms(returnTimeMs[row]), colTime, py);
        lastLabelY = py;
        labelsDrawn++;
      }
      ctx.restore();

      // --- Thin divider between every set of folded returns ---
      // Confined to the plot area: drawing across the gutter would mask the labels.
      ctx.save();
      ctx.strokeStyle = "rgba(148, 163, 184, 0.6)";
      ctx.lineWidth = 1;
      for (let i = 0; i < groupStarts.length; i++) {
        const py = Math.round(marginTop + groupStarts[i] * rowH) + 0.5;
        ctx.beginPath();
        ctx.moveTo(marginLeft, py);
        ctx.lineTo(marginLeft + plotW, py);
        ctx.stroke();
      }
      ctx.restore();

      // --- X-axis: gate index within the fold ---
      ctx.save();
      ctx.strokeStyle = "rgba(110, 118, 129, 0.5)";
      ctx.fillStyle = "rgba(17, 24, 39, 0.85)";
      ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
      ctx.lineWidth = 1;
      for (const gate of niceTicks(0, (nRange - 1) * rangeStride, 8)) {
        const px = marginLeft + (gate / rangeStride / nRange) * plotW;
        if (px < marginLeft || px > marginLeft + plotW) continue;
        ctx.beginPath();
        ctx.moveTo(px, marginTop + plotH);
        ctx.lineTo(px, marginTop + plotH + 4);
        ctx.stroke();
        ctx.fillText(String(Math.round(gate)), px - 8, marginTop + plotH + 14);
      }
      ctx.fillText(
        "Gate index within fold (hover for unfolded range)",
        marginLeft,
        height - 6,
      );

      // Gutter column headers
      ctx.fillText("VCP", colVcp, 14);
      ctx.fillText("SWP", colSwp, 14);
      ctx.fillText("TIME", colTime, 14);
      ctx.strokeRect(marginLeft, marginTop, plotW, plotH);
      ctx.restore();

      // --- Overlay text ---
      const moment = String(meta.moment || "");
      const nGroups = Number(meta.n_groups) || 0;
      const periodNote = labelPeriod > 1 ? ` | ${labelsDrawn}/${nRows} rows labelled` : "";
      const label =
        `Folded waterfall | moment=${moment} | rows ${rowOffset}-${rowOffset + nRows - 1}` +
        ` of ${rowTotal} | ${nGroups} fold sets | ${rowH.toFixed(2)} px/row${periodNote}`;
      ctx.save();
      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(17, 24, 39, 0.75)";
      ctx.fillRect(marginLeft, 4, tw + 12, 20);
      ctx.fillStyle = "#f9fafb";
      ctx.fillText(label, marginLeft + 6, 18);
      ctx.restore();
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
      const nRows = Number(meta.n_rows) || 0;
      const nRange = Number(meta.n_range) || 0;
      const rangeStride = Number(meta.range_stride) || 1;
      const rowOffset = Number(meta.row_offset) || 0;
      if (nRows === 0 || nRange === 0 || grid.length === 0) {
        hideTooltip();
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const px = event.clientX - rect.left;
      const py = event.clientY - rect.top;
      const plotW = canvas.width - marginLeft - marginRight;
      const plotH = canvas.height - marginTop - marginBottom;

      if (px < marginLeft || px > marginLeft + plotW || py < marginTop || py > marginTop + plotH) {
        hideTooltip();
        return;
      }

      const row = Math.floor(((py - marginTop) / plotH) * nRows);
      const col = Math.floor(((px - marginLeft) / plotW) * nRange);
      if (row < 0 || row >= nRows || col < 0 || col >= nRange) {
        hideTooltip();
        return;
      }

      const val = dequantize(grid[row * nRange + col], meta.vmin, meta.vmax);
      const gateIdx = col * rangeStride;
      const rangeKm = gateRangeM(row, gateIdx) / 1000;
      const valStr = Number.isFinite(val) ? val.toFixed(2) : "NaN";
      const timeStr = formatTimeUTCms(returnTimeMs[row]);
      const vcpStr = row < vcpTimeMs.length ? formatTimeUTCms(vcpTimeMs[row]) : "";

      tooltip.style.display = "block";
      tooltip.style.left = `${px + 12}px`;
      tooltip.style.top = `${py + 12}px`;
      tooltip.textContent =
        `value=${valStr} vcp=${vcpIndex[row]}@${vcpStr} sweep=${sweepNumber[row]} ` +
        `fold=${foldIndex[row]} gate=${gateIdx} range=${rangeKm.toFixed(2)}km ` +
        `az=${azimuth[row].toFixed(2)}° el=${elevation[row].toFixed(2)}° time=${timeStr}Z`;

      const hoverKey = `${row},${col}`;
      if (hoverKey !== lastHoverKey) {
        lastHoverKey = hoverKey;
        model.set("hover", {
          row,
          col,
          value: Number(val),
          return_index: rowOffset + row,
          vcp_index: Number(vcpIndex[row]),
          vcp_time_utc: vcpStr,
          sweep_number: Number(sweepNumber[row]),
          fold_index: Number(foldIndex[row]),
          gate_index: gateIdx,
          range_km: rangeKm,
          azimuth_deg: Number(azimuth[row]),
          elevation_deg: Number(elevation[row]),
          time_utc: timeStr,
        });
        model.save_changes();
      }
    }

    function onLeave() {
      hideTooltip();
    }

    const watched = [
      "width", "height", "meta", "grid_bytes", "vcp_index_bytes",
      "vcp_time_ms_bytes", "sweep_number_bytes", "azimuth_bytes",
      "elevation_bytes", "return_time_ms_bytes", "base_range_bytes",
      "range_step_bytes", "fold_index_bytes", "group_start_bytes",
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
