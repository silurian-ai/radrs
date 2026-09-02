function drawGridImage(ctx, grid, nRows, nCols, marginLeft, marginTop, plotW, plotH, lut) {
  if (nRows === 0 || nCols === 0) return;
  const image = ctx.createImageData(plotW, plotH);
  const pixels = image.data;

  for (let py = 0; py < plotH; py++) {
    const row = Math.floor((py / plotH) * nRows);
    for (let px = 0; px < plotW; px++) {
      const col = Math.floor((px / plotW) * nCols);
      const code = grid[row * nCols + col];
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

function drawAxes(ctx, meta, marginLeft, marginTop, plotW, plotH, canvasW, canvasH) {
  const xMin = Number(meta.x_min);
  const xMax = Number(meta.x_max);
  const yMin = Number(meta.y_min);
  const yMax = Number(meta.y_max);
  const xLabel = String(meta.x_label || "");
  const yLabel = String(meta.y_label || "");

  ctx.save();
  ctx.strokeStyle = "rgba(110, 118, 129, 0.5)";
  ctx.fillStyle = "rgba(17, 24, 39, 0.85)";
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
  ctx.lineWidth = 1;

  // X-axis ticks (bottom)
  const xTicks = niceTicks(xMin / 1000, xMax / 1000, 8);
  const marginBottom = canvasH - marginTop - plotH;
  for (const km of xTicks) {
    const px = marginLeft + ((km * 1000 - xMin) / (xMax - xMin)) * plotW;
    ctx.beginPath();
    ctx.moveTo(px, marginTop + plotH);
    ctx.lineTo(px, marginTop + plotH + 4);
    ctx.stroke();
    ctx.fillText(km.toFixed(0), px - 8, marginTop + plotH + 14);
  }
  ctx.fillText(xLabel, marginLeft + plotW / 2 - 30, canvasH - 4);

  // Y-axis ticks (left)
  const yTicks = niceTicks(yMin / 1000, yMax / 1000, 8);
  for (const km of yTicks) {
    const py = marginTop + (1 - (km * 1000 - yMin) / (yMax - yMin)) * plotH;
    ctx.beginPath();
    ctx.moveTo(marginLeft - 4, py);
    ctx.lineTo(marginLeft, py);
    ctx.stroke();
    ctx.fillText(km.toFixed(0), 4, py + 4);
  }
  ctx.save();
  ctx.translate(10, marginTop + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, -30, 0);
  ctx.restore();

  // Border around plot area
  ctx.strokeStyle = "rgba(110, 118, 129, 0.7)";
  ctx.strokeRect(marginLeft, marginTop, plotW, plotH);

  ctx.restore();
}

function drawCappiGuides(ctx, meta, marginLeft, marginTop, plotW, plotH) {
  const xMin = Number(meta.x_min);
  const xMax = Number(meta.x_max);
  const yMin = Number(meta.y_min);
  const yMax = Number(meta.y_max);

  ctx.save();

  // Range rings centered on radar (x=0, y=0)
  const cx = marginLeft + ((0 - xMin) / (xMax - xMin)) * plotW;
  const cy = marginTop + ((yMax - 0) / (yMax - yMin)) * plotH;
  const maxExtent = Math.max(Math.abs(xMin), xMax, Math.abs(yMin), yMax);
  const ringStep = maxExtent > 200000 ? 100000 : maxExtent > 50000 ? 50000 : 10000;

  ctx.strokeStyle = "rgba(110, 118, 129, 0.25)";
  ctx.lineWidth = 0.5;
  for (let r = ringStep; r < maxExtent * 1.5; r += ringStep) {
    const rpx = (r / (xMax - xMin)) * plotW;
    ctx.beginPath();
    ctx.arc(cx, cy, rpx, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Crosshairs through radar
  ctx.strokeStyle = "rgba(110, 118, 129, 0.35)";
  ctx.lineWidth = 0.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(marginLeft, cy);
  ctx.lineTo(marginLeft + plotW, cy);
  ctx.moveTo(cx, marginTop);
  ctx.lineTo(cx, marginTop + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  // N/E/S/W labels
  ctx.fillStyle = "rgba(17, 24, 39, 0.7)";
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
  ctx.fillText("N", cx + 2, marginTop + 10);
  ctx.fillText("S", cx + 2, marginTop + plotH - 4);
  ctx.fillText("E", marginLeft + plotW - 10, cy - 4);
  ctx.fillText("W", marginLeft + 2, cy - 4);

  ctx.restore();
}

function drawXsecGuides(ctx, meta, marginLeft, marginTop, plotW, plotH) {
  const xMin = Number(meta.x_min);
  const xMax = Number(meta.x_max);
  const yMin = Number(meta.y_min);
  const yMax = Number(meta.y_max);

  ctx.save();
  ctx.strokeStyle = "rgba(110, 118, 129, 0.3)";
  ctx.lineWidth = 0.5;
  ctx.setLineDash([4, 4]);

  // Horizontal lines at round km altitudes
  const altTicks = niceTicks(yMin / 1000, yMax / 1000, 6);
  for (const km of altTicks) {
    const py = marginTop + (1 - (km * 1000 - yMin) / (yMax - yMin)) * plotH;
    ctx.beginPath();
    ctx.moveTo(marginLeft, py);
    ctx.lineTo(marginLeft + plotW, py);
    ctx.stroke();
  }

  // Vertical lines at round km ranges
  const rngTicks = niceTicks(xMin / 1000, xMax / 1000, 8);
  for (const km of rngTicks) {
    const px = marginLeft + ((km * 1000 - xMin) / (xMax - xMin)) * plotW;
    ctx.beginPath();
    ctx.moveTo(px, marginTop);
    ctx.lineTo(px, marginTop + plotH);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // Emphasized vertical line at range=0 (the radar)
  const radarX = marginLeft + ((0 - xMin) / (xMax - xMin)) * plotW;
  ctx.strokeStyle = "rgba(220, 38, 38, 0.5)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(radarX, marginTop);
  ctx.lineTo(radarX, marginTop + plotH);
  ctx.stroke();

  ctx.restore();
}

/**
 * Per-sweep beam height at the far edge of the plot, in meters.
 *
 * Prefers the values `GridPayload` ships, which come out of the same 4/3
 * effective-earth model as the gate coordinates. Falls back to computing them
 * here for payloads built before the field existed. Either way it is the curved
 * height, not the flat-earth `range * sin(elevation)`, which understates the
 * top of a sweep by kilometres at long range.
 */
function sweepMaxAltitudes(meta) {
  const elevations = meta.sweep_elevations_deg || [];
  const shipped = meta.sweep_max_altitude_m;
  if (Array.isArray(shipped) && shipped.length === elevations.length) {
    return shipped.map(Number);
  }
  const rangeM = Math.abs(Number(meta.x_max)) || 1;
  return Array.from(elevations, (el) => beamHeightM(rangeM, el));
}

/** Does this sweep reach into the CAPPI's altitude band? */
function sweepIsActive(mode, elevationDeg, maxAltitudeM, cappiAltM, cappiTolM) {
  if (mode !== "cappi") return true;
  return (
    elevationDeg > 0 &&
    cappiAltM - cappiTolM < maxAltitudeM &&
    cappiAltM + cappiTolM > 0
  );
}

function drawElevationDiagram(ctx, meta, canvasW, canvasH) {
  const sweepElevations = meta.sweep_elevations_deg;
  if (!sweepElevations || sweepElevations.length === 0) return;

  const dw = 160;
  const dh = 100;
  const dx = canvasW - dw - 12;
  const dy = canvasH - dh - 12;

  ctx.save();
  // Background panel
  ctx.fillStyle = "rgba(17, 24, 39, 0.80)";
  ctx.beginPath();
  ctx.roundRect(dx, dy, dw, dh, 4);
  ctx.fill();

  const mode = String(meta.grid_mode);
  const ox = dx + 10;
  const oy = dy + dh - 10;
  const maxLen = dw - 20;
  const maxAlt = dh - 20;

  // Determine active sweeps
  const cappiAlt = Number(meta.cappi_altitude_m) || 0;
  const cappiTol = Number(meta.cappi_tolerance_m) || 0;
  const maxAltitudes = sweepMaxAltitudes(meta);

  for (let i = 0; i < sweepElevations.length; i++) {
    const el = sweepElevations[i];
    const rad = (el * Math.PI) / 180;
    const active = sweepIsActive(mode, el, maxAltitudes[i], cappiAlt, cappiTol);

    ctx.strokeStyle = active ? "rgba(34, 197, 94, 0.8)" : "rgba(110, 118, 129, 0.5)";
    ctx.lineWidth = active ? 1.5 : 0.8;
    ctx.beginPath();
    ctx.moveTo(ox, oy);
    const endX = ox + maxLen * Math.cos(rad);
    const endY = oy - maxLen * Math.sin(rad);
    ctx.lineTo(endX, endY);
    ctx.stroke();
  }

  // CAPPI altitude line, placed on the curved-height scale the beams reach.
  if (mode === "cappi") {
    const topAltM = Math.max(1, ...maxAltitudes.filter(Number.isFinite));
    const altPx = Math.min((cappiAlt / topAltM) * maxAlt, maxAlt);
    ctx.strokeStyle = "rgba(253, 231, 37, 0.7)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(ox, oy - altPx);
    ctx.lineTo(ox + maxLen, oy - altPx);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Label
  ctx.fillStyle = "rgba(249, 250, 251, 0.8)";
  ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
  ctx.fillText("Elevation beams", dx + 6, dy + 12);

  ctx.restore();
}

function drawAzimuthIndicator(ctx, meta, canvasW) {
  const mode = String(meta.grid_mode);
  const dw = 80;
  const dh = 80;
  const dx = canvasW - dw - 12;
  const dy = 12;

  ctx.save();
  ctx.fillStyle = "rgba(17, 24, 39, 0.80)";
  ctx.beginPath();
  ctx.roundRect(dx, dy, dw, dh, 4);
  ctx.fill();

  const cx = dx + dw / 2;
  const cy = dy + dh / 2 + 4;
  const r = 28;

  ctx.strokeStyle = "rgba(110, 118, 129, 0.5)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();

  // N marker
  ctx.fillStyle = "rgba(249, 250, 251, 0.8)";
  ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
  ctx.fillText("N", cx - 3, dy + 10);

  if (mode === "cappi") {
    // Full circle fill for 360-degree coverage
    ctx.fillStyle = "rgba(34, 197, 94, 0.2)";
    ctx.beginPath();
    ctx.arc(cx, cy, r - 2, 0, Math.PI * 2);
    ctx.fill();
  } else if (mode === "xsec") {
    // Line through center at target azimuth
    const az = Number(meta.xsec_azimuth_deg) || 0;
    const rad = ((90 - az) * Math.PI) / 180;
    ctx.strokeStyle = "rgba(253, 231, 37, 0.8)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx - r * Math.cos(rad), cy + r * Math.sin(rad));
    ctx.lineTo(cx + r * Math.cos(rad), cy - r * Math.sin(rad));
    ctx.stroke();
    ctx.fillStyle = "rgba(253, 231, 37, 0.9)";
    ctx.fillText(`${az.toFixed(0)}°`, dx + 4, dy + dh - 6);
  }

  ctx.restore();
}

function drawSweepInfo(ctx, meta, marginLeft) {
  const sweepElevations = meta.sweep_elevations_deg;
  const sweepCounts = meta.sweep_num_returns;
  if (!sweepElevations || sweepElevations.length === 0) return;

  const mode = String(meta.grid_mode);
  const cappiAlt = Number(meta.cappi_altitude_m) || 0;
  const cappiTol = Number(meta.cappi_tolerance_m) || 0;
  const maxAltitudes = sweepMaxAltitudes(meta);

  ctx.save();
  ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";

  let y = 40;
  const x = marginLeft + 4;

  // Build sweep labels, max 2 lines
  const parts = [];
  for (let i = 0; i < sweepElevations.length; i++) {
    const el = sweepElevations[i];
    const cnt = sweepCounts ? sweepCounts[i] : "?";
    const active = sweepIsActive(mode, el, maxAltitudes[i], cappiAlt, cappiTol);
    parts.push({ text: `${el.toFixed(1)}°(${cnt})`, active });
  }

  // Render up to 2 lines, 10 items per line
  const perLine = 10;
  for (let line = 0; line < 2 && line * perLine < parts.length; line++) {
    let lineX = x;
    const start = line * perLine;
    const end = Math.min(start + perLine, parts.length);
    for (let i = start; i < end; i++) {
      const p = parts[i];
      ctx.fillStyle = p.active ? "rgba(34, 197, 94, 0.9)" : "rgba(110, 118, 129, 0.7)";
      ctx.fillText(p.text, lineX, y);
      lineX += ctx.measureText(p.text).width + 6;
    }
    if (end < parts.length && line === 1) {
      ctx.fillStyle = "rgba(110, 118, 129, 0.7)";
      ctx.fillText("...", lineX, y);
    }
    y += 12;
  }

  ctx.restore();
}

function drawModeOverlay(ctx, meta, marginLeft) {
  const mode = String(meta.grid_mode);
  const moment = String(meta.moment || "");

  ctx.save();
  ctx.fillStyle = "rgba(17, 24, 39, 0.75)";
  ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";

  let label;
  if (mode === "cappi") {
    const altKm = ((Number(meta.cappi_altitude_m) || 0) / 1000).toFixed(1);
    const tolKm = ((Number(meta.cappi_tolerance_m) || 0) / 1000).toFixed(1);
    label = `CAPPI alt=${altKm}km | moment=${moment} | tol=${tolKm}km`;
  } else {
    const az = (Number(meta.xsec_azimuth_deg) || 0).toFixed(1);
    label = `Cross-section az=${az}° | moment=${moment}`;
  }

  // Background for overlay text
  const tw = ctx.measureText(label).width;
  ctx.fillStyle = "rgba(17, 24, 39, 0.75)";
  ctx.fillRect(marginLeft, 4, tw + 12, 20);
  ctx.fillStyle = "#f9fafb";
  ctx.fillText(label, marginLeft + 6, 18);

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

    let grid = new Uint16Array(0);
    let lastHover = "";

    const marginLeft = 50;
    const marginTop = 20;
    const colorbarW = 66;
    // Right margin reserves the colorbar gutter so the plot never runs under it.
    const marginRight = 20 + colorbarW;
    const marginBottom = 40;

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

      const nRows = Number(meta.n_rows) || 0;
      const nCols = Number(meta.n_cols) || 0;
      const mode = String(meta.grid_mode || "cappi");
      const lut = buildLut(String(meta.colormap || "viridis"));

      const plotW = width - marginLeft - marginRight;
      const plotH = height - marginTop - marginBottom;

      if (nRows > 0 && nCols > 0 && grid.length >= nRows * nCols) {
        drawGridImage(ctx, grid, nRows, nCols, marginLeft, marginTop, plotW, plotH, lut);
      }

      // Axes
      drawAxes(ctx, meta, marginLeft, marginTop, plotW, plotH, width, height);

      // Mode-specific guides
      if (mode === "cappi") {
        drawCappiGuides(ctx, meta, marginLeft, marginTop, plotW, plotH);
      } else {
        drawXsecGuides(ctx, meta, marginLeft, marginTop, plotW, plotH);
      }

      // Overlay diagrams
      drawModeOverlay(ctx, meta, marginLeft);
      drawSweepInfo(ctx, meta, marginLeft);
      drawElevationDiagram(ctx, meta, width, height);
      drawAzimuthIndicator(ctx, meta, width);

      // Slots between the azimuth indicator (top right) and the elevation
      // diagram (bottom right), both of which hug the same edge.
      const cbTop = 104;
      const cbH = Math.max(120, Math.min(300, height - 120 - cbTop));
      drawColorbar(ctx, lut, meta, width - colorbarW - 10, cbTop, colorbarW, cbH);
    }

    function hideTooltip() {
      tooltip.style.display = "none";
      if (lastHover !== "") {
        lastHover = "";
        model.set("hover", {});
        model.save_changes();
      }
    }

    function onMove(event) {
      const meta = model.get("meta") || {};
      const nRows = Number(meta.n_rows) || 0;
      const nCols = Number(meta.n_cols) || 0;
      if (nRows === 0 || nCols === 0 || grid.length === 0) {
        hideTooltip();
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const px = event.clientX - rect.left;
      const py = event.clientY - rect.top;
      const width = canvas.width;
      const height = canvas.height;
      const plotW = width - marginLeft - marginRight;
      const plotH = height - marginTop - marginBottom;

      // Check if inside plot area
      if (px < marginLeft || px > marginLeft + plotW || py < marginTop || py > marginTop + plotH) {
        hideTooltip();
        return;
      }

      const col = Math.floor(((px - marginLeft) / plotW) * nCols);
      const row = Math.floor(((py - marginTop) / plotH) * nRows);
      if (row < 0 || row >= nRows || col < 0 || col >= nCols) {
        hideTooltip();
        return;
      }

      const val = dequantize(grid[row * nCols + col], meta.vmin, meta.vmax);
      if (!Number.isFinite(val)) {
        hideTooltip();
        return;
      }

      const xMin = Number(meta.x_min);
      const xMax = Number(meta.x_max);
      const yMin = Number(meta.y_min);
      const yMax = Number(meta.y_max);
      const worldX = xMin + (col + 0.5) / nCols * (xMax - xMin);
      const worldY = yMax - (row + 0.5) / nRows * (yMax - yMin);
      const mode = String(meta.grid_mode || "cappi");

      let tooltipText;
      if (mode === "cappi") {
        const eastKm = (worldX / 1000).toFixed(2);
        const northKm = (worldY / 1000).toFixed(2);
        const rangeKm = (Math.sqrt(worldX * worldX + worldY * worldY) / 1000).toFixed(2);
        tooltipText = `value=${val.toFixed(2)} east=${eastKm}km north=${northKm}km range=${rangeKm}km`;
      } else {
        const grndKm = (worldX / 1000).toFixed(2);
        const altKm = (worldY / 1000).toFixed(2);
        tooltipText = `value=${val.toFixed(2)} ground_range=${grndKm}km altitude=${altKm}km`;
      }

      tooltip.style.left = `${Math.max(8, Math.round(px + 10))}px`;
      tooltip.style.top = `${Math.max(8, Math.round(py + 10))}px`;
      tooltip.style.display = "block";
      tooltip.textContent = tooltipText;

      const hoverKey = `${row},${col}`;
      if (hoverKey !== lastHover) {
        lastHover = hoverKey;
        const hover = { row, col, value: Number(val) };
        if (mode === "cappi") {
          hover.east_m = Number(worldX);
          hover.north_m = Number(worldY);
          hover.range_m = Math.sqrt(worldX * worldX + worldY * worldY);
        } else {
          hover.ground_range_m = Number(worldX);
          hover.altitude_m = Number(worldY);
        }
        model.set("hover", hover);
        model.save_changes();
      }
    }

    function onLeave() {
      hideTooltip();
    }

    const watched = ["width", "height", "meta", "grid_bytes"];
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
