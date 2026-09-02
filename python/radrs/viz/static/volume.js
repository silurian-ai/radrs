import { COORDINATE_SYSTEM, Deck, LineLayer, OrbitView, PointCloudLayer } from "https://esm.sh/deck.gl@9.2.2?bundle";

/**
 * Assemble the hover record for point `idx`.
 *
 * Azimuth, elevation and slant range come from the point's return slot, never
 * from inverting x/y/z. Those coordinates come out of the 4/3 effective-earth
 * model, so x/y is a *ground* range and z a height above the antenna; undoing
 * them with flat trigonometry would report an elevation several times too large
 * at long range.
 */
function computeHover(buf, idx) {
  const slot = buf.slots[idx];
  const posOffset = 3 * idx;
  return {
    index: idx,
    value: dequantize(buf.codes[idx], buf.vmin, buf.vmax),
    x_m: Number(buf.positions[posOffset]),
    y_m: Number(buf.positions[posOffset + 1]),
    z_m: Number(buf.positions[posOffset + 2]),
    azimuth_deg: Number(buf.azimuth[slot]),
    elevation_deg: Number(buf.elevation[slot]),
    range_m: Number(buf.rangeM[idx]),
    gate_index: Number(buf.gateIndex[idx]),
    return_index: Number(buf.returnIndex[slot]),
  };
}

function formatTooltip(hover) {
  return (
    `value=${hover.value.toFixed(2)} xyz=(${hover.x_m.toFixed(0)},${hover.y_m.toFixed(0)},${hover.z_m.toFixed(0)})m ` +
    `az=${hover.azimuth_deg.toFixed(2)}deg el=${hover.elevation_deg.toFixed(2)}deg ` +
    `r=${(hover.range_m / 1000.0).toFixed(2)}km ret=${hover.return_index} gate=${hover.gate_index}`
  );
}

function fitZoomForExtent(maxAbsMeters, width, height) {
  if (!Number.isFinite(maxAbsMeters) || maxAbsMeters <= 0) {
    return -7.0;
  }
  const pixelRadius = Math.max(1.0, 0.45 * Math.min(width, height));
  const pixelsPerMeter = pixelRadius / maxAbsMeters;
  return Math.log2(pixelsPerMeter);
}

function pointSizeForCount(n) {
  if (n > 240000) return 0.9;
  if (n > 160000) return 1.1;
  if (n > 100000) return 1.3;
  return 1.6;
}

function lineWidthForCount(n) {
  if (n > 40000) return 0.8;
  if (n > 20000) return 1.0;
  return 1.2;
}

export default {
  render({ model, el }) {
    const root = document.createElement("div");
    root.className = "radrs-viz-root";
    const deckHost = document.createElement("div");
    deckHost.className = "radrs-viz-deck-host";
    const overlay = document.createElement("div");
    overlay.className = "radrs-viz-overlay";
    // The main canvas is WebGL, so the colorbar rides on its own 2D overlay.
    const colorbar = document.createElement("canvas");
    colorbar.className = "radrs-viz-colorbar";
    const tooltip = document.createElement("div");
    tooltip.className = "radrs-viz-tooltip";
    tooltip.style.display = "none";

    root.appendChild(deckHost);
    root.appendChild(overlay);
    root.appendChild(colorbar);
    root.appendChild(tooltip);
    el.appendChild(root);

    /** @type {Deck | null} */
    let deck = null;
    let codes = new Uint16Array(0);
    let slots = new Uint32Array(0);
    let gateIndex = new Uint16Array(0);
    let returnIndex = new Uint32Array(0);
    let azimuth = new Float32Array(0);
    let elevation = new Float32Array(0);
    let baseRange = new Float32Array(0);
    let rangeStep = new Float32Array(0);
    let rangeM = new Float32Array(0);
    let positions = new Float32Array(0);
    let colors = new Uint8Array(0);
    let vmin = 0;
    let vmax = 1;
    let n = 0;
    let lastHover = -1;
    let fitZoom = -7.0;
    let viewState = null;

    function readWidth() {
      return Number(model.get("width")) || 760;
    }

    function readHeight() {
      return Number(model.get("height")) || 760;
    }

    function readMeta() {
      const meta = model.get("meta");
      return meta && typeof meta === "object" ? meta : {};
    }

    function readAngle(name, fallback) {
      const value = Number(model.get(name));
      return Number.isFinite(value) ? value : fallback;
    }

    function hideTooltip() {
      tooltip.style.display = "none";
      if (lastHover !== -1) {
        lastHover = -1;
        model.set("hover", {});
        model.save_changes();
      }
    }

    function loadAndColorize() {
      codes = decodeArray(model.get("value_bytes"), Uint16Array);
      slots = decodeArray(model.get("return_slot_bytes"), Uint32Array);
      gateIndex = decodeArray(model.get("gate_index_bytes"), Uint16Array);
      returnIndex = decodeArray(model.get("return_index_bytes"), Uint32Array);
      azimuth = decodeArray(model.get("azimuth_bytes"), Float32Array);
      elevation = decodeArray(model.get("elevation_bytes"), Float32Array);
      baseRange = decodeArray(model.get("base_range_bytes"), Float32Array);
      rangeStep = decodeArray(model.get("range_step_bytes"), Float32Array);

      const nSlots = Math.min(
        returnIndex.length,
        azimuth.length,
        elevation.length,
        baseRange.length,
        rangeStep.length,
      );
      n = nSlots > 0 ? Math.min(codes.length, slots.length, gateIndex.length) : 0;

      positions = new Float32Array(n * 3);
      rangeM = new Float32Array(n);
      colors = new Uint8Array(n * 3);
      const meta = readMeta();
      const lut = buildLut(String(meta.colormap || "viridis"));
      vmin = Number(meta.vmin);
      vmax = Number(meta.vmax);
      // deck.gl wants an explicit positions buffer, so build one here from the
      // per-return geometry: same 4/3 effective-earth mapping the Python side
      // uses, run once per point instead of shipped as three float32 each.
      for (let i = 0; i < n; i += 1) {
        const slot = slots[i] < nSlots ? slots[i] : 0;
        const el = elevation[slot];
        const r = baseRange[slot] + gateIndex[i] * rangeStep[slot];
        rangeM[i] = r;
        const ground = beamGroundRangeM(r, el);
        const azRad = (azimuth[slot] * Math.PI) / 180.0;
        const posOffset = 3 * i;
        positions[posOffset] = ground * Math.sin(azRad);
        positions[posOffset + 1] = ground * Math.cos(azRad);
        positions[posOffset + 2] = beamHeightM(r, el);
        const code = codes[i];
        if (code === QUANT_NAN) {
          // A ray with no finite gate in this moment. Left black, as before.
          continue;
        }
        const [r8, g8, b8] = lutColor(lut, quantFraction(code));
        colors[posOffset] = r8;
        colors[posOffset + 1] = g8;
        colors[posOffset + 2] = b8;
      }

      const width = readWidth();
      const height = readHeight();
      fitZoom = fitZoomForExtent(Number(meta.max_abs_m), width, height);
      if (viewState === null) {
        viewState = {
          target: [0, 0, 0],
          rotationOrbit: readAngle("yaw_deg", 35),
          rotationX: readAngle("pitch_deg", 30),
          zoom: fitZoom,
          minZoom: fitZoom - 5.0,
          maxZoom: fitZoom + 8.0,
          minRotationX: -89,
          maxRotationX: 89,
        };
      } else {
        const zoomOffset = viewState.zoom - fitZoom;
        viewState = {
          ...viewState,
          target: [0, 0, 0],
          zoom: clamp(fitZoom + zoomOffset, fitZoom - 5.0, fitZoom + 8.0),
          minZoom: fitZoom - 5.0,
          maxZoom: fitZoom + 8.0,
          minRotationX: -89,
          maxRotationX: 89,
        };
      }
    }

    function buildLayer() {
      const meta = readMeta();
      const renderMode = String(meta.render_mode || "points");
      if (renderMode === "rays") {
        return new LineLayer({
          id: "radrs-volume-rays",
          data: {
            length: n,
            attributes: {
              getTargetPosition: { value: positions, size: 3 },
              getColor: { value: colors, size: 3 },
            },
          },
          getSourcePosition: [0, 0, 0],
          coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
          pickable: true,
          widthUnits: "pixels",
          getWidth: lineWidthForCount(n),
          opacity: 0.95,
        });
      }
      return new PointCloudLayer({
        id: "radrs-volume-points",
        data: {
          length: n,
          attributes: {
            getPosition: { value: positions, size: 3 },
            getColor: { value: colors, size: 3 },
          },
        },
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        pickable: true,
        pointSize: pointSizeForCount(n),
        opacity: 0.95,
      });
    }

    function updateColorbar() {
      const meta = readMeta();
      const cbW = 66;
      const cbH = Math.max(120, Math.min(260, readHeight() - 120));
      colorbar.width = cbW;
      colorbar.height = cbH;
      const ctx = colorbar.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, cbW, cbH);
      drawColorbar(ctx, buildLut(String(meta.colormap || "viridis")), meta, 0, 0, cbW, cbH);
    }

    function updateOverlay() {
      const meta = readMeta();
      if (n === 0) {
        overlay.textContent = "Volume 3D (deck.gl) | no finite points";
        return;
      }
      const renderMode = String(meta.render_mode || "points");
      const itemLabel = renderMode === "rays" ? "rays" : "gates";
      overlay.textContent = (
        `Volume 3D (deck.gl) | ${itemLabel}=${n.toLocaleString()} ` +
        `moment=${String(meta.moment || "")} ` +
        `range=${(Number(meta.max_abs_m) / 1000.0).toFixed(2)}km`
      );
    }

    function ensureDeck() {
      if (deck !== null) {
        return;
      }
      deck = new Deck({
        parent: deckHost,
        width: readWidth(),
        height: readHeight(),
        views: [new OrbitView({ id: "orbit" })],
        controller: true,
        viewState,
        layers: [buildLayer()],
        parameters: {
          clearColor: [246, 248, 250, 255],
        },
        getCursor: ({ isDragging }) => (isDragging ? "grabbing" : "grab"),
        onViewStateChange: ({ viewState: nextViewState }) => {
          if (!nextViewState) return;
          viewState = {
            ...nextViewState,
            minRotationX: -89,
            maxRotationX: 89,
          };
          deck?.setProps({ viewState });
        },
        onHover: (info) => {
          if (!info || typeof info.index !== "number" || info.index < 0 || info.index >= n) {
            hideTooltip();
            return;
          }
          const idx = info.index;
          const hover = computeHover(
            {
              codes, slots, gateIndex, returnIndex,
              azimuth, elevation, positions, rangeM, vmin, vmax,
            },
            idx,
          );
          const px = Number.isFinite(info.x) ? info.x : 8;
          const py = Number.isFinite(info.y) ? info.y : 8;
          tooltip.style.left = `${Math.max(8, Math.round(px + 10))}px`;
          tooltip.style.top = `${Math.max(8, Math.round(py + 10))}px`;
          tooltip.style.display = "block";
          tooltip.textContent = formatTooltip(hover);
          if (idx !== lastHover) {
            lastHover = idx;
            model.set("hover", hover);
            model.save_changes();
          }
        },
      });
    }

    function renderDeck() {
      const width = readWidth();
      const height = readHeight();
      root.style.width = `${width}px`;
      root.style.height = `${height}px`;
      ensureDeck();
      deck?.setProps({
        width,
        height,
        viewState,
        layers: [buildLayer()],
      });
      updateOverlay();
      updateColorbar();
    }

    function refreshFromModel() {
      loadAndColorize();
      renderDeck();
      hideTooltip();
    }

    function onSizeChange() {
      const width = readWidth();
      const height = readHeight();
      const nextFit = fitZoomForExtent(Number(readMeta().max_abs_m), width, height);
      if (viewState !== null) {
        const zoomOffset = viewState.zoom - fitZoom;
        fitZoom = nextFit;
        viewState = {
          ...viewState,
          zoom: clamp(nextFit + zoomOffset, nextFit - 5.0, nextFit + 8.0),
          minZoom: nextFit - 5.0,
          maxZoom: nextFit + 8.0,
        };
      }
      renderDeck();
    }

    function onAngleChange() {
      if (viewState === null) {
        return;
      }
      viewState = {
        ...viewState,
        rotationOrbit: readAngle("yaw_deg", viewState.rotationOrbit),
        rotationX: readAngle("pitch_deg", viewState.rotationX),
      };
      renderDeck();
    }

    const redrawWatched = ["meta"];
    const dataWatched = [
      "value_bytes",
      "return_slot_bytes",
      "gate_index_bytes",
      "return_index_bytes",
      "azimuth_bytes",
      "elevation_bytes",
      "base_range_bytes",
      "range_step_bytes",
    ];
    for (const key of redrawWatched) {
      model.on(`change:${key}`, refreshFromModel);
    }
    for (const key of dataWatched) {
      model.on(`change:${key}`, refreshFromModel);
    }
    model.on("change:width", onSizeChange);
    model.on("change:height", onSizeChange);
    model.on("change:yaw_deg", onAngleChange);
    model.on("change:pitch_deg", onAngleChange);

    refreshFromModel();
    // deck.gl may skip the first paint if the container isn't visible yet
    // (e.g. marimo defers widget display).  Re-render once it enters the
    // viewport so the WebGL canvas gets the correct dimensions.
    let visObs = new IntersectionObserver((entries) => {
      if (entries[0]?.isIntersecting && deck) {
        visObs.disconnect();
        visObs = null;
        deck.setProps({ width: readWidth(), height: readHeight(), layers: [buildLayer()] });
      }
    }, { threshold: 0.01 });
    visObs.observe(root);

    return () => {
      if (visObs) { visObs.disconnect(); visObs = null; }
      for (const key of redrawWatched) {
        model.off(`change:${key}`, refreshFromModel);
      }
      for (const key of dataWatched) {
        model.off(`change:${key}`, refreshFromModel);
      }
      model.off("change:width", onSizeChange);
      model.off("change:height", onSizeChange);
      model.off("change:yaw_deg", onAngleChange);
      model.off("change:pitch_deg", onAngleChange);
      if (deck !== null) {
        deck.finalize();
        deck = null;
      }
    };
  },
};
