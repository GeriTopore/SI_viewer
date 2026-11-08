import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import type { Frame } from "../api/arrays";
import { useDisplay } from "../state/display";
import { useViewerImage } from "../state/viewerImage";
import { ImageRenderer, type DisplayParams, type ViewTransform } from "../viewer/renderer";
import { formatValue, niceScaleBar } from "../format";
import { isTyping, useShortcuts } from "../keyboard";
import { downloadBlob, safeFileName } from "../download";
import { compositePng } from "../viewer/snapshot";
import { useLayers } from "../state/layers";

export interface AxisCal {
  scale: number;
  offset: number;
  units: string;
}

/** A point in data pixel coordinates (pixel i spans [i, i+1)). */
export interface Pt {
  x: number;
  y: number;
}

/** Maps data coordinates to CSS pixels within the image area. */
export interface Transform {
  zoom: number;
  toScreen: (x: number, y: number) => [number, number];
}

/** Pointer/keyboard tool on top of pan/zoom (e.g. a cursor or region tool).
 * Middle-drag or Space+drag always pans. */
export interface Interaction {
  /** CSS cursor over the image. */
  cursor?: string;
  /** Return true to take the drag (otherwise the image pans). */
  onDown?: (p: Pt, e: React.PointerEvent, t: Transform) => boolean;
  onMove?: (p: Pt, e: React.PointerEvent, t: Transform) => void;
  onUp?: (p: Pt, e: React.PointerEvent, t: Transform) => void;
  /** Pointer moved without a drag (null when it leaves). */
  onHover?: (p: Pt | null) => void;
  /** Return true to stop double-click from fitting the image. */
  onDoubleClick?: (p: Pt) => boolean;
  /** Keys while the image has focus; return true if handled. */
  onKeyDown?: (e: React.KeyboardEvent) => boolean;
}

/** One image layer. The first is the base; later ones are added on top. */
export interface ViewLayer {
  key: string;
  label: string;
  frame: Frame | null;
  display: DisplayParams | undefined;
  opacity: number;
  visible: boolean;
}

interface Props {
  /** Key under which pan/zoom is remembered. */
  viewKey: string;
  layers: ViewLayer[];
  /** Full-resolution size in data pixels (frames may be binned). */
  fullWidth: number;
  fullHeight: number;
  calX: AxisCal;
  calY: AxisCal;
  /** Message shown over the image while data loads. */
  loading?: string | null;
  error?: string | null;
  /** Extra overlay (e.g. a frame label), drawn top-left. */
  children?: ReactNode;
  /** Interactive panel drawn top-right (e.g. the layer list). */
  panel?: ReactNode;
  /** Hint shown in the readout bar when the pointer is outside the image. */
  hint?: string;
  /** Message shown when there are no layers to draw. */
  empty?: string | null;
  interaction?: Interaction;
  /** SVG content drawn over the image in screen coordinates. */
  renderOverlay?: (t: Transform) => ReactNode;
  /** Layer whose visible part is published for analysis (the FFT panel);
   * defaults to the first visible layer. */
  analysisKey?: string;
  /** Show the scale bar (default true). */
  scaleBar?: boolean;
  /** Show the readout bar under the image (size, hover values, zoom; default true). */
  readout?: boolean;
  /** Publish the analysed layer for the FFT panel (default true; off for
   * secondary views such as diagnostic panels). */
  publish?: boolean;
}

const MAX_ZOOM = 64;

interface Hover {
  x: number;
  y: number;
  values: { label: string; value: number }[];
}

const valueAt = (frame: Frame, ix: number, iy: number) => {
  const fx = Math.floor(ix / frame.bin);
  const fy = Math.floor(iy / frame.bin);
  return fx < frame.width && fy < frame.height ? frame.data[fy * frame.width + fx] : null;
};

// The image view under the mouse (its owner id), which takes F and 1.
let hoveredView: string | null = null;

export function ImageView({
  viewKey, layers, fullWidth, fullHeight, calX, calY, loading, error, children, panel, hint, empty,
  interaction, renderOverlay, analysisKey, publish: publishes = true, scaleBar = true,
  readout = true,
}: Props) {
  const anyFrame = layers.some((l) => l.frame);
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<ImageRenderer | null>(null);
  const [glError, setGlError] = useState<string | null>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [view, setViewState] = useState<ViewTransform | null>(
    () => useDisplay.getState().views[viewKey] ?? null,
  );
  // While "fitted", the image re-fits when the panel resizes.
  const fitted = useRef(view === null);
  const [hover, setHover] = useState<Hover | null>(null);
  const drag = useRef<{ x: number; y: number; view: ViewTransform } | null>(null);

  const setView = useCallback(
    (v: ViewTransform, isFit = false) => {
      fitted.current = isFit;
      setViewState(v);
      useDisplay.getState().setView(viewKey, v);
    },
    [viewKey],
  );

  const fitZoom = size.w && fullWidth ? Math.min(size.w / fullWidth, size.h / fullHeight) * 0.97 : 1;
  const fit = useCallback(
    () => setView({ zoom: fitZoom, cx: fullWidth / 2, cy: fullHeight / 2 }, true),
    [fitZoom, fullWidth, fullHeight, setView],
  );

  // Renderer lifetime
  useEffect(() => {
    try {
      rendererRef.current = new ImageRenderer(canvasRef.current!);
    } catch (e) {
      setGlError((e as Error).message);
    }
    return () => {
      rendererRef.current?.dispose();
      rendererRef.current = null;
    };
  }, []);

  // Track the panel size; the canvas backing store follows devicePixelRatio.
  useLayoutEffect(() => {
    const el = wrapRef.current!;
    const ro = new ResizeObserver(() => {
      const dpr = window.devicePixelRatio || 1;
      const canvas = canvasRef.current!;
      canvas.width = Math.round(el.clientWidth * dpr);
      canvas.height = Math.round(el.clientHeight * dpr);
      setSize({ w: el.clientWidth, h: el.clientHeight });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // A new image size (e.g. a stack shown aligned and cropped) always re-fits:
  // the old view was of another image.
  const imageSize = useRef({ w: fullWidth, h: fullHeight });

  // Switching to a different view key: restore its saved view, or fit.
  useEffect(() => {
    const saved = useDisplay.getState().views[viewKey];
    fitted.current = !saved;
    imageSize.current = { w: fullWidth, h: fullHeight };
    setViewState(saved ?? null);
  }, [viewKey]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const resized = imageSize.current.w !== fullWidth || imageSize.current.h !== fullHeight;
    if (size.w && fullWidth && (fitted.current || !view || resized)) {
      imageSize.current = { w: fullWidth, h: fullHeight };
      fit();
    }
  }, [size.w, size.h, fullWidth, fullHeight]); // eslint-disable-line react-hooks/exhaustive-deps

  // Upload data and draw on any change, batched into one animation frame.
  // Hidden layers keep their textures so toggling visibility is instant.
  useEffect(() => {
    const r = rendererRef.current;
    if (!r || !view) return;
    const id = requestAnimationFrame(() => {
      r.setLayers(layers.filter((l) => l.frame).map((l) => ({ key: l.key, data: l.frame! })));
      const draws = layers
        .filter((l) => l.visible && l.frame && l.display)
        .map((l) => ({ key: l.key, display: l.display!, opacity: l.opacity }));
      r.draw(view, draws, window.devicePixelRatio || 1);
    });
    return () => cancelAnimationFrame(id);
  }, [view, layers, size]);

  // Publish the analysed layer and the part of it in view (e.g. for the FFT).
  const owner = useId();
  const analysed =
    layers.find((l) => l.key === analysisKey && l.frame) ?? layers.find((l) => l.visible && l.frame);
  useEffect(() => {
    const { publish } = useViewerImage.getState();
    if (!publishes) return;
    if (!analysed?.frame || !view || !size.w) {
      publish(owner, null);
      return;
    }
    const hw = size.w / 2 / view.zoom;
    const hh = size.h / 2 / view.zoom;
    publish(owner, {
      label: analysed.label,
      frame: analysed.frame,
      calX,
      calY,
      x0: Math.max(0, Math.floor(view.cx - hw)),
      y0: Math.max(0, Math.floor(view.cy - hh)),
      x1: Math.min(fullWidth, Math.ceil(view.cx + hw)),
      y1: Math.min(fullHeight, Math.ceil(view.cy + hh)),
    });
  }, [owner, publishes, analysed?.frame, analysed?.label, view, size.w, size.h, fullWidth, fullHeight, calX.scale, calX.units, calY.scale]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => useViewerImage.getState().clear(owner), [owner]);

  const toData = (clientX: number, clientY: number, v: ViewTransform) => {
    const rect = wrapRef.current!.getBoundingClientRect();
    return {
      x: (clientX - rect.left - size.w / 2) / v.zoom + v.cx,
      y: (clientY - rect.top - size.h / 2) / v.zoom + v.cy,
    };
  };

  // Wheel zoom around the cursor (non-passive listener so the page doesn't scroll).
  useEffect(() => {
    const el = wrapRef.current!;
    const onWheel = (e: WheelEvent) => {
      if (!view) return;
      e.preventDefault();
      const factor = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015));
      const zoom = Math.min(MAX_ZOOM, Math.max(fitZoom / 4, view.zoom * factor));
      const p = toData(e.clientX, e.clientY, view);
      const rect = el.getBoundingClientRect();
      const sx = e.clientX - rect.left - size.w / 2;
      const sy = e.clientY - rect.top - size.h / 2;
      setView({ zoom, cx: p.x - sx / zoom, cy: p.y - sy / zoom });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  });

  const transform: Transform | null = view && {
    zoom: view.zoom,
    toScreen: (x, y) => [(x - view.cx) * view.zoom + size.w / 2, (y - view.cy) * view.zoom + size.h / 2],
  };

  // Space held: drags pan even when a tool is active.
  const spaceDown = useRef(false);
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === " " && !isTyping(e)) spaceDown.current = true;
    };
    const up = (e: KeyboardEvent) => e.key === " " && (spaceDown.current = false);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);
  const tooling = useRef(false); // the current drag belongs to the interaction

  const onPointerDown = (e: React.PointerEvent) => {
    if (!view || (e.button !== 0 && e.button !== 1)) return;
    wrapRef.current!.focus({ preventScroll: true });
    e.currentTarget.setPointerCapture(e.pointerId);
    const p = toData(e.clientX, e.clientY, view);
    if (e.button === 0 && !spaceDown.current && interaction?.onDown?.(p, e, transform!)) {
      tooling.current = true;
      return;
    }
    drag.current = { x: e.clientX, y: e.clientY, view };
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!view) return;
    const d = drag.current;
    if (d) {
      setView({
        ...d.view,
        cx: d.view.cx - (e.clientX - d.x) / d.view.zoom,
        cy: d.view.cy - (e.clientY - d.y) / d.view.zoom,
      });
      return;
    }
    const p = toData(e.clientX, e.clientY, view);
    if (tooling.current) interaction?.onMove?.(p, e, transform!);
    else interaction?.onHover?.(p);
    const ix = Math.floor(p.x);
    const iy = Math.floor(p.y);
    if (ix < 0 || iy < 0 || ix >= fullWidth || iy >= fullHeight) {
      setHover(null);
      return;
    }
    const values = layers.flatMap((l) => {
      const v = l.visible && l.frame ? valueAt(l.frame, ix, iy) : null;
      return v === null ? [] : [{ label: l.label, value: v }];
    });
    setHover({ x: ix, y: iy, values });
  };

  const onPointerUp = (e: React.PointerEvent) => {
    drag.current = null;
    if (tooling.current && view) interaction?.onUp?.(toData(e.clientX, e.clientY, view), e, transform!);
    tooling.current = false;
  };

  const bar = view && niceScaleBar(calX.scale / view.zoom, calX.units);

  // F / 1 act on the view under the mouse, else on the main (published) one.
  const ownsKeys = () => (hoveredView ? hoveredView === owner : publishes);
  useEffect(() => () => {
    if (hoveredView === owner) hoveredView = null;
  }, [owner]);
  useShortcuts({
    f: () => (ownsKeys() ? fit() : false),
    "1": () => (ownsKeys() && view ? setView({ ...view, zoom: 1 }) : false),
  });
  // ⌘/Ctrl+S saves the image (instead of the browser saving the page).
  const save = useRef<() => void>(() => {});
  useEffect(() => {
    if (!publishes) return;
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && !e.shiftKey && e.key.toLowerCase() === "s") {
        e.preventDefault();
        save.current();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [publishes]);

  const saveImage = () => {
    const shown = layers.filter((l) => l.visible && l.frame && l.display);
    if (!shown.length) return;
    const name = `${safeFileName(shown.map((l) => l.label).join("+"))}.png`;
    compositePng(
      shown.map((l) => ({ key: l.key, frame: l.frame!, display: l.display!, opacity: l.opacity })),
      fullWidth, fullHeight, calX,
    )
      .then((blob) => downloadBlob(blob, name))
      .catch((e) => useLayers.getState().notify(`Could not save the image: ${(e as Error).message}`));
  };
  save.current = saveImage;

  return (
    <div className="image-view">
      <div
        ref={wrapRef}
        className="image-canvas-wrap"
        tabIndex={0}
        style={interaction?.cursor ? { cursor: interaction.cursor } : undefined}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerEnter={() => {
          hoveredView = owner;
        }}
        onPointerLeave={() => {
          if (hoveredView === owner) hoveredView = null;
          setHover(null);
          interaction?.onHover?.(null);
        }}
        onDoubleClick={(e) => {
          if (view && interaction?.onDoubleClick?.(toData(e.clientX, e.clientY, view))) return;
          fit();
        }}
        onKeyDown={(e) => {
          if (interaction?.onKeyDown?.(e)) e.preventDefault();
        }}
      >
        <canvas ref={canvasRef} />
        {renderOverlay && transform && (
          <svg className="image-overlay-svg" width={size.w} height={size.h}>
            {renderOverlay(transform)}
          </svg>
        )}
        {children && <div className="image-overlay-tl">{children}</div>}
        {panel && (
          <div
            className="image-overlay-tr"
            onPointerDown={(e) => e.stopPropagation()}
            onPointerMove={(e) => e.stopPropagation()}
            onDoubleClick={(e) => e.stopPropagation()}
          >
            {panel}
          </div>
        )}
        {scaleBar && bar && anyFrame && (
          <div className="scale-bar">
            <div className="scale-bar-line" style={{ width: bar.pixels }} />
            <span>{bar.label}</span>
          </div>
        )}
        {empty && !loading && !error && !glError && <div className="image-message">{empty}</div>}
        {(loading || error || glError) && (
          <div className={`image-message${error || glError ? " error" : ""}`}>
            {loading && !error && <span className="spinner" aria-hidden />}
            {glError ?? error ?? loading}
          </div>
        )}
      </div>
      {readout && <div className="readout">
        <span className="readout-pos">
          {hover ? (
            <>
              x {hover.x}, y {hover.y}
              {calX.units && (
                <span className="muted">
                  {" "}
                  ({formatValue(calX.offset + (hover.x + 0.5) * calX.scale)},{" "}
                  {formatValue(calY.offset + (hover.y + 0.5) * calY.scale)} {calX.units})
                </span>
              )}
              {hover.values.length === 1 && <> · {formatValue(hover.values[0].value)}</>}
              {hover.values.length > 1 &&
                hover.values.map((v, i) => (
                  <span key={i}>
                    {" · "}
                    <span className="muted">{v.label}</span> {formatValue(v.value)}
                  </span>
                ))}
            </>
          ) : (
            <span className="muted">
              {fullWidth} × {fullHeight} px
              {(layers[0]?.frame?.bin ?? 1) > 1 && ` · preview binned ×${layers[0].frame!.bin}`}
              {hint && ` · ${hint}`}
            </span>
          )}
        </span>
        <span className="readout-zoom">
          {view && `${Math.round(view.zoom * 100)}%`}
          <button className="link-button" onClick={fit} title="Fit to window (double-click image)">
            Fit
          </button>
          <button
            className="link-button"
            onClick={() => view && setView({ ...view, zoom: 1 })}
            title="One screen pixel per data pixel"
          >
            1:1
          </button>
          <button
            className="link-button"
            onClick={saveImage}
            disabled={!anyFrame}
            title="Save the whole image as shown (layers, colours, contrast) at full resolution, with a scale bar, as PNG"
          >
            Save PNG
          </button>
        </span>
      </div>}
    </div>
  );
}
