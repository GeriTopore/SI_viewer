import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import type { Frame } from "../api/arrays";
import { useDisplay } from "../state/display";
import { ImageRenderer, type DisplayParams, type ViewTransform } from "../viewer/renderer";
import { formatValue, niceScaleBar } from "../format";

export interface AxisCal {
  scale: number;
  offset: number;
  units: string;
}

interface Props {
  /** Key under which pan/zoom is remembered. */
  viewKey: string;
  frame: Frame | null;
  /** Full-resolution size in data pixels (frames may be binned). */
  fullWidth: number;
  fullHeight: number;
  display: DisplayParams | undefined;
  calX: AxisCal;
  calY: AxisCal;
  /** Message shown over the image while data loads. */
  loading?: string | null;
  error?: string | null;
  /** Extra overlay (e.g. a frame label), drawn top-left. */
  children?: ReactNode;
}

const MAX_ZOOM = 64;

interface Hover {
  x: number;
  y: number;
  value: number | null;
}

export function ImageView({
  viewKey, frame, fullWidth, fullHeight, display, calX, calY, loading, error, children,
}: Props) {
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

  // Switching to a different view key: restore its saved view, or fit.
  useEffect(() => {
    const saved = useDisplay.getState().views[viewKey];
    fitted.current = !saved;
    setViewState(saved ?? null);
  }, [viewKey]);

  useEffect(() => {
    if (size.w && fullWidth && (fitted.current || !view)) fit();
  }, [size.w, size.h, fullWidth, fullHeight]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const r = rendererRef.current;
    if (!r) return;
    if (frame) r.setData(frame.data, frame.width, frame.height, frame.bin);
    else r.clearData();
  }, [frame]);

  // Draw on any change, batched into one animation frame.
  useEffect(() => {
    const r = rendererRef.current;
    if (!r || !view || !display) return;
    const id = requestAnimationFrame(() => r.draw(view, display, window.devicePixelRatio || 1));
    return () => cancelAnimationFrame(id);
  }, [view, display, frame, size]);

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

  const onPointerDown = (e: React.PointerEvent) => {
    if (!view || (e.button !== 0 && e.button !== 1)) return;
    e.currentTarget.setPointerCapture(e.pointerId);
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
    const ix = Math.floor(p.x);
    const iy = Math.floor(p.y);
    if (ix < 0 || iy < 0 || ix >= fullWidth || iy >= fullHeight) {
      setHover(null);
      return;
    }
    let value: number | null = null;
    if (frame) {
      const fx = Math.floor(ix / frame.bin);
      const fy = Math.floor(iy / frame.bin);
      if (fx < frame.width && fy < frame.height) value = frame.data[fy * frame.width + fx];
    }
    setHover({ x: ix, y: iy, value });
  };

  const onPointerUp = () => {
    drag.current = null;
  };

  const bar = view && niceScaleBar(calX.scale / view.zoom, calX.units);

  return (
    <div className="image-view">
      <div
        ref={wrapRef}
        className="image-canvas-wrap"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => setHover(null)}
        onDoubleClick={fit}
      >
        <canvas ref={canvasRef} />
        {children && <div className="image-overlay-tl">{children}</div>}
        {bar && frame && (
          <div className="scale-bar">
            <div className="scale-bar-line" style={{ width: bar.pixels }} />
            <span>{bar.label}</span>
          </div>
        )}
        {(loading || error || glError) && (
          <div className={`image-message${error || glError ? " error" : ""}`}>
            {loading && !error && <span className="spinner" aria-hidden />}
            {glError ?? error ?? loading}
          </div>
        )}
      </div>
      <div className="readout">
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
              {hover.value !== null && <> · {formatValue(hover.value)}</>}
            </>
          ) : (
            <span className="muted">
              {fullWidth} × {fullHeight} px
              {frame && frame.bin > 1 && ` · preview binned ×${frame.bin}`}
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
        </span>
      </div>
    </div>
  );
}
