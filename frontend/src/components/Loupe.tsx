import { useEffect, useRef, type ReactNode } from "react";
import { create } from "zustand";
import { useUi } from "../state/ui";
import { ImageRenderer } from "../viewer/renderer";
import type { ViewLayer } from "./ImageView";

/** Gap between the pointer and the magnifier (px). */
const GAP = 18;
/** The image viewers' magnifier: its size (px) and how much it enlarges the view. */
export const VIEWER_LOUPE_PX = 140;
export const VIEWER_LOUPE_MAG = 4;

/** Where a ``size`` px magnifier goes for a pointer at (cx, cy) in the
 * window: to its right, or to its left near the window's right edge. */
export function placeLoupe(cx: number, cy: number, size: number) {
  const right = cx + GAP + size < window.innerWidth - 4;
  return {
    left: right ? cx + GAP : cx - GAP - size,
    top: Math.min(Math.max(4, cy - size / 2), window.innerHeight - size - 4),
  };
}

/** The magnifier's frame: fixed beside the pointer, a crosshair at its centre
 * (the pointer), ``children`` (a canvas, overlays) inside. */
export function LoupeFrame({ cx, cy, size, children }: { cx: number; cy: number; size: number; children: ReactNode }) {
  const { left, top } = placeLoupe(cx, cy, size);
  const c = size / 2;
  return (
    <div className="loupe" style={{ left, top, width: size, height: size }} aria-hidden>
      {children}
      <svg width={size} height={size}>
        <line className="loupe-cross" x1={c} y1={0} x2={c} y2={c - 6} />
        <line className="loupe-cross" x1={c} y1={c + 6} x2={c} y2={size} />
        <line className="loupe-cross" x1={0} y1={c} x2={c - 6} y2={c} />
        <line className="loupe-cross" x1={c + 6} y1={c} x2={size} y2={c} />
      </svg>
    </div>
  );
}

/** The magnifier of an image viewer: its visible layers drawn by a second
 * renderer around the data point (x, y), VIEWER_LOUPE_MAG × the view's zoom. */
export function ViewerLoupe({ layers, x, y, zoom, cx, cy }: {
  layers: ViewLayer[];
  /** Data point under the pointer (full-resolution px) and the view's zoom. */
  x: number;
  y: number;
  zoom: number;
  /** The pointer in the window. */
  cx: number;
  cy: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const renderer = useRef<ImageRenderer | null>(null);
  useEffect(() => {
    try {
      renderer.current = new ImageRenderer(canvasRef.current!);
    } catch {
      renderer.current = null;
    }
    return () => {
      renderer.current?.dispose();
      renderer.current = null;
    };
  }, []);
  useEffect(() => {
    const r = renderer.current;
    const canvas = canvasRef.current;
    if (!r || !canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const d = Math.round(VIEWER_LOUPE_PX * dpr);
    if (canvas.width !== d) {
      canvas.width = d;
      canvas.height = d;
    }
    r.setLayers(layers.filter((l) => l.frame).map((l) => ({ key: l.key, data: l.frame! })));
    r.draw(
      { zoom: zoom * VIEWER_LOUPE_MAG, cx: x, cy: y },
      layers.filter((l) => l.visible && l.frame && l.display).map((l) => ({ key: l.key, display: l.display!, opacity: l.opacity })),
      dpr,
    );
  }, [layers, x, y, zoom]);
  return (
    <LoupeFrame cx={cx} cy={cy} size={VIEWER_LOUPE_PX}>
      <canvas ref={canvasRef} style={{ width: VIEWER_LOUPE_PX, height: VIEWER_LOUPE_PX }} />
    </LoupeFrame>
  );
}

/** The magnifier on or off: for the FFT panel or for the image viewers (M
 * toggles the one under the mouse). */
export function LoupeToggle({ which, className = "" }: { which: "fft" | "viewer"; className?: string }) {
  const on = useUi((s) => (which === "fft" ? s.fftLoupe : s.viewerLoupe));
  const toggle = () => {
    const ui = useUi.getState();
    if (which === "fft") ui.setFftLoupe(!on);
    else ui.setViewerLoupe(!on);
  };
  return (
    <button
      className={`icon-toggle${on ? " active" : ""}${className ? ` ${className}` : ""}`}
      aria-pressed={on}
      onClick={toggle}
      title={`Magnifier ${on ? "on" : "off"} (M over the ${which === "fft" ? "FFT" : "image"})`}
    >
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden>
        <circle cx="6.5" cy="6.5" r="4.5" />
        <path d="M10 10L14.5 14.5" />
      </svg>
    </button>
  );
}

/** The data point magnified in a group of linked views (a grid's panels):
 * set by the panel under the pointer, followed by the others. */
export const useLoupeLink = create<{
  at: { group: string; owner: string; x: number; y: number } | null;
  set: (at: { group: string; owner: string; x: number; y: number }) => void;
  clear: (owner: string) => void;
}>((set, get) => ({
  at: null,
  set: (at) => set({ at }),
  clear: (owner) => get().at?.owner === owner && set({ at: null }),
}));

/** The pointer's last position in the window, so a magnifier switched on
 * with M appears at once, before the pointer moves. */
export const lastPointer = { x: -1, y: -1 };
window.addEventListener("pointermove", (e) => {
  lastPointer.x = e.clientX;
  lastPointer.y = e.clientY;
}, { capture: true, passive: true });

/** M toggles the magnifier of the FFT when the pointer is over it, else the
 * image viewers'. */
export const overFft = () => !!document.querySelector(".fft-canvas-wrap:hover");
