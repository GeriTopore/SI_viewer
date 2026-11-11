import type { Frame } from "../api/arrays";
import type { AxisCal } from "../components/ImageView";
import { niceScaleBar, scaleBarLabel } from "../format";
import { ImageRenderer, type DisplayParams } from "./renderer";

export interface SnapshotLayer {
  key: string;
  frame: Frame;
  display: DisplayParams;
  opacity: number;
}

/** One image of an export: its layers (composited), size and calibration. */
export interface SnapshotPanel {
  layers: SnapshotLayer[];
  /** Full-resolution size in data pixels. */
  width: number;
  height: number;
  calX: AxisCal;
  /** Grid panels: the label, with an optional colour dot (an element map's colour). */
  label?: string;
  colour?: string;
}

// Small images are enlarged (whole pixels) to at least about this size.
const MIN_SIZE = 1024;
/** Largest side of a rendered panel. */
export const MAX_SIDE = 8192;
/** Largest side and area of the exported figure (canvas limits). */
export const MAX_FIGURE_SIDE = 16384;
export const MAX_FIGURE_AREA = 268_435_456;

/** The whole factor that enlarges an image to about ``MIN_SIZE`` (1 for large ones). */
export const autoFactor = (width: number, height: number) =>
  Math.max(1, Math.min(16, Math.floor(MIN_SIZE / Math.max(width, height))));

/** Whole scale factors offered for these panels: the automatic one and those
 * keeping the longest side within ``MAX_SIDE``. */
export function factorChoices(panels: { width: number; height: number }[]): number[] {
  const long = Math.max(...panels.map((p) => Math.max(p.width, p.height)));
  const auto = Math.min(...panels.map((p) => autoFactor(p.width, p.height)));
  const set = new Set([1, 2, 3, 4, 6, 8].filter((f) => f * long <= MAX_SIDE));
  set.add(auto);
  return [...set].sort((a, b) => a - b);
}

/** The panel's layers as displayed (colours, contrast, opacity), enlarged by
 * the whole factor ``s``, as a 2D canvas. */
export function renderBase(panel: SnapshotPanel, s: number): HTMLCanvasElement {
  const { layers, width, height } = panel;
  const gl = document.createElement("canvas");
  gl.width = width * s;
  gl.height = height * s;
  const renderer = new ImageRenderer(gl);
  const out = document.createElement("canvas");
  out.width = gl.width;
  out.height = gl.height;
  const ctx = out.getContext("2d")!;
  try {
    renderer.setLayers(layers.map((l) => ({ key: l.key, data: l.frame })));
    renderer.draw(
      { zoom: s, cx: width / 2, cy: height / 2 },
      layers.map((l) => ({ key: l.key, display: l.display, opacity: l.opacity })),
      1,
    );
    ctx.drawImage(gl, 0, 0); // same task as the draw, so the buffer is still there
  } finally {
    renderer.dispose();
  }
  return out;
}

export type Corner = "lr" | "ll" | "ur" | "ul";
export type Ink = "white" | "black";
export type BarBackground = "none" | "black" | "white" | "translucent";

export interface BarStyle {
  /** In the image's calibrated units. */
  length: number;
  /** Bar thickness and label font size, in output pixels. */
  thickness: number;
  font: number;
  label: boolean;
  corner: Corner;
  colour: Ink;
  background: BarBackground;
}

export interface LabelStyle {
  font: number;
  colour: Ink;
}

export interface FigureStyle {
  /** Columns, or the number of panels in each row (in order). */
  cols: number | number[];
  /** Between panels and around the figure, in output pixels. */
  spacing: number;
  padding: number;
  /** CSS colour behind and between the panels. */
  background: string;
  labels: LabelStyle | null;
  bar: BarStyle | null;
  /** Panels that get a scale bar. */
  barOn: "first" | "all";
}

export interface FigurePanel {
  base: HTMLCanvasElement;
  /** Data width of the panel (``base.width`` is this times the scale factor). */
  dataWidth: number;
  calX: AxisCal;
  label?: string;
  colour?: string;
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface FigureLayout {
  width: number;
  height: number;
  cellW: number;
  cellH: number;
  /** Where each panel is drawn (fitted and centred in its cell). */
  rects: Rect[];
}

/** Size of the figure and the place of each panel: ``sizes`` (of the rendered
 * panels) fitted into equal cells, in ``cols`` columns or in rows of the
 * given lengths (a shorter row is centred). */
export function figureLayout(
  sizes: { w: number; h: number }[], cols: number | number[], spacing: number, padding: number,
): FigureLayout {
  const cellW = Math.max(...sizes.map((b) => b.w));
  const cellH = Math.max(...sizes.map((b) => b.h));
  const lengths = Array.isArray(cols)
    ? cols
    : Array.from({ length: Math.ceil(sizes.length / cols) }, (_, r) => Math.min(cols, sizes.length - r * cols));
  const width = Math.max(...lengths);
  // Row and column of each panel.
  const place = lengths.flatMap((n, r) => Array.from({ length: n }, (_, c) => ({ r, c: c + (width - n) / 2 })));
  const rects = sizes.map((b, i) => {
    const { r, c } = place[i] ?? { r: lengths.length, c: 0 };
    const x0 = padding + c * (cellW + spacing);
    const y0 = padding + r * (cellH + spacing);
    const k = Math.min(cellW / b.w, cellH / b.h);
    const w = b.w * k;
    const h = b.h * k;
    return { x: x0 + (cellW - w) / 2, y: y0 + (cellH - h) / 2, w, h };
  });
  return {
    width: 2 * padding + width * cellW + (width - 1) * spacing,
    height: 2 * padding + lengths.length * cellH + (lengths.length - 1) * spacing,
    cellW,
    cellH,
    rects,
  };
}

/** Automatic spacing between the panels of a grid. */
export const autoSpacing = (cellW: number, cellH: number) => Math.round(Math.max(cellW, cellH) / 100) + 2;

/** Automatic scale bar thickness and label font for a panel drawn ``w`` wide. */
export const autoBarSizes = (w: number) => ({
  thickness: Math.max(3, Math.round(w / 150)),
  font: Math.max(12, Math.round(w / 32)),
});

/** Automatic font of the panel labels of a grid. */
export const autoLabelFont = (cellW: number) => Math.max(12, Math.round(cellW / 32));

/** The automatic scale bar length (about a fifth of the width), in the
 * calibrated units; null when the calibration has no usable scale. */
export function autoBarLength(calX: AxisCal, dataWidth: number): number | null {
  const bar = niceScaleBar((calX.scale * dataWidth) / 500, calX.units);
  return bar ? Number(bar.length.toPrecision(6)) : null; // no floating-point residue
}

const FONT = `-apple-system, "Segoe UI", sans-serif`;
const INK = { white: "#fff", black: "#000" };
const OUTLINE = { white: "rgba(0, 0, 0, 0.7)", black: "rgba(255, 255, 255, 0.7)" };
const BOX = { black: "#000", white: "#fff", translucent: "rgba(0, 0, 0, 0.5)" };

/**
 * The figure: panels in a grid on a background, each with its label and, on
 * the chosen panels, a scale bar. Pure 2D drawing on top of the rendered bases.
 */
export function composeFigure(panels: FigurePanel[], style: FigureStyle): HTMLCanvasElement {
  const layout = figureLayout(
    panels.map((p) => ({ w: p.base.width, h: p.base.height })),
    style.cols, style.spacing, style.padding,
  );
  const out = document.createElement("canvas");
  out.width = layout.width;
  out.height = layout.height;
  const ctx = out.getContext("2d")!;
  ctx.fillStyle = style.background;
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.imageSmoothingEnabled = false;
  panels.forEach((p, i) => {
    const r = layout.rects[i];
    ctx.drawImage(p.base, r.x, r.y, r.w, r.h);
    if (style.bar && (style.barOn === "all" || i === 0)) drawScaleBar(ctx, r, p, style.bar);
    if (style.labels && p.label !== undefined) drawLabel(ctx, r, p, style.labels);
  });
  return out;
}

/** Length of the scale bar in output pixels on a panel drawn into ``r``. */
export const barPixels = (length: number, calX: AxisCal, dataWidth: number, r: Rect) =>
  (length * r.w) / (calX.scale * dataWidth);

function drawScaleBar(ctx: CanvasRenderingContext2D, r: Rect, p: FigurePanel, bar: BarStyle) {
  if (!(p.calX.scale > 0) || !(bar.length > 0)) return;
  const length = barPixels(bar.length, p.calX, p.dataWidth, r);
  if (length > r.w) return; // does not fit this panel
  const { thickness: thick, font } = bar;
  const margin = Math.round(r.w * 0.03);
  const text = bar.label ? scaleBarLabel(bar.length, p.calX.units) : null;
  ctx.save();
  ctx.font = `600 ${font}px ${FONT}`;
  const textW = text ? ctx.measureText(text).width : 0;
  const gap = font * 0.25;
  // The block (label over bar) is anchored to its corner; both are centred in it.
  const bw = Math.max(length, textW);
  const bh = thick + (text ? gap + font : 0);
  const left = bar.corner[1] === "r" ? r.x + r.w - margin - bw : r.x + margin;
  const top = bar.corner[0] === "l" ? r.y + r.h - margin - bh : r.y + margin;
  const cx = left + bw / 2;
  const barY = top + bh - thick;
  const x0 = cx - length / 2;
  if (bar.background !== "none") {
    const pad = Math.max(4, Math.round(font * 0.4));
    ctx.fillStyle = BOX[bar.background];
    ctx.fillRect(left - pad, top - pad, bw + 2 * pad, bh + 2 * pad);
  }
  const outline = bar.background === "none";
  if (text) {
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.lineJoin = "round";
    const ty = barY - gap;
    if (outline) {
      ctx.strokeStyle = OUTLINE[bar.colour];
      ctx.lineWidth = Math.max(2, font / 6);
      ctx.strokeText(text, cx, ty);
    }
    ctx.fillStyle = INK[bar.colour];
    ctx.fillText(text, cx, ty);
  }
  if (outline) {
    ctx.fillStyle = OUTLINE[bar.colour];
    ctx.fillRect(x0 - 1, barY - 1, length + 2, thick + 2);
  }
  ctx.fillStyle = INK[bar.colour];
  ctx.fillRect(x0, barY, length, thick);
  ctx.restore();
}

function drawLabel(ctx: CanvasRenderingContext2D, r: Rect, p: FigurePanel, labels: LabelStyle) {
  const { font } = labels;
  ctx.save();
  ctx.font = `600 ${font}px ${FONT}`;
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.lineJoin = "round";
  let tx = r.x + font * 0.6;
  const ty = r.y + font * 0.5;
  if (p.colour) {
    ctx.fillStyle = p.colour;
    ctx.beginPath();
    ctx.arc(tx + font * 0.35, ty + font * 0.55, font * 0.35, 0, 2 * Math.PI);
    ctx.fill();
    tx += font * 1.0;
  }
  ctx.strokeStyle = OUTLINE[labels.colour];
  ctx.lineWidth = Math.max(2, font / 6);
  ctx.strokeText(p.label!, tx, ty);
  ctx.fillStyle = INK[labels.colour];
  ctx.fillText(p.label!, tx, ty);
  ctx.restore();
}

export const canvasBlob = (c: HTMLCanvasElement) =>
  new Promise<Blob>((resolve, reject) =>
    c.toBlob((b) => (b ? resolve(b) : reject(new Error("Could not encode the PNG"))), "image/png"),
  );
