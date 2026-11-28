import type { Frame } from "../api/arrays";
import type { AxisCal } from "../components/ImageView";
import { formatValue, niceScaleBar, scaleBarLabel, type Titled } from "../format";
import { colormapLut } from "./colormaps";
import { ImageRenderer, type DisplayParams } from "./renderer";

export interface SnapshotLayer {
  key: string;
  frame: Frame;
  display: DisplayParams;
  opacity: number;
  /** Values are fractions (e.g. strain): its colour bar may show percentages. */
  fraction?: boolean;
}

/** One image of an export: its layers (composited), size and calibration. */
export interface SnapshotPanel {
  layers: SnapshotLayer[];
  /** Full-resolution size in data pixels. */
  width: number;
  height: number;
  calX: AxisCal;
  /** The y axis' calibration (axes of a single image); the x one if absent. */
  calY?: AxisCal;
  /** Automatic titles of a single image's axes and of its values (a colour bar). */
  titles?: { x: Titled; y: Titled; value: Titled };
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

// ---------- decorations of a single image: axes and a colour bar ----------

export type Edge = "right" | "left" | "top" | "bottom";

/** How a scale's ticks are labelled: how many (evenly from end to end) and
 * the decimals (null: automatic). */
export interface TickStyle {
  ticks: number;
  decimals: number | null;
  font: number;
}

export interface ColourBarStyle extends TickStyle {
  edge: Edge;
  /** Thickness of the bar and its distance from the image (or its axis), in output pixels. */
  width: number;
  distance: number;
  /** Values (fractions, e.g. strain) labelled as percentages (the units say %). */
  percent: boolean;
  /** Written along the bar, beyond its labels ("" for none). */
  title: string;
}

export interface AxesStyle extends TickStyle {
  /** Under the x axis and beside the y axis ("" for none). */
  x: string;
  y: string;
}


/** Behind the added margins ("none": transparent) and of their text. */
export interface MarginStyle {
  background: "white" | "black" | "none";
  colour: Ink;
}

/** The layer a colour bar describes: the first drawn with a colour map (not
 * one carrying its own colours, e.g. an IPF map); null if none. */
export function colourBarLayer(panel: SnapshotPanel): SnapshotLayer | null {
  return panel.layers.find((l) => l.opacity > 0 && !l.frame.rgb) ?? null;
}

/** Automatic colour bar width and distance for an image drawn ``w`` × ``h``. */
export const autoColourBar = (w: number, h: number) => {
  const width = Math.max(8, Math.round(Math.max(w, h) / 30));
  return { width, distance: Math.round(width / 2) };
};

/** ``n`` values evenly from ``a`` to ``b`` (ends included). */
const evenly = (a: number, b: number, n: number) =>
  Array.from({ length: Math.max(2, n) }, (_, i) => a + ((b - a) * i) / (Math.max(2, n) - 1));

const label = (v: number, decimals: number | null) =>
  decimals === null ? formatValue(v) : (Math.abs(v) < 0.5 * 10 ** -decimals ? 0 : v).toFixed(decimals);

/** A scale's ticks along a line of ``length`` output pixels: where (0 at the
 * start) and their text. */
function scaleTicks(a: number, b: number, length: number, t: TickStyle) {
  return evenly(a, b, t.ticks).map((v, i, all) => ({ at: (length * i) / (all.length - 1), text: label(v, t.decimals) }));
}

/**
 * ``figure`` (one image filling the canvas) with margins for its axes (the
 * calibrated x below, y on the left, ticks from end to end) and a colour bar
 * of ``layer`` (its colour map, contrast and gamma) beside ``cb.edge``, as
 * long as that edge, beyond the axis on that side.
 */
export function decorate(
  figure: HTMLCanvasElement, panel: SnapshotPanel, layer: SnapshotLayer | null,
  axes: AxesStyle | null, cb: ColourBarStyle | null, margin: MarginStyle,
): HTMLCanvasElement {
  const W = figure.width, H = figure.height;
  const measure = document.createElement("canvas").getContext("2d")!;
  const widest = (texts: string[], font: number) => {
    measure.font = `500 ${font}px ${FONT}`;
    return Math.max(0, ...texts.map((t) => measure.measureText(t).width));
  };
  const tickLen = (font: number) => Math.max(3, Math.round(font * 0.35));
  const gapOf = (font: number) => Math.round(font * 0.3);

  // Axes: x from the left edge, y from the top, in calibrated units.
  const calY = panel.calY ?? panel.calX;
  const xTicks = axes ? scaleTicks(panel.calX.offset, panel.calX.offset + panel.calX.scale * panel.width, W, axes) : [];
  const yTicks = axes ? scaleTicks(calY.offset, calY.offset + calY.scale * panel.height, H, axes) : [];
  const xTitle = axes?.x ?? "";
  const yTitle = axes?.y ?? "";
  const titleRoom = (t: string, font: number) => (t ? gapOf(font) + font * 1.2 : 0);
  const axisLeft = axes ? tickLen(axes.font) + gapOf(axes.font) + widest(yTicks.map((t) => t.text), axes.font) + titleRoom(yTitle, axes.font) : 0;
  const axisBottom = axes ? tickLen(axes.font) + gapOf(axes.font) + axes.font * 1.2 + titleRoom(xTitle, axes.font) : 0;

  // Colour bar: values lo → hi (as percentages if asked).
  const vertical = cb ? cb.edge === "left" || cb.edge === "right" : false;
  const k = cb?.percent ? 100 : 1;
  const lo = layer ? layer.display.lo : 0, hi = layer ? layer.display.hi : 1;
  const barTicks = cb && layer ? scaleTicks(lo * k, hi * k, vertical ? H : W, cb) : [];
  const barLabels = cb && layer ? (vertical ? widest(barTicks.map((t) => t.text), cb.font) : cb.font * 1.2) : 0;
  const barBlock = cb && layer
    ? Math.ceil(cb.distance + cb.width + tickLen(cb.font) + gapOf(cb.font) + barLabels
      + titleRoom(cb.title, cb.font) + cb.font * 0.3)
    : 0;
  const m = {
    left: axisLeft + (cb?.edge === "left" ? barBlock : 0),
    right: cb?.edge === "right" ? barBlock : 0,
    top: cb?.edge === "top" ? barBlock : 0,
    bottom: axisBottom + (cb?.edge === "bottom" ? barBlock : 0),
  };
  // A little room so end labels centred on the image's corners aren't cut.
  if (axes) m.right = Math.max(m.right, Math.ceil(axes.font * 0.3));
  const out = document.createElement("canvas");
  out.width = Math.ceil(W + m.left + m.right);
  out.height = Math.ceil(H + m.top + m.bottom);
  const ctx = out.getContext("2d")!;
  if (margin.background !== "none") {
    ctx.fillStyle = INK[margin.background];
    ctx.fillRect(0, 0, out.width, out.height);
  }
  const ix = Math.round(m.left), iy = Math.round(m.top);
  ctx.drawImage(figure, ix, iy);
  const ink = INK[margin.colour];
  ctx.strokeStyle = ink;
  ctx.fillStyle = ink;

  if (axes) {
    const f = axes.font, tl = tickLen(f), g = gapOf(f);
    ctx.lineWidth = Math.max(1, f / 14);
    ctx.font = `500 ${f}px ${FONT}`;
    // The frame's two axis lines.
    ctx.beginPath();
    ctx.moveTo(ix, iy);
    ctx.lineTo(ix, iy + H);
    ctx.lineTo(ix + W, iy + H);
    ctx.stroke();
    xTicks.forEach((t, i) => {
      const x = Math.min(Math.max(ix + t.at, ix + ctx.lineWidth / 2), ix + W - ctx.lineWidth / 2);
      ctx.beginPath();
      ctx.moveTo(x, iy + H);
      ctx.lineTo(x, iy + H + tl);
      ctx.stroke();
      ctx.textAlign = i === 0 ? "left" : i === xTicks.length - 1 ? "right" : "center";
      ctx.textBaseline = "top";
      ctx.fillText(t.text, ix + t.at, iy + H + tl + g);
    });
    yTicks.forEach((t, i) => {
      const y = Math.min(Math.max(iy + t.at, iy + ctx.lineWidth / 2), iy + H - ctx.lineWidth / 2);
      ctx.beginPath();
      ctx.moveTo(ix, y);
      ctx.lineTo(ix - tl, y);
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.textBaseline = i === 0 ? "top" : i === yTicks.length - 1 ? "bottom" : "middle";
      ctx.fillText(t.text, ix - tl - g, iy + t.at);
    });
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    if (xTitle) ctx.fillText(xTitle, ix + W / 2, iy + H + axisBottom);
    if (yTitle) {
      ctx.save();
      ctx.translate(ix - axisLeft, iy + H / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = "top";
      ctx.fillText(yTitle, 0, 0);
      ctx.restore();
    }
  }

  if (cb && layer) {
    const length = vertical ? H : W;
    const across = cb.edge === "right" ? ix + W + cb.distance
      : cb.edge === "left" ? ix - axisLeft - cb.distance - cb.width
      : cb.edge === "bottom" ? iy + H + axisBottom + cb.distance
      : iy - cb.distance - cb.width;
    // The scale, coloured as the renderer colours (gamma on the normalised value).
    const lut = colormapLut(layer.display.colormap);
    const n = Math.max(2, Math.round(length));
    const strip = new ImageData(vertical ? 1 : n, vertical ? n : 1);
    for (let i = 0; i < n; i++) {
      const c = Math.round(Math.pow(i / (n - 1), layer.display.gamma) * 255) * 4;
      strip.data.set([lut[c], lut[c + 1], lut[c + 2], 255], (vertical ? n - 1 - i : i) * 4); // vertical: hi at the top
    }
    const tmp = document.createElement("canvas");
    tmp.width = strip.width;
    tmp.height = strip.height;
    tmp.getContext("2d")!.putImageData(strip, 0, 0);
    ctx.imageSmoothingEnabled = true;
    if (vertical) ctx.drawImage(tmp, across, iy, cb.width, H);
    else ctx.drawImage(tmp, ix, across, W, cb.width);

    const f = cb.font, tl = tickLen(f), g = gapOf(f);
    ctx.lineWidth = Math.max(1, f / 14);
    ctx.font = `500 ${f}px ${FONT}`;
    const outward = cb.edge === "right" || cb.edge === "bottom" ? 1 : -1;
    const base = outward > 0 ? across + cb.width : across;
    barTicks.forEach((t, i) => {
      const last = i === barTicks.length - 1;
      // A vertical bar runs up: its first tick at the bottom.
      const at = vertical ? iy + H - t.at : ix + t.at;
      const lw = ctx.lineWidth / 2;
      const a = vertical ? Math.min(Math.max(at, iy + lw), iy + H - lw) : Math.min(Math.max(at, ix + lw), ix + W - lw);
      ctx.beginPath();
      if (vertical) { ctx.moveTo(base, a); ctx.lineTo(base + outward * tl, a); }
      else { ctx.moveTo(a, base); ctx.lineTo(a, base + outward * tl); }
      ctx.stroke();
      const off = base + outward * (tl + g);
      if (vertical) {
        ctx.textAlign = outward > 0 ? "left" : "right";
        ctx.textBaseline = i === 0 ? "bottom" : last ? "top" : "middle";
        ctx.fillText(t.text, off, at);
      } else {
        ctx.textAlign = i === 0 ? "left" : last ? "right" : "center";
        ctx.textBaseline = outward > 0 ? "top" : "bottom";
        ctx.fillText(t.text, at, off);
      }
    });
    // The title beyond the labels, along the bar (reading upwards beside a vertical one).
    if (cb.title) {
      const reach = base + outward * (tl + g + barLabels + g);
      ctx.save();
      ctx.textAlign = "center";
      if (vertical) {
        ctx.translate(reach, iy + H / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textBaseline = outward > 0 ? "top" : "bottom";
        ctx.fillText(cb.title, 0, 0);
      } else {
        ctx.textBaseline = outward > 0 ? "top" : "bottom";
        ctx.fillText(cb.title, ix + W / 2, reach);
      }
      ctx.restore();
    }
  }
  return out;
}

export const canvasBlob = (c: HTMLCanvasElement) =>
  new Promise<Blob>((resolve, reject) =>
    c.toBlob((b) => (b ? resolve(b) : reject(new Error("Could not encode the PNG"))), "image/png"),
  );
