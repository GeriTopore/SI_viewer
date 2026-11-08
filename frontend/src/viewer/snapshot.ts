import type { Frame } from "../api/arrays";
import type { AxisCal } from "../components/ImageView";
import { niceScaleBar } from "../format";
import { ImageRenderer, type DisplayParams } from "./renderer";

export interface SnapshotLayer {
  key: string;
  frame: Frame;
  display: DisplayParams;
  opacity: number;
}

// Small images are enlarged (whole pixels) to at least about this size.
const MIN_SIZE = 1024;

/**
 * The whole image as displayed (layers composited with their colours,
 * contrast and opacity) at full resolution, with a scale bar, as a PNG.
 * Small images are scaled up by a whole factor so pixels stay sharp.
 */
export function compositePng(
  layers: SnapshotLayer[], width: number, height: number, calX: AxisCal, scaleBar = true,
): Promise<Blob> {
  const s = Math.max(1, Math.min(16, Math.floor(MIN_SIZE / Math.max(width, height))));
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
  if (scaleBar) drawScaleBar(ctx, out.width, out.height, calX.scale / s, calX.units);
  return new Promise((resolve, reject) =>
    out.toBlob((b) => (b ? resolve(b) : reject(new Error("Could not encode the PNG"))), "image/png"),
  );
}

/** A white scale bar, about a fifth of the width, bottom right. */
function drawScaleBar(ctx: CanvasRenderingContext2D, w: number, h: number, unitsPerPixel: number, units: string) {
  const k = w / 5 / 100; // niceScaleBar aims at 100 px
  const bar = niceScaleBar(unitsPerPixel * k, units);
  if (!bar) return;
  const length = bar.pixels * k;
  const margin = Math.round(w * 0.03);
  const thick = Math.max(3, Math.round(w / 150));
  const font = Math.max(12, Math.round(w / 32));
  const x1 = w - margin;
  const x0 = x1 - length;
  const y = h - margin;
  ctx.save();
  ctx.font = `600 ${font}px -apple-system, "Segoe UI", sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(0, 0, 0, 0.7)";
  ctx.lineWidth = Math.max(2, font / 6);
  ctx.strokeText(bar.label, (x0 + x1) / 2, y - thick - font * 0.25);
  ctx.fillStyle = "#fff";
  ctx.fillText(bar.label, (x0 + x1) / 2, y - thick - font * 0.25);
  ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
  ctx.fillRect(x0 - 1, y - thick - 1, length + 2, thick + 2);
  ctx.fillStyle = "#fff";
  ctx.fillRect(x0, y - thick, length, thick);
  ctx.restore();
}

export interface GridSnapshotPanel {
  label: string;
  /** Colour dot beside the label (an element map's colour). */
  colour?: string;
  layer: SnapshotLayer;
  width: number;
  height: number;
  calX: AxisCal;
}

/**
 * Panels laid out in ``cols`` columns as shown in the viewer, each at full
 * resolution (scaled to a common cell) with its label; the scale bar is
 * drawn on the first panel only.
 */
export async function gridPng(panels: GridSnapshotPanel[], cols: number): Promise<Blob> {
  const shots = await Promise.all(
    panels.map(async (p, i) => {
      const blob = await compositePng([p.layer], p.width, p.height, p.calX, i === 0);
      return createImageBitmap(blob);
    }),
  );
  const rows = Math.ceil(shots.length / cols);
  const cellW = Math.max(...shots.map((b) => b.width));
  const cellH = Math.max(...shots.map((b) => b.height));
  const gap = Math.round(Math.max(cellW, cellH) / 100) + 2;
  const out = document.createElement("canvas");
  out.width = cols * cellW + (cols - 1) * gap;
  out.height = rows * cellH + (rows - 1) * gap;
  const ctx = out.getContext("2d")!;
  ctx.fillStyle = "#0f1115";
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.imageSmoothingEnabled = false;
  const font = Math.max(12, Math.round(cellW / 32));
  shots.forEach((b, i) => {
    const x0 = (i % cols) * (cellW + gap);
    const y0 = Math.floor(i / cols) * (cellH + gap);
    const k = Math.min(cellW / b.width, cellH / b.height);
    const w = b.width * k;
    const h = b.height * k;
    const x = x0 + (cellW - w) / 2;
    const y = y0 + (cellH - h) / 2;
    ctx.drawImage(b, x, y, w, h);
    ctx.save();
    ctx.font = `600 ${font}px -apple-system, "Segoe UI", sans-serif`;
    ctx.textBaseline = "top";
    ctx.lineJoin = "round";
    let tx = x + font * 0.6;
    const ty = y + font * 0.5;
    const colour = panels[i].colour;
    if (colour) {
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(tx + font * 0.35, ty + font * 0.55, font * 0.35, 0, 2 * Math.PI);
      ctx.fill();
      tx += font * 1.0;
    }
    ctx.strokeStyle = "rgba(0, 0, 0, 0.7)";
    ctx.lineWidth = Math.max(2, font / 6);
    ctx.strokeText(panels[i].label, tx, ty);
    ctx.fillStyle = "#fff";
    ctx.fillText(panels[i].label, tx, ty);
    ctx.restore();
    b.close();
  });
  return new Promise((resolve, reject) =>
    out.toBlob((b) => (b ? resolve(b) : reject(new Error("Could not encode the PNG"))), "image/png"),
  );
}
