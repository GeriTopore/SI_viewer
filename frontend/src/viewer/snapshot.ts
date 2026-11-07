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
