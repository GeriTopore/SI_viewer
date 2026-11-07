import { useEffect, useRef } from "react";
import { colormapLut } from "../viewer/colormaps";

interface Props {
  /** Rows of values (e.g. a correlation cut-out). */
  data: number[][];
  /** Display range; defaults to the data's min–max. */
  range?: [number, number];
  colormap?: string;
  /** Displayed size (CSS px); pixels are drawn as sharp blocks. */
  size?: number;
  /** A marker at (x, y) in data pixels (pixel i spans [i, i+1)). */
  marker?: { x: number; y: number } | null;
  title?: string;
}

/** A small array drawn on a canvas (no zoom), e.g. a Gaussian fit's cut-out. */
export function MiniImage({ data, range, colormap = "viridis", size = 88, marker, title }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const rows = data.length;
  const cols = data[0]?.length ?? 0;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !rows || !cols) return;
    canvas.width = cols;
    canvas.height = rows;
    const ctx = canvas.getContext("2d")!;
    let [lo, hi] = range ?? [Infinity, -Infinity];
    if (!range) for (const row of data) for (const v of row) [lo, hi] = [Math.min(lo, v), Math.max(hi, v)];
    const lut = colormapLut(colormap);
    const img = ctx.createImageData(cols, rows);
    const scale = hi > lo ? 255 / (hi - lo) : 0;
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const k = Math.max(0, Math.min(255, Math.round((data[y][x] - lo) * scale)));
        const o = (y * cols + x) * 4;
        img.data[o] = lut[k * 4];
        img.data[o + 1] = lut[k * 4 + 1];
        img.data[o + 2] = lut[k * 4 + 2];
        img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [data, range, colormap, rows, cols]);

  return (
    <span className="mini-image" style={{ width: size, height: size }} title={title}>
      <canvas ref={ref} />
      {marker && cols > 0 && (
        <svg viewBox={`0 0 ${cols} ${rows}`} preserveAspectRatio="none">
          <path
            d={`M${marker.x - 1.2},${marker.y} h2.4 M${marker.x},${marker.y - 1.2} v2.4`}
            stroke="#ff453a"
            strokeWidth={0.35}
          />
        </svg>
      )}
    </span>
  );
}
