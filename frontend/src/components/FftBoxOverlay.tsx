import { useViewFftBoxes } from "../state/fftBoxes";
import type { Transform } from "./ImageView";

const HANDLE = 8;

/** The FFT boxes of an image view: the selected one with corner handles, the
 * one the FFT panel shows drawn solid (the others dashed). */
export function FftBoxOverlay({ t, view }: { t: Transform; view: string }) {
  const { boxes, selected, source } = useViewFftBoxes(view);
  return (
    <g className="fft-boxes">
      {boxes.map((b) => {
        const [x0, y0] = t.toScreen(b.x, b.y);
        const [x1, y1] = t.toScreen(b.x + b.size, b.y + b.size);
        const corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
        return (
          <g key={b.id} className={`fft-box${b.id === source ? " source" : ""}`} style={{ color: b.color }}>
            <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} />
            <text className="roi-label" x={x0 + 3} y={y0 - 4}>
              {b.name}
            </text>
            {b.id === selected &&
              corners.map(([x, y], i) => (
                <rect key={i} className="roi-handle" x={x - HANDLE / 2} y={y - HANDLE / 2} width={HANDLE} height={HANDLE} />
              ))}
          </g>
        );
      })}
    </g>
  );
}
