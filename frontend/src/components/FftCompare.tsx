import { useLayoutEffect, useRef, useState } from "react";
import { useFftBoxes, useViewFftBoxes } from "../state/fftBoxes";
import { useUi } from "../state/ui";
import { useViewerImage } from "../state/viewerImage";
import { FftView } from "./FftPanel";

const ZOOMS = [1, 2, 4, 8];
const GAP = 8;
// Height of a cell's title and readout lines.
const CAPTION = 44;

/** Columns and square side that fit ``n`` cells in ``w`` × ``h`` best. */
function fit(n: number, w: number, h: number): { cols: number; side: number } {
  let best = { cols: 1, side: 0 };
  for (let cols = 1; cols <= n; cols++) {
    const rows = Math.ceil(n / cols);
    const side = Math.floor(Math.min((w - GAP * (cols - 1)) / cols, (h - GAP * (rows - 1)) / rows - CAPTION));
    if (side > best.side) best = { cols, side };
  }
  return best;
}

/** The FFTs of the view and of each FFT box, side by side beside the image
 * (``view``: the image view's key). Clicking a title makes that region the
 * FFT panel's. */
export function FftCompare({ view }: { view: string }) {
  const image = useViewerImage((s) => s.image);
  const setCompare = useViewerImage((s) => s.setCompare);
  const { boxes, source } = useViewFftBoxes(view);
  const measure = useUi((s) => s.fftMeasure);
  const [zoom, setZoom] = useState(1);
  const regions = [null, ...boxes];

  const gridRef = useRef<HTMLDivElement>(null);
  const [area, setArea] = useState({ w: 0, h: 0 });
  useLayoutEffect(() => {
    const el = gridRef.current!;
    const ro = new ResizeObserver(() => setArea({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const { cols, side } = fit(regions.length, area.w, area.h);

  return (
    <section className="fft-compare" aria-label="FFTs">
      <div className="fft-compare-header">
        <span className="fft-compare-title">FFT</span>
        <div className="segmented fft-zoom" role="group" aria-label="FFT zoom">
          {ZOOMS.map((z) => (
            <button
              key={z}
              className={zoom === z ? "active" : undefined}
              aria-pressed={zoom === z}
              onClick={() => setZoom(z)}
              title={z === 1 ? "Whole FFT (up to the Nyquist frequency)" : `Centre of the FFT, ×${z}`}
            >
              ×{z}
            </button>
          ))}
        </div>
        <button className="icon-button" title="Close" onClick={() => setCompare(false)}>
          ×
        </button>
      </div>
      <div
        ref={gridRef}
        className="fft-compare-grid"
        style={{ gridTemplateColumns: `repeat(${cols}, ${Math.max(side, 0)}px)`, gap: GAP }}
      >
        {side > 0 &&
          regions.map((b) => (
            <div key={b?.id ?? "view"} className="fft-compare-cell" style={{ width: side }}>
              <button
                className={`fft-compare-name${(b?.id ?? null) === source ? " active" : ""}`}
                onClick={() => useFftBoxes.getState().patch(view, { source: b?.id ?? null, selected: b?.id ?? null })}
                title="Show this region in the FFT panel"
              >
                {b && <span className="fft-region-swatch" style={{ background: b.color }} />}
                {b?.name ?? "View"}
              </button>
              <FftView
                image={image}
                box={b}
                zoom={zoom}
                mask={null}
                band={null}
                measure={measure}
                onMask={null}
                onBand={null}
                compact
              />
            </div>
          ))}
      </div>
    </section>
  );
}
