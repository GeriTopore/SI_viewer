import { useEffect, useMemo, useRef } from "react";
import type { Frame } from "../api/arrays";
import { COLORMAPS, OVERLAY_COLOURS, colormapGradient, isColour } from "../viewer/colormaps";
import { autoRange, fullRange, histogram, percentile, sortedSample } from "../viewer/contrast";
import type { DisplayParams } from "../viewer/renderer";
import { formatValue } from "../format";
import { useUi } from "../state/ui";

interface Props {
  frame: Frame | null;
  display: DisplayParams | undefined;
  onChange: (patch: Partial<DisplayParams>) => void;
}

const BINS = 96;
const SOLID = "__solid";

/** Colormap, histogram with draggable min/max handles, auto contrast and gamma. */
export function ContrastControls({ frame, display, onChange }: Props) {
  const sample = useMemo(() => (frame ? sortedSample(frame.data) : null), [frame]);
  const theme = useUi((s) => s.theme); // histogram colours come from CSS
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragging = useRef<"lo" | "hi" | null>(null);
  // Colour restored when switching back to "solid colour".
  const lastColour = useRef(display && isColour(display.colormap) ? display.colormap : OVERLAY_COLOURS[0]);

  // Histogram range: the bulk of the data, widened to include the current limits.
  const bulk = useMemo(
    (): [number, number] | null =>
      sample ? [percentile(sample, 0.05), percentile(sample, 99.95)] : null,
    [sample],
  );
  let range: [number, number] | null = null;
  if (bulk && display) {
    const a = Math.min(bulk[0], display.lo);
    range = [a, Math.max(bulk[1], display.hi, a + 1e-12)];
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !sample || !range || !display) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(dpr, dpr);
    const css = getComputedStyle(canvas);
    const counts = histogram(sample, range[0], range[1], BINS);
    const peak = Math.sqrt(Math.max(...counts)) || 1;
    const x = (v: number) => ((v - range[0]) / (range[1] - range[0])) * w;

    ctx.fillStyle = css.getPropertyValue("--hist-sel");
    ctx.fillRect(x(display.lo), 0, x(display.hi) - x(display.lo), h);
    ctx.fillStyle = css.getPropertyValue("--hist-bar");
    const bw = w / BINS;
    counts.forEach((c, i) => {
      const bh = (Math.sqrt(c) / peak) * (h - 2);
      ctx.fillRect(i * bw, h - bh, Math.max(1, bw - 0.5), bh);
    });
    ctx.fillStyle = css.getPropertyValue("--accent");
    for (const v of [display.lo, display.hi]) ctx.fillRect(Math.round(x(v)) - 1, 0, 2, h);
  }, [sample, range?.[0], range?.[1], display, theme]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!display) return <div className="contrast-controls" />;

  const valueAt = (clientX: number) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const f = (clientX - rect.left) / rect.width;
    return range ? range[0] + f * (range[1] - range[0]) : 0;
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const v = valueAt(e.clientX);
    dragging.current = Math.abs(v - display.lo) < Math.abs(v - display.hi) ? "lo" : "hi";
    e.currentTarget.setPointerCapture(e.pointerId);
    onPointerMove(e);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!dragging.current) return;
    const v = valueAt(e.clientX);
    if (dragging.current === "lo") onChange({ lo: Math.min(v, display.hi) });
    else onChange({ hi: Math.max(v, display.lo) });
  };

  const gammaSlider = Math.log2(display.gamma) / 2; // gamma 0.25–4 ↔ -1…1

  return (
    <div className="contrast-controls">
      <div className="colormap-picker" title="Colormap">
        <span className="colormap-swatch" style={{ background: colormapGradient(display.colormap) }} />
        <select
          value={isColour(display.colormap) ? SOLID : display.colormap}
          onChange={(e) =>
            onChange({ colormap: e.target.value === SOLID ? lastColour.current : e.target.value })
          }
          aria-label="Colormap"
        >
          {COLORMAPS.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
          <option value={SOLID}>solid colour</option>
        </select>
        {isColour(display.colormap) && (
          <input
            type="color"
            className="colour-input"
            value={display.colormap}
            onChange={(e) => {
              lastColour.current = e.target.value;
              onChange({ colormap: e.target.value });
            }}
            title="Colour at maximum intensity (black at minimum)"
          />
        )}
      </div>

      <div className="histogram">
        <canvas
          ref={canvasRef}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={() => (dragging.current = null)}
          title="Drag to set the display range"
        />
        <div className="histogram-labels">
          <span>{formatValue(display.lo)}</span>
          <span>{formatValue(display.hi)}</span>
        </div>
      </div>

      <div className="contrast-buttons">
        <button
          className="secondary small-button"
          disabled={!sample}
          onClick={() => sample && setRange(autoRange(sample), onChange)}
          title="Clip 0.5 % at each end"
        >
          Auto
        </button>
        <button
          className="secondary small-button"
          disabled={!sample}
          onClick={() => sample && setRange(fullRange(sample), onChange)}
          title="Full data range"
        >
          Min–max
        </button>
      </div>

      <label className="gamma" title="Gamma (double-click to reset)">
        γ {display.gamma.toFixed(2)}
        <input
          type="range"
          min={-1}
          max={1}
          step={0.01}
          value={gammaSlider}
          onChange={(e) => onChange({ gamma: 2 ** (2 * Number(e.target.value)) })}
          onDoubleClick={() => onChange({ gamma: 1 })}
        />
      </label>
    </div>
  );
}

function setRange([lo, hi]: [number, number], onChange: Props["onChange"]) {
  onChange({ lo, hi });
}
