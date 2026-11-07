import { useMemo, useState } from "react";
import { fetchCorrectedMatrices, fetchOutlierMask, fetchShiftMatrices, type Frame } from "../api/arrays";
import { api, type DatasetSummary } from "../api/client";
import { ArrayPanel } from "../components/ArrayPanel";
import type { Interaction, Transform } from "../components/ImageView";
import { useAlignment, type Alignment } from "../state/alignment";
import { colormapGradient } from "../viewer/colormaps";
import { formatValue } from "../format";
import { useLatest } from "./useLatest";
import { DriftPlot } from "../components/DriftPlot";
import { imageGeometry } from "./geometry";

const ACTIVE = new Set(["queued", "running"]);

/** Largest |value| of a matrix (its colour range is ± this). */
const maxAbs = (f: Frame | null) => (f ? f.data.reduce((m, v) => Math.max(m, Math.abs(v)), 0) || 1 : 1);

/** Step 4: every pair's shift, as Δx and Δy matrices (row i, column j: frame
 * j relative to frame i). Hover reads a cell; clicking one opens that pair
 * in Peak finding. */
export function MatricesStep({ ds, a }: { ds: DatasetSummary; a: Alignment }) {
  const { patch, computeShifts, cancelShifts } = useAlignment.getState();
  const g = imageGeometry(ds, "signal");
  const shifts = a.session?.shifts ?? null;
  const running = !!a.job && ACTIVE.has(a.job.status);
  const m = useLatest(shifts ? `${a.sessionStamp}:${a.matricesStamp}` : null, () => fetchShiftMatrices(ds.id));
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);
  const n = a.last - a.first;
  // The outlier mask follows its settings live (computed on the backend).
  const maskKey = shifts ? `${a.sessionStamp}:${a.matricesStamp}:${JSON.stringify(a.outliers)}:${a.badFrames.join(",")}` : null;
  const outlierMask = useLatest(maskKey, () => fetchOutlierMask(ds.id, a.outliers, a.badFrames));
  const kept = a.showOutlierMask && outlierMask.value?.n === n ? outlierMask.value.kept : null;
  // The correction follows the mask live too: masked shifts filled from
  // paths of good ones, then each frame's drift.
  const correction = useLatest(maskKey, async () => {
    const drift = await api.correctShifts(ds.id, a.outliers, a.badFrames);
    return { drift, ...(await fetchCorrectedMatrices(ds.id)) };
  });
  const c = correction.value;
  const known = a.showOutlierMask && c?.dx.width === n ? c.known : null;

  const cell = (p: { x: number; y: number }) => {
    const j = Math.floor(p.x);
    const i = Math.floor(p.y);
    return i >= 0 && j >= 0 && i < n && j < n ? { i: a.first + i, j: a.first + j } : null;
  };
  const interaction: Interaction = {
    cursor: "pointer",
    onHover: (p) => setHover(p ? cell(p) : null),
    onDown: (p) => {
      const c = cell(p);
      if (!c || c.i === c.j) return false;
      patch(ds.id, { pair: [c.i, c.j], step: "peaks" });
      return true;
    },
  };

  if (!shifts || !m.value) {
    return (
      <div className="placeholder">
        {running ? (
          <>
            <p>{a.job!.message || "Computing the shifts…"}</p>
            <div className="q-progress align-progress">
              <div style={{ width: `${Math.round(a.job!.progress * 100)}%` }} />
            </div>
            <button className="secondary small-button" onClick={() => cancelShifts(ds.id)}>
              Cancel
            </button>
          </>
        ) : shifts ? (
          <p className="muted">
            <span className="spinner" aria-hidden /> Loading the shift matrices…
          </p>
        ) : (
          <>
            <p className="muted">
              Correlates every pair of the {n} frames ({(n * (n - 1)) / 2} pairs) with the mask and peak finding set
              in the previous steps.
            </p>
            <button className="primary" onClick={() => computeShifts(ds.id)}>
              Compute shifts
            </button>
          </>
        )}
      </div>
    );
  }

  const { dx, dy } = m.value;
  const at = (f: Frame | undefined, h: { i: number; j: number }) =>
    f ? f.data[(h.i - a.first) * n + h.j - a.first] : undefined;
  const cErr = correction.error;
  // Measured and corrected share each axis's colour range.
  const rangeX = maxAbs(dx);
  const rangeY = maxAbs(dy);

  return (
    <div className="align-body align-matrices">
      {shifts.stale && (
        <div className="align-stale">
          The mask or peak finding changed since these were computed.{" "}
          <button className="link-button" onClick={() => computeShifts(ds.id)} disabled={running}>
            Compute again
          </button>
        </div>
      )}
      <div className="align-matrix-bar">
        <button
          className={`fft-mask-toggle${a.showOutlierMask ? " active" : ""}`}
          aria-pressed={a.showOutlierMask}
          onClick={() => patch(ds.id, { showOutlierMask: !a.showOutlierMask })}
          title={a.showOutlierMask ? "Hide the masks" : "Show the masks"}
        >
          Mask
        </button>
        {outlierMask.value && (
          <span className="muted">
            {outlierMask.value.masked} of {outlierMask.value.pairs} entries masked (
            {Math.round((100 * outlierMask.value.masked) / outlierMask.value.pairs)}%)
          </span>
        )}
        {correction.loading && <span className="muted">Correcting…</span>}
      </div>
      <div className="align-matrix-area">
        <div className="align-matrix-grid">
          <MatrixPanel ds={ds} a={a} name="dx" title="Δx measured" frame={dx} range={rangeX} kept={kept} hover={hover} interaction={interaction} />
          <MatrixPanel ds={ds} a={a} name="dy" title="Δy measured" frame={dy} range={rangeY} kept={kept} hover={hover} interaction={interaction} />
          <MatrixPanel
            ds={ds} a={a} name="dx-c" title="Δx corrected" frame={c?.dx ?? null} range={rangeX} kept={known}
            hover={hover} interaction={interaction} error={cErr}
          />
          <MatrixPanel
            ds={ds} a={a} name="dy-c" title="Δy corrected" frame={c?.dy ?? null} range={rangeY} kept={known}
            hover={hover} interaction={interaction} error={cErr}
          />
          <section className="align-drift">
            <header className="array-panel-title">
              <span>Drift</span>
            </header>
            {c ? (
              <DriftPlot drift={c.drift} calX={g.calX} highlight={hover ? [hover.i, hover.j] : a.pair} />
            ) : (
              <p className={cErr ? "field-error" : "muted"}>{cErr ?? "Correcting…"}</p>
            )}
          </section>
        </div>
      </div>
      <div className="align-result">
        {hover && hover.i !== hover.j ? (
          <>
            Frame {hover.j} relative to {hover.i}: measured <b>Δx {formatValue(at(dx, hover)!)}</b>,{" "}
            <b>Δy {formatValue(at(dy, hover)!)} px</b>
            {outlierMask.value?.kept[(hover.i - a.first) * n + hover.j - a.first] === 0 && (
              <span className="muted"> (masked)</span>
            )}
            {c && at(c.dx, hover) !== undefined && (
              <>
                {" "}
                · corrected <b>Δx {formatValue(at(c.dx, hover)!)}</b>, <b>Δy {formatValue(at(c.dy, hover)!)} px</b>
              </>
            )}
          </>
        ) : null}
        {running && (
          <span className="align-running">
            {" "}
            · computing again {Math.round(a.job!.progress * 100)}%{" "}
            <button className="link-button" onClick={() => cancelShifts(ds.id)}>
              Cancel
            </button>
          </span>
        )}
      </div>
    </div>
  );
}

/** SVG path (matrix-cell units) covering the masked cells. */
function maskPath(kept: Float32Array, n: number): string {
  const parts: string[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) if (kept[i * n + j] === 0) parts.push(`M${j},${i}h1v1h-1z`);
  }
  return parts.join("");
}

function MatrixPanel({
  ds, a, name, title, frame, range, kept, hover, interaction, error,
}: {
  ds: DatasetSummary;
  a: Alignment;
  name: string;
  title: string;
  frame: Frame | null;
  range: number;
  /** Mask (1 kept, 0 masked) to draw, or null to hide it. */
  kept: Float32Array | null;
  hover: { i: number; j: number } | null;
  interaction: Interaction;
  error?: string | null;
}) {
  const n = a.last - a.first;
  const masked = useMemo(() => (kept ? maskPath(kept, n) : ""), [kept, n]);
  const overlay = useMemo(
    () => (t: Transform) => {
      const rect = (i: number, j: number, cls: string) => {
        const [x0, y0] = t.toScreen(j - a.first, i - a.first);
        const [x1, y1] = t.toScreen(j - a.first + 1, i - a.first + 1);
        return <rect key={`${cls}${i},${j}`} x={x0} y={y0} width={x1 - x0} height={y1 - y0} className={cls} />;
      };
      const [ox, oy] = t.toScreen(0, 0);
      return (
        <g className="align-matrix-overlay">
          {masked && <path d={masked} transform={`translate(${ox},${oy}) scale(${t.zoom})`} className="align-masked" />}
          {rect(a.pair[0], a.pair[1], "align-pair-cell")}
          {hover && rect(hover.i, hover.j, "align-hover-cell")}
        </g>
      );
    },
    [a.first, a.pair, hover, masked],
  );
  return (
    <div className="align-matrix">
      <ArrayPanel
        viewKey={`align:${ds.id}:matrix-${name}`}
        title={title}
        layers={[{ key: name, label: title, frame, colormap: "rdbu", range: [-range, range] }]}
        version={`${a.matricesStamp}`}
        width={n}
        height={n}
        loading={frame || error ? null : "Correcting…"}
        error={frame ? null : error}
        interaction={interaction}
        renderOverlay={overlay}
      />
      <div className="align-colorbar">
        <span>−{formatValue(range)}</span>
        <span className="align-colorbar-bar" style={{ background: colormapGradient("rdbu") }} />
        <span>+{formatValue(range)} px</span>
      </div>
    </div>
  );
}
