import { useMemo } from "react";
import { fetchCorrelation, type Frame } from "../api/arrays";
import { api, type DatasetSummary, type PeakCandidate, type PeakResult } from "../api/client";
import { ArrayPanel } from "../components/ArrayPanel";
import type { Transform } from "../components/ImageView";
import { MiniImage } from "../components/MiniImage";
import type { Alignment } from "../state/alignment";
import { formatValue } from "../format";
import { imageGeometry } from "./geometry";
import { useLatest } from "./useLatest";

const CORRELATIONS = { cc: "Cross-correlation", mc: "Mutual correlation", pc: "Phase correlation" };

/** Candidates (white), their fitted centres (+), the chosen peak (red) and
 * the zero-shift centre, offset by a crop's corner. Positions are in
 * correlation pixels (index i spans [i, i+1)). */
function peakMarkers(peak: PeakResult | null, centre: { x: number; y: number }, origin = { x: 0, y: 0 }) {
  return (t: Transform) => {
    const at = (x: number, y: number) => t.toScreen(x - origin.x + 0.5, y - origin.y + 0.5);
    const [cx, cy] = at(centre.x, centre.y);
    return (
      <g className="align-peak">
        <path d={`M${cx - 6},${cy} h12 M${cx},${cy - 6} v12`} className="align-centre" />
        {peak?.candidates.map((c, k) => {
          const [x, y] = at(c.x, c.y);
          const fit = c.fitted && c.fit_x !== undefined ? at(c.fit_x, c.fit_y!) : null;
          return (
            <g key={k}>
              {!c.chosen && <circle cx={x} cy={y} r={5} className="align-candidate" />}
              {fit && <path d={`M${fit[0] - 4},${fit[1]} h8 M${fit[0]},${fit[1] - 4} v8`} className="align-fit" />}
            </g>
          );
        })}
        {peak &&
          (() => {
            const [px, py] = at(peak.x, peak.y);
            return <circle cx={px} cy={py} r={8} />;
          })()}
      </g>
    );
  };
}

/** The square of ``half`` pixels around (x, y), inside the frame. */
function crop(f: Frame, x: number, y: number, half: number) {
  const size = Math.min(2 * half + 1, f.width, f.height);
  const x0 = Math.max(0, Math.min(f.width - size, Math.round(x) - half));
  const y0 = Math.max(0, Math.min(f.height - size, Math.round(y) - half));
  const data = new Float32Array(size * size);
  for (let r = 0; r < size; r++) data.set(f.data.subarray((y0 + r) * f.width + x0, (y0 + r) * f.width + x0 + size), r * size);
  return { frame: { data, width: size, height: size, bin: 1 } as Frame, origin: { x: x0, y: y0 } };
}

/** Step 3: how each pair's shift is read from its correlation, for the pair
 * chosen in the Process tab (or clicked in the shift matrices). */
export function PeaksStep({ ds, a }: { ds: DatasetSummary; a: Alignment }) {
  const g = imageGeometry(ds, "signal");
  const [i, j] = a.pair;
  const { mask, peaks } = a;
  const corrKey = `${a.sessionStamp}:${i}:${j}:${JSON.stringify(mask)}:${peaks.correlation}`;
  const corr = useLatest(corrKey, () => fetchCorrelation(ds.id, i, j, mask, peaks.correlation));
  const peak = useLatest(`${corrKey}:${JSON.stringify(peaks)}`, () => api.peakPreview(ds.id, i, j, mask, peaks));
  const p = peak.value;
  const centre = { x: Math.floor(g.width / 2), y: Math.floor(g.height / 2) };

  // Zoom on the peak: a few fit windows around it.
  const half = Math.max(12, 3 * peaks.window_radius);
  const zoom = useMemo(() => (corr.value && p ? crop(corr.value, p.x, p.y, half) : null), [corr.value, p, half]);
  const fitted = p?.candidates.filter((c) => c.fitted) ?? [];
  const label = CORRELATIONS[peaks.correlation];

  return (
    <div className="align-body align-peaks">
      <div className="align-mask-row">
        <ArrayPanel
          viewKey={`align:${ds.id}:peaks-cc`}
          title={`${label} ${i} × ${j}`}
          layers={[{ key: "cc", label, frame: corr.value, colormap: "viridis" }]}
          version={corr.key ?? ""}
          width={g.width}
          height={g.height}
          loading={corr.value ? null : "Computing…"}
          error={corr.error}
          renderOverlay={peakMarkers(p, centre)}
        />
        <ArrayPanel
          viewKey={`align:${ds.id}:peaks-zoom`}
          title="Around the peak"
          layers={[{ key: "zoom", label: "Around the peak", frame: zoom?.frame ?? null, colormap: "viridis" }]}
          version={zoom ? `${corr.key}:${zoom.origin.x},${zoom.origin.y}` : ""}
          width={zoom?.frame.width ?? 2 * half + 1}
          height={zoom?.frame.height ?? 2 * half + 1}
          loading={zoom ? null : peak.error ? null : "Finding the peak…"}
          error={peak.error}
          renderOverlay={zoom ? peakMarkers(p, centre, zoom.origin) : undefined}
        />
      </div>
      {peaks.method === "gf" && fitted.length > 0 && (
        <div className="align-candidates">
          {fitted.map((c, k) => (
            <CandidateCard key={k} c={c} radius={peaks.window_radius} centre={centre} />
          ))}
        </div>
      )}
      <div className="align-result">
        {p ? (
          <>
            Frame {j} relative to {i}: <b>Δx {formatValue(p.dx)} px</b>, <b>Δy {formatValue(p.dy)} px</b>
          </>
        ) : (
          <span className="muted">{peak.error ?? "Finding the peak…"}</span>
        )}
      </div>
    </div>
  );
}

/** One fitted candidate: its cut-out and the Gaussian fitted to it (as the
 * package's show_Gaussian_fit), with the fit's numbers. */
function CandidateCard({ c, radius, centre }: { c: PeakCandidate; radius: number; centre: { x: number; y: number } }) {
  const data = c.data!;
  const range = useMemo<[number, number]>(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const row of data) for (const v of row) [lo, hi] = [Math.min(lo, v), Math.max(hi, v)];
    return [lo, hi];
  }, [data]);
  const marker = { x: c.fit_x! - c.x + radius + 0.5, y: c.fit_y! - c.y + radius + 0.5 };
  return (
    <div className={`align-candidate-card${c.chosen ? " chosen" : ""}`}>
      <div className="align-candidate-images">
        <MiniImage data={data} range={range} marker={marker} title="Data (cut-out)" />
        <MiniImage data={c.model!} range={range} marker={marker} title="Fitted Gaussian" />
      </div>
      <div className="align-candidate-text">
        <b>{c.chosen ? "Used" : "Candidate"}</b>
        {c.success === false && <span className="q-warning-inline"> fit failed</span>}
        <br />
        Δx {formatValue(c.fit_x! - centre.x)}, Δy {formatValue(c.fit_y! - centre.y)}
        <br />
        <span className="muted">
          A {formatValue(c.amplitude!)} · σ {formatValue(c.sigma![0])}, {formatValue(c.sigma![1])} · offset{" "}
          {formatValue(c.offset!)}
        </span>
      </div>
    </div>
  );
}
