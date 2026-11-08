import { fetchAlignedAverage, fetchAlignedFrame, fetchRegistrationFrame } from "../api/arrays";
import { api, type DatasetSummary } from "../api/client";
import { ArrayPanel } from "../components/ArrayPanel";
import type { Transform } from "../components/ImageView";
import { useShortcuts } from "../keyboard";
import { useAlignment, type Alignment } from "../state/alignment";
import { imageGeometry } from "./geometry";
import { useLatest } from "./useLatest";

/** Step 4: the frames shifted into register, scrolled through, cropped to
 * the square inside every frame (or as loaded, with that square drawn where
 * it falls on each frame), and below them their average. The average feeds
 * the FFT panel. */
export function ResultStep({ ds, a }: { ds: DatasetSummary; a: Alignment }) {
  const { patch } = useAlignment.getState();
  const g = imageGeometry(ds, "signal");
  const shifts = a.session?.shifts ?? null;
  // Correct with the current outlier settings (as step 3 shows them), then
  // shift and average: the backend does this once per correction.
  const key = shifts ? `${a.sessionStamp}:${a.matricesStamp}:${JSON.stringify(a.outliers)}:${a.badFrames.join(",")}` : null;
  const result = useLatest(key, async () => {
    await api.correctShifts(ds.id, a.outliers, a.badFrames);
    return api.registrationResult(ds.id);
  });
  const r = result.value;
  const frames = r?.frames ?? [];
  // The frame shown, or the nearest one kept (bad frames are left out).
  const k = frames.length
    ? frames.reduce((best, f, i) => (Math.abs(f - a.resultFrame) < Math.abs(frames[best] - a.resultFrame) ? i : best), 0)
    : 0;
  const frame = frames[k];
  const setK = (i: number) => frames.length && patch(ds.id, { resultFrame: frames[Math.max(0, Math.min(frames.length - 1, i))] });
  useShortcuts({ ArrowLeft: () => setK(k - 1), ArrowRight: () => setK(k + 1) });

  const aligned = a.resultView === "aligned";
  const shown = useLatest(r && frame !== undefined ? `${result.key}:${a.resultView}:${frame}` : null, () =>
    aligned ? fetchAlignedFrame(ds.id, frame) : fetchRegistrationFrame(ds.id, frame),
  );
  const average = useLatest(r ? result.key : null, () => fetchAlignedAverage(ds.id));

  if (!shifts) {
    return (
      <div className="placeholder">
        <p className="muted">Compute the shifts in step 3 first.</p>
      </div>
    );
  }
  if (!r) {
    return (
      <div className="placeholder">
        {result.error ? (
          <p className="field-error">{result.error}</p>
        ) : (
          <p className="muted">
            <span className="spinner" aria-hidden /> Aligning the frames…
          </p>
        )}
      </div>
    );
  }

  const [sx, sy, side] = r.square;
  // The kept square on frame k as loaded: aligned pixel p is the frame's p + drift.
  const square = (t: Transform) => {
    const d = r.drift.frames.indexOf(frame);
    if (d < 0) return null;
    const [x0, y0] = t.toScreen(sx + r.drift.x[d], sy + r.drift.y[d]);
    const [x1, y1] = t.toScreen(sx + r.drift.x[d] + side, sy + r.drift.y[d] + side);
    return <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} className="align-crop" />;
  };

  return (
    <div className="align-body align-result-step">
      {shifts.stale && (
        <div className="align-stale">
          The mask or peak finding changed since the shifts were computed: compute them again in step 3 to use them.
        </div>
      )}
      <div className="align-result-frames">
        <ArrayPanel
          viewKey={`align:${ds.id}:result-${a.resultView}`}
          title={aligned ? `Frame ${frame}, aligned` : `Frame ${frame}, as loaded`}
          layers={[{ key: "frame", label: `Frame ${frame}`, frame: shown.value }]}
          version={`${result.key}:${a.resultView}`} // one contrast for all frames, so they compare
          width={aligned ? side : g.width}
          height={aligned ? side : g.height}
          calX={g.calX}
          calY={g.calY}
          loading={shown.value ? null : "Loading frame…"}
          error={shown.error}
          renderOverlay={aligned ? undefined : square}
        />
        <div className="align-frame-bar">
          <div className="segmented align-result-view" role="group" aria-label="Frames shown">
            {(["aligned", "original"] as const).map((v) => (
              <button
                key={v}
                className={a.resultView === v ? "active" : undefined}
                aria-pressed={a.resultView === v}
                onClick={() => patch(ds.id, { resultView: v })}
                title={v === "aligned" ? "Frames shifted into register, cropped" : "Frames as loaded, with the kept area"}
              >
                {v === "aligned" ? "Aligned" : "Original"}
              </button>
            ))}
          </div>
          <input
            type="range"
            className="stack-slider"
            min={0}
            max={frames.length - 1}
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            aria-label="Frame"
          />
          <span className="stack-index">
            {frame} ({k + 1} of {frames.length})
          </span>
          <button
            className="secondary small-button"
            onClick={() => patch(ds.id, { badFrames: [...a.badFrames, frame].sort((x, y) => x - y) })}
            disabled={frames.length <= 2}
            title="Leave this frame out: the shifts are corrected and the frames averaged again without it"
          >
            Mark bad
          </button>
        </div>
      </div>
      <ArrayPanel
        viewKey={`align:${ds.id}:result-average`}
        title={`Average of ${frames.length} aligned frames`}
        layers={[{ key: "average", label: "Average", frame: average.value }]}
        version={result.key ?? ""}
        width={side}
        height={side}
        calX={g.calX}
        calY={g.calY}
        loading={average.value ? null : "Averaging…"}
        error={average.error}
        publish
      />
    </div>
  );
}
