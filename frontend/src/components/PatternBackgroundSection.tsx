import type { DatasetSummary } from "../api/client";
import {
  backgroundJobRunning, backgroundOptionsOf, usePatternBackground, type PatternBackgroundOptions,
} from "../state/patternBackground";
import { segmented } from "../views/FourDView";
import { InPlaceChoice, useInPlace } from "./InPlaceChoice";

type Field = { key: "footprint" | "min_sigma" | "max_sigma" | "h"; label: string; title: string; min: number; max: number; step: number };

const FIELDS: Record<PatternBackgroundOptions["method"], Field[]> = {
  "median kernel": [
    { key: "footprint", label: "Kernel (px)", title: "Median filter size: 2–3 × a disk's diameter", min: 3, max: 255, step: 2 },
  ],
  "difference of gaussians": [
    { key: "min_sigma", label: "Narrow σ (px)", title: "The narrow blur: what counts as a peak", min: 0.5, max: 100, step: 0.5 },
    { key: "max_sigma", label: "Wide σ (px)", title: "The wide blur: the background subtracted", min: 1, max: 500, step: 1 },
  ],
  "radial median": [],
  "h-dome": [
    { key: "h", label: "h", title: "Lowest peak height kept, as a fraction of the maximum", min: 0.01, max: 1, step: 0.05 },
  ],
};

/** Background subtraction of 4D-STEM patterns (diffuse scattering), with
 * pyxem's methods: tried live on the patterns shown, then done to every
 * pattern (a job), in place or as a copy. */
export function PatternBackgroundSection({ d }: { d: DatasetSummary }) {
  const opts = usePatternBackground((s) => backgroundOptionsOf(s, d.id));
  const preview = usePatternBackground((s) => s.preview[d.id] ?? true);
  const job = usePatternBackground((s) => s.jobs[d.id]);
  const error = usePatternBackground((s) => s.errors[d.id] ?? null);
  const { setOptions, setPreview, run, cancel } = usePatternBackground.getState();
  const [inPlace, setInPlace] = useInPlace("background");
  const running = backgroundJobRunning(job);
  const set = (patch: Partial<PatternBackgroundOptions>) => setOptions(d.id, patch);

  return (
    <section className="q-form virtual-form">
      <div className="virtual-row">
        {segmented("Method", [
          { id: "median kernel", label: "Median", title: "Subtract a median-filtered copy of the pattern" },
          { id: "difference of gaussians", label: "DoG", title: "Difference of Gaussians: keeps compact spots" },
          { id: "radial median", label: "Radial", title: "Subtract the median at each radius from the beam" },
          { id: "h-dome", label: "h-dome", title: "Keep only peaks rising more than h; for finding peaks" },
        ], opts.method, (v) => set({ method: v }))}
      </div>
      {FIELDS[opts.method].map(({ key, label, title, min, max, step }) => (
        <label key={key} title={title}>
          <span>{label}</span>
          <input
            type="number"
            min={min}
            max={max}
            step={step}
            value={opts[key]}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (Number.isFinite(v) && e.target.value !== "") set({ [key]: Math.min(max, Math.max(min, v)) });
            }}
          />
        </label>
      ))}
      <label className="q-check-line small" title="Preview the subtraction on the patterns">
        <input type="checkbox" checked={preview} onChange={(e) => setPreview(d.id, e.target.checked)} />
        Show on the patterns
      </label>
      <InPlaceChoice inPlace={inPlace} onChange={setInPlace} disabled={running} what={d.title} />
      <div className="rebin-warning" role="note" hidden={!inPlace}>
        Overwrites the loaded data; the file is unchanged.
      </div>
      {error && <p className="field-error">{error}</p>}
      {running ? (
        <div className="q-job">
          <div className="q-progress" role="progressbar" aria-valuenow={Math.round(job!.progress * 100)}>
            <div style={{ width: `${Math.round(job!.progress * 100)}%` }} />
          </div>
          <div className="q-job-row small">
            <span>{job!.message || "Starting…"} {Math.round(job!.progress * 100)}%</span>
            <button className="link-button" onClick={() => void cancel(d.id)}>Cancel</button>
          </div>
        </div>
      ) : (
        <div className="process-actions">
          <button className="primary" onClick={() => void run(d.id, inPlace)} title="Subtract it from every pattern">
            Subtract
          </button>
        </div>
      )}
      {job?.status === "failed" && !running && <p className="field-error">{job.error}</p>}
      {job?.status === "cancelled" && !running && <p className="muted small">Cancelled.</p>}
    </section>
  );
}
