import type { DatasetSummary } from "../api/client";
import { allDatasets, useDatasets } from "../state/datasets";
import { diskJobRunning, diskOptionsOf, useDisks, type DiskOptions } from "../state/disks";
import { segmented } from "../views/FourDView";

const NUMBER_FIELDS: { key: keyof DiskOptions; label: string; title: string; min: number; max: number; step: number }[] = [
  { key: "threshold", label: "Threshold", title: "Lowest correlation with the template a disk may have (−1…1)", min: -1, max: 1, step: 0.05 },
  { key: "min_spacing", label: "Spacing (px)", title: "Closest two disks may be", min: 1, max: 1000, step: 1 },
  { key: "edge", label: "Edge (px)", title: "Disks this close to the pattern's edge are left out", min: 0, max: 1000, step: 1 },
  { key: "max_disks", label: "Max disks", title: "Most disks kept per pattern (the strongest)", min: 1, max: 10000, step: 1 },
];

/** Bragg disk detection on 4D-STEM data: the template (the probe, measured
 * as the bright-field disk of the mean pattern, or a disc), the detection
 * settings tried live on the patterns shown, and finding every pattern's
 * disks (a job) into a "Bragg vectors" table and its "Bragg vector map". */
export function DiskDetectionSection({ d }: { d: DatasetSummary }) {
  const opts = useDisks((s) => diskOptionsOf(s, d.id));
  const preview = useDisks((s) => s.preview[d.id] ?? true);
  const job = useDisks((s) => s.jobs[d.id]);
  const error = useDisks((s) => s.errors[d.id] ?? null);
  const { setOptions, setPreview, run, cancel } = useDisks.getState();
  const running = diskJobRunning(job);
  const files = useDatasets((s) => s.files);
  const found = allDatasets(files).find((x) => x.origin === "disks" && x.parent_id === d.id);
  const set = (patch: Partial<DiskOptions>) => setOptions(d.id, patch);

  return (
    <section className="q-form virtual-form">
      <div className="virtual-row">
        {segmented("Template", [
          { id: "probe", label: "Probe", title: "The bright-field disk of the mean pattern" },
          { id: "disc", label: "Disc", title: "A flat disc" },
        ], opts.template, (v) => set({ template: v }))}
      </div>
      {opts.template === "disc" && (
        <label title="The disc's radius; empty: the bright-field disk's">
          <span>Radius (px)</span>
          <input
            type="number"
            min={0.5}
            step={0.5}
            placeholder="BF disk"
            value={opts.radius ?? ""}
            onChange={(e) => set({ radius: e.target.value === "" ? null : Math.max(0.5, Number(e.target.value)) })}
          />
        </label>
      )}
      {NUMBER_FIELDS.map(({ key, label, title, min, max, step }) => (
        <label key={key} title={title}>
          <span>{label}</span>
          <input
            type="number"
            min={min}
            max={max}
            step={step}
            value={opts[key] as number}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (Number.isFinite(v) && e.target.value !== "") set({ [key]: Math.min(max, Math.max(min, v)) });
            }}
          />
        </label>
      ))}
      <label className="q-check-line small" title="Centre of mass of the correlation around each disk">
        <input type="checkbox" checked={opts.subpixel} onChange={(e) => set({ subpixel: e.target.checked })} />
        Sub-pixel
      </label>
      <label className="q-check-line small" title="Find the disks of the patterns shown as the settings change">
        <input type="checkbox" checked={preview} onChange={(e) => setPreview(d.id, e.target.checked)} />
        Show on the patterns
      </label>
      {found?.disks && (
        <p className="muted small">
          {found.disks.count} disks in {found.disks.positions} patterns ({(found.disks.count / Math.max(1, found.disks.positions)).toFixed(1)} each)
        </p>
      )}
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
          <button
            className="primary"
            onClick={() => void run(d.id)}
            title="Find the disks of every pattern"
          >
            {found ? "Find again" : "Find in every pattern"}
          </button>
        </div>
      )}
      {job?.status === "failed" && !running && <p className="field-error">{job.error}</p>}
      {job?.status === "cancelled" && !running && <p className="muted small">Cancelled.</p>}
    </section>
  );
}
