import { useEffect, useState } from "react";
import { api, dataVersion, type DatasetSummary } from "../api/client";
import { formatValue } from "../format";
import { useDatasets } from "../state/datasets";
import { energyRange } from "../state/background";
import {
  lowLossOf, resultOf, snapThreshold, thicknessProblem, useThickness, useThicknessState,
} from "../state/thickness";
import { isAbort } from "../views/geometry";

const ACTIVE = new Set(["queued", "running"]);
const ESTIMATE_DEBOUNCE_MS = 200;

// Turning "Show on spectrum" off when the section goes away, unless a section
// for the same low loss comes straight back (another member of its group was
// picked).
const pendingOff = new Map<string, ReturnType<typeof setTimeout>>();

/** A number typed in a field, kept as typed until Enter or leaving the field
 * (empty: none). */
export function ValueInput({ value, onCommit, label, step = "any", disabled }: {
  value: number | null; onCommit: (v: number | null) => void; label: string; step?: number | "any"; disabled?: boolean;
}) {
  const shown = (v: number | null) => (v === null ? "" : String(v));
  const [text, setText] = useState(shown(value));
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    if (!editing) setText(shown(value));
  }, [value, editing]);
  const commit = () => {
    setEditing(false);
    const t = text.trim();
    const v = t === "" ? null : Number(t);
    if (v === null || Number.isFinite(v)) onCommit(v);
    else setText(shown(value));
  };
  return (
    <input
      type="number"
      step={step}
      value={text}
      aria-label={label}
      disabled={disabled}
      onFocus={() => setEditing(true)}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
    />
  );
}

/** Process tab, EELS spectrum images and line scans: the thickness of the low
 * loss (of the shown one, or of its group). The zero-loss threshold is drawn
 * on the low-loss spectrum while "Show on spectrum" is on; Apply saves the
 * thickness as a dataset under the low loss. A low loss that has one shows
 * the settings it was made with, to change and re-run. */
export function ThicknessSection({ d }: { d: DatasetSummary }) {
  const files = useDatasets((s) => s.files);
  const ll = lowLossOf(d, files);
  const st = useThicknessState(ll);
  const result = ll ? resultOf(ll, files) : undefined;
  const estimate = useThickness((s) => (ll ? s.estimate[ll.id] : undefined));
  const job = useThickness((s) => (ll ? s.jobs[ll.id] : undefined));
  const startError = useThickness((s) => (ll ? s.startError[ll.id] : null));
  const { patch, run, cancel, setEstimate } = useThickness.getState();

  const range = ll ? energyRange(ll) : null;
  const units = range?.units ?? "";
  const running = !!job && ACTIVE.has(job.status);
  const locked = !ll || running;
  const problem = ll ? thicknessProblem(st, ll) : null;
  const auto = st.threshold === "auto";
  const llId = ll?.id;

  // The threshold is drawn while this section is shown.
  useEffect(() => {
    if (!llId) return;
    clearTimeout(pendingOff.get(llId));
    return () => {
      const found = ll!;
      pendingOff.set(
        llId,
        setTimeout(() => {
          if (useThickness.getState().byId[llId]?.on) useThickness.getState().patch(found, { on: false });
        }, 50),
      );
    };
  }, [llId]); // eslint-disable-line react-hooks/exhaustive-deps

  // The automatic threshold of the summed spectrum (the line drawn while it is automatic).
  const key = ll ? `${dataVersion(ll)}|${st.window}` : "";
  const wanted = !!ll && st.on && auto && st.window > 0;
  useEffect(() => {
    if (!wanted || !llId) return;
    if (useThickness.getState().estimate[llId]?.key === key) return;
    const ac = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const r = await api.thicknessThreshold(llId, st.window, ac.signal);
        setEstimate(llId, { key, value: r.threshold, error: null });
      } catch (e) {
        if (!isAbort(e)) setEstimate(llId, { key, value: null, error: (e as Error).message });
      }
    }, ESTIMATE_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      ac.abort();
    };
  }, [wanted, key, llId]); // eslint-disable-line react-hooks/exhaustive-deps

  const message = problem ?? (wanted ? estimate?.error : null) ?? startError;

  // Fixed starts at the estimate for the summed spectrum, when there is one.
  const switchToFixed = () => {
    if (ll) patch(ll, { threshold: snapThreshold(ll, estimate?.value ?? 5) });
  };

  return (
    <section className="q-form">
      {!ll && <p className="muted small q-elements">No low loss in its group</p>}
      <label className="q-check" title="Draw the zero-loss threshold on the low-loss spectrum; drag it to set a fixed one">
        <input
          type="checkbox"
          checked={st.on}
          disabled={locked}
          onChange={(e) => ll && patch(ll, { on: e.target.checked })}
        />
        Show on spectrum
      </label>
      <div className="q-row">
        <span>Threshold</span>
        <div className="segmented process-choice" role="group" aria-label="Zero-loss threshold">
          <button
            className={auto ? "active" : undefined}
            aria-pressed={auto}
            disabled={locked}
            title="Estimated at every pixel"
            onClick={() => ll && patch(ll, { threshold: "auto" })}
          >
            Auto
          </button>
          <button
            className={!auto ? "active" : undefined}
            aria-pressed={!auto}
            disabled={locked}
            title="One energy for every pixel"
            onClick={() => auto && switchToFixed()}
          >
            Fixed
          </button>
        </div>
      </div>
      {auto ? (
        <>
          <label title="How far above the zero-loss peak its end is looked for">
            <span>Search window</span>
            <span className="q-inline">
              <ValueInput
                value={st.window}
                label="Search window"
                disabled={locked}
                onCommit={(v) => ll && v !== null && v > 0 && patch(ll, { window: v })}
              />
              {units}
            </span>
          </label>
          {st.on && (
            <p className="muted small q-elements" title="Threshold of the summed spectrum">
              Summed spectrum: {estimate?.value != null ? `${formatValue(estimate.value)} ${units}` : "–"}
            </p>
          )}
        </>
      ) : (
        <label title="Counts below this energy are the zero-loss peak">
          <span>Energy</span>
          <span className="q-inline">
            <ValueInput
              value={st.threshold as number}
              label="Threshold"
              step={range?.step}
              disabled={locked}
              onCommit={(v) => ll && v !== null && patch(ll, { threshold: snapThreshold(ll, v) })}
            />
            {units}
          </span>
        </label>
      )}
      <div className="q-row">
        <span>Output</span>
        <div className="segmented process-choice" role="group" aria-label="Output">
          <button
            className={st.output === "relative" ? "active" : undefined}
            aria-pressed={st.output === "relative"}
            disabled={locked}
            title="Thickness relative to the mean free path"
            onClick={() => ll && patch(ll, { output: "relative" })}
          >
            t/λ
          </button>
          <button
            className={st.output === "absolute" ? "active" : undefined}
            aria-pressed={st.output === "absolute"}
            disabled={locked}
            title="Thickness in nm"
            onClick={() => ll && patch(ll, { output: "absolute" })}
          >
            Absolute (nm)
          </button>
        </div>
      </div>
      {st.output === "absolute" && (
        <>
          <div className="q-row">
            <span>Given</span>
            <div className="segmented process-choice" role="group" aria-label="Given">
              <button
                className={st.by === "mean_free_path" ? "active" : undefined}
                aria-pressed={st.by === "mean_free_path"}
                disabled={locked}
                onClick={() => ll && patch(ll, { by: "mean_free_path" })}
              >
                Mean free path
              </button>
              <button
                className={st.by === "density" ? "active" : undefined}
                aria-pressed={st.by === "density"}
                disabled={locked}
                title="Mean free path from the density, beam energy and angles"
                onClick={() => ll && patch(ll, { by: "density" })}
              >
                Density
              </button>
            </div>
          </div>
          {st.by === "mean_free_path" ? (
            <label title="Inelastic mean free path">
              <span>λ</span>
              <span className="q-inline">
                <ValueInput
                  value={st.mean_free_path}
                  label="Mean free path"
                  disabled={locked}
                  onCommit={(v) => ll && patch(ll, { mean_free_path: v })}
                />
                nm
              </span>
            </label>
          ) : (
            <label title="Mass density">
              <span>Density</span>
              <span className="q-inline">
                <ValueInput
                  value={st.density}
                  label="Density"
                  disabled={locked}
                  onCommit={(v) => ll && patch(ll, { density: v })}
                />
                g/cm³
              </span>
            </label>
          )}
        </>
      )}
      {message && <p className="field-error">{message}</p>}
      {running ? (
        <div className="q-job">
          <div className="q-progress" role="progressbar" aria-valuenow={Math.round(job.progress * 100)}>
            <div style={{ width: `${Math.round(job.progress * 100)}%` }} />
          </div>
          <div className="q-job-row small">
            <span>
              {job.message || "Starting…"} {Math.round(job.progress * 100)}%
            </span>
            <button className="link-button" onClick={() => ll && cancel(ll.id)}>
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <button className="primary" disabled={!ll || !!problem} onClick={() => ll && run(ll)}>
          {result ? "Re-run" : "Apply"}
        </button>
      )}
      {job?.status === "cancelled" && !running && <p className="muted small q-elements">Cancelled.</p>}
      {job?.status === "failed" && !running && <p className="field-error">{job.error}</p>}
    </section>
  );
}
