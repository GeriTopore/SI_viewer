import { useEffect, useState } from "react";
import type { BackgroundModel, DatasetSummary } from "../api/client";
import { formatValue } from "../format";
import { allDatasets, useDatasets } from "../state/datasets";
import { energyRange, snapEnergy, useBackground, useBgState, validateWindows } from "../state/background";

const ACTIVE = new Set(["queued", "running"]);

const MODELS: { id: BackgroundModel; label: string; title: string }[] = [
  { id: "power_law", label: "Power law", title: "A·E^−r: the usual model for the pre-edge background" },
  { id: "exponential", label: "Exponential", title: "A·exp(−b·E)" },
  { id: "polynomial", label: "Polynomial", title: "A polynomial of the energy loss" },
  { id: "offset", label: "Offset", title: "A constant: the mean of the background window" },
];

/** A limit of a window, typed in energy units; kept as typed until Enter or
 * leaving the field, then snapped to a channel. */
function EnergyInput({ value, step, onCommit, label }: {
  value: number; step: number; onCommit: (v: number) => void; label: string;
}) {
  const [text, setText] = useState(String(value));
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    if (!editing) setText(String(value));
  }, [value, editing]);
  const commit = () => {
    setEditing(false);
    const v = Number(text);
    if (text.trim() !== "" && Number.isFinite(v)) onCommit(v);
    else setText(String(value));
  };
  return (
    <input
      type="number"
      step={step}
      value={text}
      aria-label={label}
      onFocus={() => setEditing(true)}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
    />
  );
}

/** Process tab, EELS spectra and spectrum images: background removal. The
 * two windows and the preview are drawn on the spectrum while "Show on
 * spectrum" is on; Apply saves a new spectrum image of the signal window with
 * the background subtracted. On such a spectrum image it shows the settings
 * it was made with, to change and re-run. */
export function BackgroundSection({ d }: { d: DatasetSummary }) {
  const st = useBgState(d);
  const { options } = st;
  const info = useBackground((s) => s.info[d.id]);
  const job = useBackground((s) => s.jobs[d.id]);
  const startError = useBackground((s) => s.startError[d.id]);
  const files = useDatasets((s) => s.files);
  const { patch, setOptions, run, cancel } = useBackground.getState();

  const made = d.background; // a background-removed spectrum image
  const source = made ? allDatasets(files).find((x) => x.id === made.source_id) : undefined;
  const range = energyRange(source ?? d);
  const units = made?.units ?? range?.units ?? "";
  const running = !!job && ACTIVE.has(job.status);
  const problem = validateWindows(options);

  // The windows and preview belong to this tab: gone when it closes.
  useEffect(() => () => useBackground.getState().patch(d, { on: false }), [d.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const limit = (v: number) => {
    if (!range) return v;
    return snapEnergy(range.e0, range.step, Math.min(range.e1, Math.max(range.e0, v)));
  };
  const setWindow = (key: "background" | "signal", end: 0 | 1, v: number) => {
    const w: [number, number] = [...options[key]];
    w[end] = limit(v);
    setOptions(d, { [key]: w });
  };

  const windowRow = (key: "background" | "signal", label: string, colour: string, hint: string) => (
    <label title={hint}>
      <span>
        <i className="bg-swatch" style={{ background: colour }} />
        {label}
      </span>
      <span className="q-inline bg-range">
        <EnergyInput value={options[key][0]} step={range?.step ?? 1} label={`${label} from`} onCommit={(v) => setWindow(key, 0, v)} />
        –
        <EnergyInput value={options[key][1]} step={range?.step ?? 1} label={`${label} to`} onCommit={(v) => setWindow(key, 1, v)} />
        {units}
      </span>
    </label>
  );

  const message = problem ?? info?.error ?? startError;
  return (
    <section className="q-form">
      {made && <p className="muted small q-elements">From {source?.title ?? "its source"}</p>}
      {!made && (
        <label className="q-check" title="Draw the Background and Signal windows and the fitted backgrounds on the spectrum">
          <input type="checkbox" checked={st.on} onChange={(e) => patch(d, { on: e.target.checked })} />
          Show on spectrum
        </label>
      )}
      <label title="Model of the background under the edge">
        <span>Model</span>
        <select value={options.model} onChange={(e) => setOptions(d, { model: e.target.value as BackgroundModel })} disabled={running}>
          {MODELS.map((m) => (
            <option key={m.id} value={m.id} title={m.title}>
              {m.label}
            </option>
          ))}
        </select>
      </label>
      {options.model === "polynomial" && (
        <label title="Degree of the polynomial">
          <span>Order</span>
          <select value={options.order} onChange={(e) => setOptions(d, { order: Number(e.target.value) })} disabled={running}>
            {[1, 2, 3, 4, 5, 6].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
      )}
      <div className="q-row">
        <span>Fit</span>
        <div className="segmented process-choice" role="group" aria-label="Fit">
          <button
            className={options.fast ? "active" : undefined}
            aria-pressed={options.fast}
            title="Fast: a closed-form estimate, instant"
            onClick={() => setOptions(d, { fast: true })}
          >
            Fast
          </button>
          <button
            className={!options.fast ? "active" : undefined}
            aria-pressed={!options.fast}
            title="Full: a least-squares fit at every pixel; slower"
            onClick={() => setOptions(d, { fast: false })}
          >
            Full
          </button>
        </div>
      </div>
      {windowRow("background", "Background", "#f97316", "The range the model is fitted to, before the edge")}
      {windowRow("signal", "Signal", "#22c55e", "The range kept, starting where the background window ends or later")}
      {!made && st.on && (
        <>
          <label className="q-check" title="Show the spectra minus their fitted backgrounds, over the signal window">
            <input type="checkbox" checked={st.subtracted} onChange={(e) => patch(d, { subtracted: e.target.checked })} />
            Subtracted
          </label>
          {options.model === "power_law" && (
            <p className="muted small q-elements" title="Exponent of the fitted power law A·E^−r">
              r at the cursor: {info?.r != null ? formatValue(info.r) : "–"}
            </p>
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
            <button className="link-button" onClick={() => cancel(d.id)}>
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <button className="primary" disabled={!!problem} onClick={() => run(d, !!made)}>
          {made ? "Re-run" : "Apply"}
        </button>
      )}
      {job?.status === "cancelled" && !running && <p className="muted small q-elements">Cancelled.</p>}
      {job?.status === "failed" && !running && <p className="field-error">{job.error}</p>}
    </section>
  );
}
