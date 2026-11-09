import { useEffect, useRef, useState } from "react";
import {
  api, registrationReportUrl, registrationShiftsUrl, type CorrelationType, type DatasetSummary, type MaskSettings, type MaskType, type OutlierSettings,
  type PeakMethod, type PeakSettings,
} from "../api/client";
import { maskLabel } from "../viewer/fourierMask";
import { ALIGN_STEPS, clampFrame, editsOf, useAlignment, useStackAlignment, type Alignment } from "../state/alignment";
import { useDatasets } from "../state/datasets";
import { downloadUrl } from "../download";

const MASKS: { id: MaskType; label: string }[] = [
  { id: "hann", label: "Hann (lowpass, cos²)" },
  { id: "bandpass", label: "Bandpass (sin², 0 at k = 0)" },
  { id: "gaussian", label: "Gaussian (cutoff at 3σ)" },
  { id: "hamming", label: "Hamming" },
  { id: "blackman", label: "Blackman" },
  { id: "elliptical", label: "Elliptical Gaussian" },
  { id: "none", label: "None" },
];

/** A slider with a number box for one parameter. */
function SliderNumber({
  label, title, value, min, max, step, onChange, disabled,
}: {
  label: string;
  title: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  disabled?: boolean;
}) {
  return (
    <label title={title}>
      {label}
      <span className="q-inline process-slider">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          disabled={disabled}
        />
        <input type="number" step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} disabled={disabled} />
      </span>
    </label>
  );
}

// Mask changes are stored with the session this long after the last one.
const SAVE_MS = 400;

/** Opens the alignment workspace for ``d`` (selecting it in the viewer). */
export function openAlignment(d: DatasetSummary) {
  const { selection, clickDataset } = useDatasets.getState();
  if (!selection.includes(d.id)) clickDataset(d);
  useAlignment.getState().open(d.id, d.navigation_shape[0] ?? 1);
}

/** Process tab: drift correction of a stack, step by step. The steps'
 * diagnostics fill the viewer; their parameters are set here. */
export function AlignSection({ d }: { d: DatasetSummary }) {
  const a = useStackAlignment(d.id);
  const isOpen = useAlignment((s) => s.openId === d.id);
  const n = d.navigation_shape[0] ?? 1;

  if (!isOpen || !a) {
    const applied = d.alignment;
    return (
      <>
        {applied && (
          <p className="small q-elements">
            Drift-corrected: {applied.frames.length} of {applied.loaded_frames} frames, cropped to {applied.square[2]} ×{" "}
            {applied.square[3]} px{applied.active ? "" : " (showing the frames as loaded)"}.
          </p>
        )}
        <div className="process-actions">
          <button className="process-rebin" onClick={() => openAlignment(d)} disabled={n < 2}>
            {a?.session ? `Resume aligning (${ALIGN_STEPS.find((s) => s.id === a.step)?.label})…` : "Align frames…"}
          </button>
          {applied && <RemoveAlignment d={d} />}
        </div>
      </>
    );
  }
  switch (a.step) {
    case "mask":
      return <MaskControls d={d} a={a} />;
    case "peaks":
      return <PeaksControls d={d} a={a} />;
    case "matrices":
      return <MatricesControls d={d} a={a} />;
    default:
      return <ResultControls d={d} a={a} />;
  }
}

function ResultControls({ d, a }: { d: DatasetSummary; a: Alignment }) {
  const { patch, close } = useAlignment.getState();
  const replaceFile = useDatasets((s) => s.replaceFile);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const apply = async () => {
    setBusy(true);
    setError(null);
    try {
      replaceFile((await api.applyAlignment(d.id)).file);
      close();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const k = ALIGN_STEPS.findIndex((s) => s.id === a.step);
  return (
    <section className="q-form">
      <h4 className="process-heading">Step 4 · Result</h4>
      <BadFrames d={d} a={a} hint="Mark bad under the frame" />
      {d.alignment && <p className="muted small process-note">Replaces the current alignment.</p>}
      <div className="align-save">
        <span className="muted small">Save</span>
        <button onClick={() => downloadUrl(registrationShiftsUrl(d.id))} disabled={!a.session?.shifts}>
          Shifts (CSV)
        </button>
        <button onClick={() => downloadUrl(registrationReportUrl(d.id))} disabled={!a.session?.shifts}>
          Report (PDF)
        </button>
      </div>
      {error && <p className="field-error">{error}</p>}
      <div className="process-actions">
        <button onClick={() => patch(d.id, { step: ALIGN_STEPS[k - 1].id })}>Back</button>
        <button className="primary" onClick={apply} disabled={busy || !a.session?.shifts}>
          {busy ? "Applying…" : "Apply"}
        </button>
        <button className="link-button" onClick={close}>
          Close
        </button>
      </div>
    </section>
  );
}

function RemoveAlignment({ d }: { d: DatasetSummary }) {
  const replaceFile = useDatasets((s) => s.replaceFile);
  const [error, setError] = useState<string | null>(null);
  return (
    <>
      <button
        className="secondary"
        title="Back to the frames as loaded; the aligned average is removed too"
        onClick={async () => {
          try {
            replaceFile((await api.removeAlignment(d.id)).file);
          } catch (e) {
            setError((e as Error).message);
          }
        }}
      >
        Remove alignment
      </button>
      {error && <p className="field-error">{error}</p>}
    </>
  );
}

/** The frames registered (first … last); changing them retakes the FFTs. */
function FrameRange({ d, a }: { d: DatasetSummary; a: Alignment }) {
  const { patch, start } = useAlignment.getState();
  const n = d.navigation_shape[0] ?? 1;
  const [first, setFirst] = useState(String(a.first));
  const [last, setLast] = useState(String(a.last - 1));
  useEffect(() => {
    setFirst(String(a.first));
    setLast(String(a.last - 1));
  }, [a.first, a.last]);

  const applyRange = () => {
    const f = Math.max(0, Math.min(n - 2, Math.round(Number(first) || 0)));
    const l = Math.max(f + 1, Math.min(n - 1, Math.round(Number(last) || n - 1)));
    setFirst(String(f));
    setLast(String(l));
    if (f === a.first && l + 1 === a.last) return;
    const clamp = (x: number) => Math.min(l, Math.max(f, x));
    patch(d.id, {
      first: f,
      last: l + 1,
      pair: [clamp(a.pair[0]), clamp(a.pair[1])],
      badFrames: a.badFrames.filter((b) => b >= f && b <= l),
      // Edits of pairs with a frame outside the new range are dropped.
      maskEdits: Object.fromEntries(
        Object.entries(a.maskEdits).filter(([k]) => k.split(",").every((x) => Number(x) >= f && Number(x) <= l)),
      ),
    });
    start(d.id);
  };

  return (
    <label title="First and last frame to register: leave out frames at the start or end that differ (e.g. before the drift settled)">
      Register frames
      <span className="q-inline">
        <input
          type="number"
          min={0}
          max={n - 1}
          value={first}
          onChange={(e) => setFirst(e.target.value)}
          onBlur={applyRange}
          onKeyDown={(e) => e.key === "Enter" && applyRange()}
        />
        to
        <input
          type="number"
          min={0}
          max={n - 1}
          value={last}
          onChange={(e) => setLast(e.target.value)}
          onBlur={applyRange}
          onKeyDown={(e) => e.key === "Enter" && applyRange()}
        />
        <span className="muted">of 0–{n - 1}</span>
      </span>
    </label>
  );
}

/** Frames left out of the registration, each with × to use it again, and a
 * field to add one by number. */
function BadFrames({ d, a, hint }: { d: DatasetSummary; a: Alignment; hint: string }) {
  const { patch } = useAlignment.getState();
  const [add, setAdd] = useState("");
  const addFrame = () => {
    const f = Math.round(Number(add));
    if (add.trim() && f >= a.first && f < a.last && !a.badFrames.includes(f)) {
      patch(d.id, { badFrames: [...a.badFrames, f].sort((x, y) => x - y) });
    }
    setAdd("");
  };
  return (
    <div className="align-bad-frames">
      <span className="muted small">Bad frames:</span>
      {a.badFrames.map((f) => (
        <span key={f} className="align-chip">
          {f}
          <button
            className="icon-button"
            onClick={() => patch(d.id, { badFrames: a.badFrames.filter((b) => b !== f) })}
            title="Use this frame again"
          >
            ×
          </button>
        </span>
      ))}
      <input
        type="number"
        className="align-bad-add"
        min={a.first}
        max={a.last - 1}
        placeholder="add"
        value={add}
        onChange={(e) => setAdd(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && addFrame()}
        onBlur={addFrame}
        title={`Leave a frame out (${hint})`}
      />
    </div>
  );
}

function MaskControls({ d, a }: { d: DatasetSummary; a: Alignment }) {
  const { patch, save } = useAlignment.getState();
  const [ns, setNs] = useState(a.compareNs.join(", "));

  // The mask is stored with the session shortly after it last changed.
  const maskKey = JSON.stringify(a.mask);
  const firstRun = useRef(true);
  useEffect(() => {
    if (firstRun.current) {
      firstRun.current = false;
      return;
    }
    const t = setTimeout(() => save(d.id), SAVE_MS);
    return () => clearTimeout(t);
  }, [maskKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const setMask = (m: Partial<MaskSettings>) => patch(d.id, { mask: { ...a.mask, ...m } });
  const cutoff = (field: "n" | "n2") => (v: number) => v > 0.5 && setMask({ [field]: Math.min(200, v) });
  const elliptical = a.mask.type === "elliptical";
  const applyNs = () => {
    const values = ns
      .split(/[,\s]+/)
      .map(Number)
      .filter((v) => v > 0.5 && v <= 200)
      .slice(0, 8);
    if (values.length) patch(d.id, { compareNs: values });
    setNs((values.length ? values : a.compareNs).join(", "));
  };

  return (
    <section className="q-form">
      <h4 className="process-heading">Step 1 · Fourier mask</h4>
      <FrameRange d={d} a={a} />
      <PairInputs d={d} a={a} />
      <label>
        Mask
        <select value={a.mask.type} onChange={(e) => patch(d.id, { mask: { ...a.mask, type: e.target.value as MaskType } })}>
          {MASKS.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
      </label>
      <SliderNumber
        label={elliptical ? "Cutoff n1" : "Cutoff n"}
        title={
          elliptical
            ? "Cutoff along the first axis: features smaller than about n1 pixels along it are ignored"
            : "Cutoff: features smaller than about n pixels are ignored"
        }
        value={a.mask.n}
        min={1}
        max={30}
        step={0.1}
        onChange={cutoff("n")}
        disabled={a.mask.type === "none"}
      />
      {elliptical && (
        <>
          <SliderNumber
            label="Cutoff n2"
            title="Cutoff along the second axis (n1 / n2 is the ellipticity)"
            value={a.mask.n2}
            min={1}
            max={30}
            step={0.1}
            onChange={cutoff("n2")}
          />
          <SliderNumber
            label="Angle θ (°)"
            title="Direction of the first axis (the package's theta, in degrees)"
            value={a.mask.theta}
            min={-90}
            max={90}
            step={0.5}
            onChange={(v) => Number.isFinite(v) && Math.abs(v) <= 360 && setMask({ theta: v })}
          />
        </>
      )}
      <label className="q-check" title="Show the cross-correlation for several cutoffs side by side">
        <input type="checkbox" checked={a.compare} onChange={(e) => patch(d.id, { compare: e.target.checked })} />
        Compare cutoffs n =
        <input
          type="text"
          className="align-ns"
          value={ns}
          onChange={(e) => setNs(e.target.value)}
          onBlur={applyNs}
          onKeyDown={(e) => e.key === "Enter" && applyNs()}
        />
      </label>
      <p className="muted small process-note">Current mask: {maskLabel(a.mask)}.</p>
      <NextStep d={d} a={a} />
    </section>
  );
}

/** The pair of frames previewed (mask and peak steps). */
function PairInputs({ d, a }: { d: DatasetSummary; a: Alignment }) {
  const { patch } = useAlignment.getState();
  const setPair = (k: 0 | 1, v: string) => {
    const pair: [number, number] = [...a.pair];
    pair[k] = clampFrame(a, Number(v) || 0);
    patch(d.id, { pair });
  };
  return (
    <label title="The pair of frames previewed">
      Preview pair
      <span className="q-inline">
        <input type="number" min={a.first} max={a.last - 1} value={a.pair[0]} onChange={(e) => setPair(0, e.target.value)} />
        and
        <input type="number" min={a.first} max={a.last - 1} value={a.pair[1]} onChange={(e) => setPair(1, e.target.value)} />
      </span>
    </label>
  );
}

const PEAK_METHODS: { id: PeakMethod; label: string; title: string }[] = [
  { id: "pixel", label: "Brightest pixel", title: "Fastest; whole-pixel shifts" },
  {
    id: "gf",
    label: "Gaussian fit",
    title: "Fits Gaussians to the brightest pixels and uses the best fit: subpixel, and robust to unit-cell jumps in atomic-resolution data",
  },
  { id: "com", label: "Centre of mass", title: "Centre of mass of the correlation, refined around the first estimate" },
];

function PeaksControls({ d, a }: { d: DatasetSummary; a: Alignment }) {
  const { patch, save } = useAlignment.getState();
  const p = a.peaks;
  const set = (patchP: Partial<PeakSettings>) => patch(d.id, { peaks: { ...p, ...patchP } });

  // Stored with the session shortly after the last change.
  const key = JSON.stringify(p);
  const firstRun = useRef(true);
  useEffect(() => {
    if (firstRun.current) {
      firstRun.current = false;
      return;
    }
    const t = setTimeout(() => save(d.id), SAVE_MS);
    return () => clearTimeout(t);
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps

  const num = (field: "num_peaks" | "sigma_guess" | "window_radius", v: string, min: number, max: number, int = true) => {
    const x = Number(v);
    if (Number.isFinite(x) && x >= min && x <= max) set({ [field]: int ? Math.round(x) : x });
  };

  return (
    <section className="q-form">
      <h4 className="process-heading">Step 2 · Peak finding</h4>
      <PairInputs d={d} a={a} />
      <label title="Cross-correlation is the usual choice; mutual and phase correlation sharpen the peak by normalising the spectrum">
        Correlation
        <select value={p.correlation} onChange={(e) => set({ correlation: e.target.value as CorrelationType })}>
          <option value="cc">Cross-correlation</option>
          <option value="mc">Mutual correlation</option>
          <option value="pc">Phase correlation</option>
        </select>
      </label>
      <div className="segmented process-choice" role="group" aria-label="Peak finding">
        {PEAK_METHODS.map((m) => (
          <button
            key={m.id}
            className={p.method === m.id ? "active" : undefined}
            aria-pressed={p.method === m.id}
            title={m.title}
            onClick={() => set({ method: m.id })}
          >
            {m.label}
          </button>
        ))}
      </div>
      {p.method === "gf" && (
        <>
          <label title="How many of the brightest pixels get a Gaussian fitted">
            Candidates
            <input type="number" min={1} max={20} value={p.num_peaks} onChange={(e) => num("num_peaks", e.target.value, 1, 20)} />
          </label>
          <label title="Starting width of each Gaussian (px)">
            σ guess
            <input
              type="number"
              min={0.5}
              max={100}
              step={0.5}
              value={p.sigma_guess}
              onChange={(e) => num("sigma_guess", e.target.value, 0.1, 100, false)}
            />
          </label>
          <label title="Half-size of the cut-out each Gaussian is fitted to (px)">
            Window radius
            <input type="number" min={2} max={64} value={p.window_radius} onChange={(e) => num("window_radius", e.target.value, 2, 64)} />
          </label>
        </>
      )}
      <NextStep d={d} a={a} />
    </section>
  );
}

const OUTLIER_METHODS: { id: OutlierSettings["method"]; label: string; title: string }[] = [
  {
    id: "transitivity",
    label: "Transitivity",
    title: "Shifts must add up along paths of pairs (i→k→j equals i→j); entries that don't, within the threshold, are masked",
  },
  {
    id: "neighbours",
    label: "Neighbours",
    title: "An entry differing from too many of its neighbouring entries by more than the maximum shift is masked",
  },
  { id: "none", label: "None", title: "Mask no outliers (only bad frames and far pairs)" },
];

function MatricesControls({ d, a }: { d: DatasetSummary; a: Alignment }) {
  const { computeShifts, cancelShifts, patch, save } = useAlignment.getState();
  const n = a.last - a.first;
  const running = !!a.job && (a.job.status === "queued" || a.job.status === "running");
  const shifts = a.session?.shifts;
  const method = PEAK_METHODS.find((m) => m.id === a.peaks.method)?.label;
  const o = a.outliers;
  const set = (patchO: Partial<OutlierSettings>) => patch(d.id, { outliers: { ...o, ...patchO } });

  // Stored with the session shortly after the last change.
  const key = JSON.stringify(o);
  const firstRun = useRef(true);
  useEffect(() => {
    if (firstRun.current) {
      firstRun.current = false;
      return;
    }
    const t = setTimeout(() => save(d.id), SAVE_MS);
    return () => clearTimeout(t);
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps

  const num = (field: "threshold" | "max_paths" | "max_shift", v: string, min: number, max: number, int = false) => {
    const x = Number(v);
    if (Number.isFinite(x) && x >= min && x <= max) set({ [field]: int ? Math.round(x) : x });
  };

  return (
    <section className="q-form">
      <h4 className="process-heading">Step 3 · Shift matrices</h4>
      <p className="muted small process-note">
        With: {maskLabel(a.mask)} mask; {method}
        {a.peaks.correlation !== "cc" ? ` (${a.peaks.correlation === "mc" ? "mutual" : "phase"} correlation)` : ""}.
      </p>
      {running ? (
        <div className="q-job-row">
          <span>{a.job!.message}</span>
          <button className="link-button" onClick={() => cancelShifts(d.id)}>
            Cancel
          </button>
        </div>
      ) : (
        <button className="primary" onClick={() => computeShifts(d.id)}>
          {shifts ? (shifts.stale ? "Compute again (settings changed)" : "Compute again") : "Compute shifts"}
        </button>
      )}
      {running && (
        <div className="q-progress">
          <div style={{ width: `${Math.round(a.job!.progress * 100)}%` }} />
        </div>
      )}
      {a.job?.status === "cancelled" && <p className="muted small process-note">Cancelled.</p>}

      {shifts && (
        <>
          <h4 className="process-heading">Outliers</h4>
          <div className="segmented process-choice" role="group" aria-label="Outlier detection">
            {OUTLIER_METHODS.map((m) => (
              <button
                key={m.id}
                className={o.method === m.id ? "active" : undefined}
                aria-pressed={o.method === m.id}
                title={m.title}
                onClick={() => set({ method: m.id })}
              >
                {m.label}
              </button>
            ))}
          </div>
          {o.method === "transitivity" && (
            <>
              <label title="Allowed mismatch of a shift with the paths through other frames (px); lower masks more">
                Threshold (px)
                <span className="q-inline process-slider">
                  <input
                    type="range"
                    min={0.5}
                    max={30}
                    step={0.5}
                    value={o.threshold}
                    onChange={(e) => num("threshold", e.target.value, 0.1, 1000)}
                  />
                  <input
                    type="number"
                    min={0.1}
                    step={0.5}
                    value={o.threshold}
                    onChange={(e) => num("threshold", e.target.value, 0.1, 1000)}
                  />
                </span>
              </label>
              <label title="How many paths through other frames each shift is checked against">
                Paths
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={o.max_paths}
                  onChange={(e) => num("max_paths", e.target.value, 1, 50, true)}
                />
              </label>
            </>
          )}
          {o.method === "neighbours" && (
            <label title="Allowed difference from neighbouring entries (px); lower masks more">
              Max shift (px)
              <span className="q-inline process-slider">
                <input
                  type="range"
                  min={0.5}
                  max={30}
                  step={0.5}
                  value={o.max_shift}
                  onChange={(e) => num("max_shift", e.target.value, 0.1, 1000)}
                />
                <input
                  type="number"
                  min={0.1}
                  step={0.5}
                  value={o.max_shift}
                  onChange={(e) => num("max_shift", e.target.value, 0.1, 1000)}
                />
              </span>
            </label>
          )}
          <label className="q-check" title="Leave out pairs of frames far apart in time (e.g. when the sample changes)">
            <input
              type="checkbox"
              checked={o.off_diagonal !== null}
              onChange={(e) => set({ off_diagonal: e.target.checked ? Math.max(1, Math.min(n - 1, 10)) : null })}
            />
            Only pairs at most
            <input
              type="number"
              className="align-ns"
              min={1}
              max={n - 1}
              value={o.off_diagonal ?? 10}
              disabled={o.off_diagonal === null}
              onChange={(e) => {
                const v = Math.round(Number(e.target.value));
                if (v >= 1 && v <= n - 1) set({ off_diagonal: v });
              }}
            />
            frames apart
          </label>
          <BadFrames d={d} a={a} hint="their rows and columns are masked" />
          <ManualEdits d={d} a={a} />
        </>
      )}
      <NextStep d={d} a={a} />
    </section>
  );
}

/** Count of the entries set by hand (Edit mask in the matrices), with Clear. */
function ManualEdits({ d, a }: { d: DatasetSummary; a: Alignment }) {
  const { patch } = useAlignment.getState();
  const e = editsOf(a);
  if (!e.keep.length && !e.mask.length) return null;
  return (
    <div className="align-edits">
      <span className="muted small">
        Manual edits: {e.mask.length} masked, {e.keep.length} kept
      </span>
      <button className="link-button" onClick={() => patch(d.id, { maskEdits: {} })} title="Go back to the automatic mask">
        Clear
      </button>
    </div>
  );
}

function NextStep({ d, a }: { d: DatasetSummary; a: Alignment }) {
  const { patch, close } = useAlignment.getState();
  const k = ALIGN_STEPS.findIndex((s) => s.id === a.step);
  const next = ALIGN_STEPS[k + 1];
  return (
    <div className="process-actions">
      {k > 0 && (
        <button onClick={() => patch(d.id, { step: ALIGN_STEPS[k - 1].id })}>Back</button>
      )}
      <button className="primary" disabled={!next?.ready || !a.session} onClick={() => next && patch(d.id, { step: next.id })}>
        Next: {next?.label ?? "done"}
      </button>
      <button className="link-button" onClick={close}>
        Close
      </button>
    </div>
  );
}
