import { useEffect, useRef, useState } from "react";
import type { DatasetSummary, MaskType } from "../api/client";
import { ALIGN_STEPS, clampFrame, useAlignment, useStackAlignment, type Alignment } from "../state/alignment";
import { useDatasets } from "../state/datasets";
import { formatValue } from "../format";

const MASKS: { id: MaskType; label: string }[] = [
  { id: "hann", label: "Hann (lowpass, cos²)" },
  { id: "bandpass", label: "Bandpass (sin², 0 at k = 0)" },
  { id: "gaussian", label: "Gaussian (cutoff at 3σ)" },
  { id: "hamming", label: "Hamming" },
  { id: "blackman", label: "Blackman" },
  { id: "none", label: "None" },
];

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
    return (
      <>
        <p className="muted small q-elements">
          Registers the frames to each other (rigidregistration: every pair cross-correlated, outliers corrected) so
          drift can be removed and the frames averaged. You choose the parameters step by step while looking at
          their effect in the viewer.
        </p>
        <button className="process-rebin" onClick={() => openAlignment(d)} disabled={n < 2}>
          {a?.session ? `Resume aligning (${ALIGN_STEPS.find((s) => s.id === a.step)?.label})…` : "Align frames…"}
        </button>
      </>
    );
  }
  return a.step === "frames" ? <FramesControls d={d} a={a} /> : <MaskControls d={d} a={a} />;
}

function FramesControls({ d, a }: { d: DatasetSummary; a: Alignment }) {
  const { patch, start, save } = useAlignment.getState();
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
      frame: clamp(a.frame),
      pair: [clamp(a.pair[0]), clamp(a.pair[1])],
      badFrames: a.badFrames.filter((b) => b >= f && b <= l),
    });
    start(d.id);
  };

  // Bad frames are stored with the session as they change.
  const badKey = a.badFrames.join(",");
  const first_ = useRef(true);
  useEffect(() => {
    if (first_.current) {
      first_.current = false;
      return;
    }
    save(d.id);
  }, [badKey]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <section className="q-form">
      <h4 className="process-heading">Step 1 · Frames</h4>
      <p className="muted small process-note">
        Look through the frames (slider, ← / →) and their FFTs (FFT panel, top right). Leave out frames at the
        start or end that differ (e.g. before the drift settled), and mark damaged or blank frames as bad.
      </p>
      <label title="First and last frame to register">
        Frames
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
      <div className="align-bad-frames">
        <span className="muted small">Bad frames:</span>
        {a.badFrames.length ? (
          a.badFrames.map((f) => (
            <span key={f} className="align-chip">
              <button className="link-button" onClick={() => patch(d.id, { frame: f })} title="Show this frame">
                {f}
              </button>
              <button
                className="icon-button"
                onClick={() => patch(d.id, { badFrames: a.badFrames.filter((b) => b !== f) })}
                title="Use this frame again"
              >
                ×
              </button>
            </span>
          ))
        ) : (
          <span className="muted small">none (use “Mark bad” under the frame)</span>
        )}
      </div>
      <NextStep d={d} a={a} />
    </section>
  );
}

function MaskControls({ d, a }: { d: DatasetSummary; a: Alignment }) {
  const { patch, save } = useAlignment.getState();
  const [i, j] = a.pair;
  const [ns, setNs] = useState(a.compareNs.join(", "));

  // The mask is stored with the session shortly after it last changed.
  const maskKey = `${a.mask.type}:${a.mask.n}`;
  const firstRun = useRef(true);
  useEffect(() => {
    if (firstRun.current) {
      firstRun.current = false;
      return;
    }
    const t = setTimeout(() => save(d.id), SAVE_MS);
    return () => clearTimeout(t);
  }, [maskKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const setPair = (k: 0 | 1, v: string) => {
    const pair: [number, number] = [...a.pair];
    pair[k] = clampFrame(a, Number(v) || 0);
    patch(d.id, { pair });
  };
  const setN = (v: number) => v > 0.5 && patch(d.id, { mask: { ...a.mask, n: Math.min(200, v) } });
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
      <h4 className="process-heading">Step 2 · Fourier mask</h4>
      <p className="muted small process-note">
        The mask weights the frequencies the frames share; features smaller than about n pixels are ignored. It is
        drawn on the FFT panel (top right), with a ring at the cutoff; its Mask button hides it to see the whole
        FFT. For noisy data a cutoff near the first Bragg spots avoids correlating noise; with good signal it can
        sit at the information limit. The cross-correlation should show one sharp, isolated peak; check several
        pairs, especially frames far apart.
      </p>
      <label title="The pair of frames previewed: FFT of the first, cross-correlation of both">
        Frames
        <span className="q-inline">
          <input type="number" min={a.first} max={a.last - 1} value={i} onChange={(e) => setPair(0, e.target.value)} />
          and
          <input type="number" min={a.first} max={a.last - 1} value={j} onChange={(e) => setPair(1, e.target.value)} />
        </span>
      </label>
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
      <label title="Cutoff: features smaller than about n pixels are ignored">
        Cutoff n
        <span className="q-inline process-slider">
          <input
            type="range"
            min={1}
            max={30}
            step={0.1}
            value={a.mask.n}
            onChange={(e) => setN(Number(e.target.value))}
            disabled={a.mask.type === "none"}
          />
          <input
            type="number"
            min={0.6}
            max={200}
            step={0.1}
            value={a.mask.n}
            onChange={(e) => setN(Number(e.target.value))}
            disabled={a.mask.type === "none"}
          />
        </span>
      </label>
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
      <p className="muted small process-note">
        Current mask: {a.mask.type}
        {a.mask.type !== "none" && `, n = ${formatValue(a.mask.n)}`}. Peak finding, the shift matrices and outlier
        correction come next (in development).
      </p>
      <NextStep d={d} a={a} />
    </section>
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
