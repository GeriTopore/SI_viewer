import { useEffect } from "react";
import type { DatasetSummary, DeconvolutionOptions } from "../api/client";
import { formatValue } from "../format";
import { allDatasets, useDatasets } from "../state/datasets";
import { energyRange } from "../state/background";
import {
  METHODS, deconvolutionLowLoss, methodProblem, optionsProblem, useDeconvState, useDeconvolution,
} from "../state/deconvolution";
import { snapThreshold } from "../state/thickness";
import { ValueInput } from "./ThicknessSection";

const ACTIVE = new Set(["queued", "running"]);

/** Process tab, EELS spectrum images: plural-scattering deconvolution with the
 * group's low loss. While "Show on spectrum" is on, the deconvolved spectra
 * are drawn dashed over the originals; Apply saves a new spectrum image. On
 * such a spectrum image it shows the settings it was made with, to change and
 * re-run. */
export function DeconvolutionSection({ d }: { d: DatasetSummary }) {
  const st = useDeconvState(d);
  const { options: o } = st;
  const files = useDatasets((s) => s.files);
  const info = useDeconvolution((s) => s.info[d.id]);
  const job = useDeconvolution((s) => s.jobs[d.id]);
  const startError = useDeconvolution((s) => s.startError[d.id]);
  const { patch, setOptions, run, cancel } = useDeconvolution.getState();

  const made = d.deconvolution; // a deconvolved spectrum image
  const source = made ? allDatasets(files).find((x) => x.id === made.source_id) : undefined;
  const target = source ?? d; // what is deconvolved
  const ll = deconvolutionLowLoss(target, files);
  const units = (ll && energyRange(ll)?.units) ?? "";
  const running = !!job && ACTIVE.has(job.status);
  const problem = methodProblem(o.method, o.psf, target, ll) ?? optionsProblem(o);
  const set = (p: Partial<DeconvolutionOptions>) => setOptions(d, p);
  const usesThreshold = o.method !== "richardson_lucy" || o.psf === "zlp";
  const auto = o.threshold === "auto";

  // The preview belongs to this tab: gone when it closes.
  useEffect(() => () => useDeconvolution.getState().patch(d, { on: false }), [d.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const segmented = <T extends string | boolean>(
    label: string, value: T, choices: { id: T; label: string; title: string; problem?: string | null }[],
    onPick: (v: T) => void,
  ) => (
    <div className="q-row">
      <span>{label}</span>
      <div className="segmented process-choice" role="group" aria-label={label}>
        {choices.map((c) => (
          <button
            key={String(c.id)}
            className={value === c.id ? "active" : undefined}
            aria-pressed={value === c.id}
            disabled={running || !!c.problem}
            title={c.problem ?? c.title}
            onClick={() => onPick(c.id)}
          >
            {c.label}
          </button>
        ))}
      </div>
    </div>
  );

  const message = problem ?? (st.on ? info?.error : null) ?? startError;
  return (
    <section className="q-form">
      {made && <p className="muted small q-elements">From {source?.title ?? "its source"}</p>}
      {!made && (
        <label className="q-check" title="Draw the deconvolved spectra over the originals">
          <input type="checkbox" checked={st.on} onChange={(e) => patch(d, { on: e.target.checked })} />
          Show on spectrum
        </label>
      )}
      <label title={METHODS.find((m) => m.id === o.method)?.title}>
        <span>Method</span>
        <select
          value={o.method}
          disabled={running}
          onChange={(e) => {
            const method = e.target.value as DeconvolutionOptions["method"];
            // The low loss itself: Richardson–Lucy by its zero-loss peak.
            set(method === "richardson_lucy" && ll && target.id === ll.id ? { method, psf: "zlp" } : { method });
          }}
        >
          {METHODS.map((m) => {
            const why = methodProblem(m.id, m.id === "richardson_lucy" && ll && target.id === ll.id ? "zlp" : o.psf, target, ll);
            return (
              <option key={m.id} value={m.id} disabled={!!why} title={why ?? m.title}>
                {m.label}
              </option>
            );
          })}
        </select>
      </label>
      {o.method === "richardson_lucy" && (
        <>
          {segmented(
            "PSF", o.psf,
            [
              {
                id: "low_loss", label: "Low loss", title: "Removes plural scattering",
                problem: methodProblem("richardson_lucy", "low_loss", target, ll),
              },
              { id: "zlp", label: "ZLP", title: "Its zero-loss peak only: sharpens the energy resolution" },
            ],
            (psf) => set({ psf }),
          )}
          <label title="More iterations sharpen more and amplify the noise more">
            <span>Iterations</span>
            <ValueInput
              value={o.iterations}
              label="Iterations"
              step={1}
              disabled={running}
              onCommit={(v) => v !== null && set({ iterations: Math.round(v) })}
            />
          </label>
        </>
      )}
      {usesThreshold && (
        <>
          {segmented(
            "ZLP end", auto ? "auto" : "fixed",
            [
              { id: "auto", label: "Auto", title: "Estimated at every pixel" },
              { id: "fixed", label: "Fixed", title: "One energy for every pixel" },
            ],
            (v) => {
              if (v === "auto") set({ threshold: "auto" });
              else if (auto && ll) set({ threshold: snapThreshold(ll, info?.threshold ?? 3) });
            },
          )}
          {auto ? (
            <label title="How far above the zero-loss peak its end is looked for">
              <span>Search window</span>
              <span className="q-inline">
                <ValueInput
                  value={o.window}
                  label="Search window"
                  disabled={running}
                  onCommit={(v) => v !== null && v > 0 && set({ window: v })}
                />
                {units}
              </span>
            </label>
          ) : (
            <label title="Counts below this energy are the zero-loss peak">
              <span>Energy</span>
              <span className="q-inline">
                <ValueInput
                  value={o.threshold as number}
                  label="ZLP end"
                  step={ll ? energyRange(ll)?.step : undefined}
                  disabled={running}
                  onCommit={(v) => v !== null && ll && set({ threshold: snapThreshold(ll, v) })}
                />
                {units}
              </span>
            </label>
          )}
        </>
      )}
      {o.method === "fourier_log" && (
        <label className="q-check" title="Add the zero-loss peak back to the result">
          <input type="checkbox" checked={o.add_zlp} disabled={running} onChange={(e) => set({ add_zlp: e.target.checked })} />
          Keep the zero-loss peak
        </label>
      )}
      {o.method === "fourier_ratio" && (
        <>
          <label title="Width of the smoothing Gaussian; empty: the zero-loss peak's">
            <span>FWHM</span>
            <span className="q-inline">
              <ValueInput
                value={o.fwhm}
                label="FWHM"
                disabled={running}
                onCommit={(v) => set({ fwhm: v !== null && v > 0 ? v : null })}
              />
              {units}
            </span>
          </label>
          <label className="q-check" title="Continue both spectra with a power law before transforming">
            <input
              type="checkbox"
              checked={o.extrapolate}
              disabled={running}
              onChange={(e) => set({ extrapolate: e.target.checked })}
            />
            Extrapolate
          </label>
        </>
      )}
      {!made && st.on && info && !info.error && ((usesThreshold && auto) || (o.method === "fourier_ratio" && o.fwhm === null)) && (
        <p className="muted small q-elements" title="Used for the first spectrum on the plot">
          {[
            usesThreshold && auto && info.threshold != null && `ZLP end ${formatValue(info.threshold)} ${units}`,
            o.method === "fourier_ratio" && o.fwhm === null && info.fwhm != null && `FWHM ${formatValue(info.fwhm)} ${units}`,
          ]
            .filter(Boolean)
            .join(" · ")}
        </p>
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
