import { useEffect, useState } from "react";
import type { DatasetSummary } from "../api/client";
import { formatValue } from "../format";
import { energyRange } from "../state/background";
import { allDatasets, useDatasets } from "../state/datasets";
import { ALGORITHMS, algorithmOf, canDecompose, keptOf, snapToAxis, usePca, usePcaState } from "../state/pca";
import { EnergyInput } from "./BackgroundSection";
import { normText, useNorm, useNormSettings } from "../state/normalise";

/** The Process tab's decomposition and denoising: on an EELS or EDS spectrum
 * image, decompose it with one of HyperSpy's algorithms (the window drawn on
 * its spectrum); on the decomposition made, denoise it with as many
 * components as chosen (on the Scree tab or here). */
export function PcaSection({ d }: { d: DatasetSummary }) {
  if (d.origin === "pca") return <DenoiseControls pca={d} />;
  return <DecomposeControls d={d} />;
}

export const hasPcaSection = (d: DatasetSummary) => canDecompose(d) || d.origin === "pca";

function DecomposeControls({ d }: { d: DatasetSummary }) {
  const st = usePcaState(d);
  const { options } = st;
  const job = usePca((s) => s.jobs[d.id]);
  const error = usePca((s) => s.errors[d.id]);
  const { setOptions, run } = usePca.getState();
  const range = energyRange(d);
  const files = useDatasets((s) => s.files);
  const made = allDatasets(files).find(
    (x) => x.origin === "pca" && x.pca?.source_id === d.id && algorithmOf(x) === options.algorithm,
  );
  const centred = !!ALGORITHMS.find((a) => a.id === options.algorithm)?.centred;
  // Normalised spectra: the dataset's normalisation (Process tab, the plot's
  // checkbox), 0–1 unless another was chosen; the same setting as theirs.
  const norm = useNormSettings(d);
  const units = energyRange(d)?.units;

  // The window is drawn on the spectrum while this section is open.
  useEffect(() => {
    usePca.getState().patch(d, { on: true });
    return () => usePca.getState().patch(d, { on: false });
  }, [d.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const running = !!job;
  const setEnd = (end: 0 | 1, v: number) => setOptions(d, end === 0 ? { lo: snapToAxis(d, v) } : { hi: snapToAxis(d, v) });
  const invalid = options.lo !== null && options.hi !== null && options.hi <= options.lo ? "The window needs a start below its end." : null;
  return (
    <section className="q-form">
      <label title="Decomposition algorithm">
        <span>Algorithm</span>
        <select
          value={options.algorithm}
          onChange={(e) => setOptions(d, { algorithm: e.target.value as typeof options.algorithm })}
          disabled={running}
        >
          {ALGORITHMS.map((a) => (
            <option key={a.id} value={a.id}>
              {a.label}
            </option>
          ))}
        </select>
      </label>
      <label title="The energy range decomposed (drag the band on the spectrum)">
        <span>
          <i className="bg-swatch" style={{ background: "#0891b2" }} />
          Window
        </span>
        <span className="q-inline bg-range">
          <EnergyInput value={options.lo ?? 0} step={range?.step ?? 1} label="Window from" onCommit={(v) => setEnd(0, v)} />
          –
          <EnergyInput value={options.hi ?? 0} step={range?.step ?? 1} label="Window to" onCommit={(v) => setEnd(1, v)} />
          {range?.units}
        </span>
      </label>
      {range && (options.lo !== range.e0 || options.hi !== range.e1) && (
        <button className="link-button" onClick={() => setOptions(d, { lo: range.e0, hi: range.e1 })} disabled={running}>
          Whole range
        </button>
      )}
      <label title="Number of components computed">
        <span>Components</span>
        <input
          type="number"
          min={1}
          max={256}
          step={1}
          value={options.components}
          onChange={(e) => {
            const v = Math.round(Number(e.target.value));
            if (v >= 1 && v <= 256) setOptions(d, { components: v });
          }}
          disabled={running}
        />
      </label>
      <label
        className="q-check"
        title="Decompose the normalised spectra"
      >
        <input
          type="checkbox"
          checked={norm.on}
          onChange={(e) => useNorm.getState().set(d, { on: e.target.checked })}
          disabled={running}
        />
        Normalised spectra ({normText(norm, units)})
      </label>
      <label
        className="q-check"
        title={
          centred
            ? "Not with this algorithm: it centres the data"
            : norm.on
              ? "Not with normalised spectra: they are not counts"
              : "Weight by the counting noise (Keenan & Kotula): for counts"
        }
      >
        <input
          type="checkbox"
          checked={options.poisson && !centred && !norm.on}
          onChange={(e) => setOptions(d, { poisson: e.target.checked })}
          disabled={running || centred || norm.on}
        />
        Poisson weighting
      </label>
      {(invalid || error) && <p className="field-error">{invalid ?? error}</p>}
      {job && (
        <p className="muted small">
          {Math.round(job.progress * 100)} %{job.message ? ` · ${job.message}` : ""}
        </p>
      )}
      <button className="primary" onClick={() => run(d)} disabled={running || !!invalid}>
        {running ? "Decomposing…" : made ? "Decompose again" : "Decompose"}
      </button>
      {made && !running && (
        <p className="muted small">
          Replaces {made.title} ({made.pca?.components} components).{" "}
          <button className="link-button" onClick={() => useDatasets.getState().select(made.id)}>
            Show it
          </button>
        </p>
      )}
    </section>
  );
}

function DenoiseControls({ pca }: { pca: DatasetSummary }) {
  const info = pca.pca!;
  const chosen = usePca((s) => s.kept[pca.id]);
  const error = usePca((s) => s.errors[pca.id]);
  const n = keptOf(pca, chosen);
  const [busy, setBusy] = useState(false);
  const files = useDatasets((s) => s.files);
  const source = allDatasets(files).find((x) => x.id === info.source_id);
  const explained = info.explained.slice(0, n).reduce((a, b) => a + b, 0);
  return (
    <section className="q-form">
      <p className="muted small q-elements">
        {ALGORITHMS.find((a) => a.id === algorithmOf(pca))?.label} of {source?.title ?? "its spectrum image"},{" "}
        {formatValue(info.window[0])}–{formatValue(info.window[1])}
        {source && energyRange(source)?.units ? ` ${energyRange(source)!.units}` : ""} · {info.components} components
        {info.poisson ? " · Poisson weighted" : ""}
        {info.normalise ? ` · normalised (${info.normalise.mode === "range" ? "0–1" : info.normalise.mode === "value" ? "value" : "mean"})` : ""}
      </p>
      <label title="Components rebuilt from: click the Scree tab to choose">
        <span>Keep</span>
        <span className="q-inline">
          <input
            type="number"
            min={1}
            max={info.components}
            step={1}
            value={n}
            onChange={(e) => {
              const v = Math.round(Number(e.target.value));
              if (v >= 1 && v <= info.components) usePca.getState().setKept(pca.id, v);
            }}
          />
          {info.explained.length > 0 && <span className="muted">{(explained * 100).toFixed(1)} % of the variance</span>}
        </span>
      </label>
      {error && <p className="field-error">{error}</p>}
      <button
        className="primary"
        disabled={busy}
        onClick={async () => {
          setBusy(true);
          await usePca.getState().denoise(pca, n);
          setBusy(false);
        }}
        title="A new spectrum image, rebuilt from the first components"
      >
        {busy ? "Denoising…" : "Denoise"}
      </button>
    </section>
  );
}
