import { useEffect, useState } from "react";
import { api, type DatasetSummary, type EelsGos, type FitSource } from "../api/client";
import { downloadCsv, safeFileName } from "../download";
import { formatValue, isLineScan } from "../format";
import { useDatasets } from "../state/datasets";
import { EELS_FIT_DEFAULTS, useEelsFit } from "../state/eelsFit";
import { useLayers } from "../state/layers";
import { datasetKey, useQuant, type SourceChoice } from "../state/quant";
import { shapeToRegion, useSi, useSiState } from "../state/si";
import { lowLossOf } from "../state/thickness";
import { edgeLabel } from "../views/si/xray";

const ACTIVE = new Set(["queued", "running"]);

const GOS: { id: EelsGos; label: string; title: string }[] = [
  { id: "dft", label: "DFT", title: "Segger, Guzzinati & Kohl" },
  { id: "dirac", label: "Dirac", title: "Relativistic (Zhang et al.)" },
  { id: "hydrogenic", label: "Hydrogenic", title: "K and L edges only" },
];

/** An energy typed in eV; empty means none. Committed on Enter or on leaving the field. */
function OptionalEnergy({ value, placeholder, label, onCommit, disabled }: {
  value: number | null; placeholder: string; label: string; onCommit: (v: number | null) => void; disabled?: boolean;
}) {
  const [text, setText] = useState(value === null ? "" : String(value));
  useEffect(() => setText(value === null ? "" : String(value)), [value]);
  const commit = () => {
    const v = Number(text);
    if (text.trim() === "") onCommit(null);
    else if (Number.isFinite(v)) onCommit(v);
    else setText(value === null ? "" : String(value));
  };
  return (
    <input
      type="number"
      value={text}
      placeholder={placeholder}
      aria-label={label}
      disabled={disabled}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
    />
  );
}

/** Quantify tab, EELS: fit one spectrum with exSpy's model (a power-law
 * background and the edges of the chosen elements, shaped by their
 * cross-sections), and every pixel of a 2D spectrum image as a job. */
export function EelsFitPanel({ ds }: { ds: DatasetSummary }) {
  const key = datasetKey(ds);
  const options = useEelsFit((s) => s.options[ds.id]) ?? EELS_FIT_DEFAULTS;
  const fits = useEelsFit((s) => s.fits[key]);
  const chosen = useEelsFit((s) => s.sources[key]);
  const { setOptions, setFits, setSources } = useEelsFit.getState();
  // The fit whose numbers are listed (its source).
  const [listed, setListed] = useState<SourceChoice | null>(null);
  const fit = fits?.find((f) => f.source === listed) ?? fits?.[0];
  const q = useQuant();
  const si = useSiState(ds.id);
  const cursorSize = useSi((s) => s.cursorSize);
  const files = useDatasets((s) => s.files);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isSi = ds.kind === "spectrum_image";
  const lineScan = isLineScan(ds);
  // The spectra to fit (regions that are gone are dropped).
  const choices = (isSi ? (chosen ?? ["all"]) : ["all"]).filter(
    (c) => c === "all" || c === "cursor" || si.rois.some((r) => r.id === c),
  );
  const toggleChoice = (c: SourceChoice, on: boolean) =>
    setSources(key, on ? [...choices.filter((x) => x !== c), c] : choices.filter((x) => x !== c));
  const elements = ds.edges?.elements ?? [];
  const axis = ds.axes.find((a) => !a.navigate);
  const e0 = axis ? axis.offset : 0;
  const e1 = axis ? axis.offset + axis.scale * (axis.size - 1) : 0;
  // The model of a core loss is convolved with its group's low loss.
  const lowLoss = ds.eels && ds.eels.role !== "low_loss" ? lowLossOf(ds, files) : undefined;

  const sourceFor = (c: SourceChoice): { source: FitSource; label: string } | null => {
    if (!isSi || c === "all") return { source: { type: "all" }, label: !isSi ? ds.title : lineScan ? "All positions" : "All pixels" };
    if (c === "cursor") {
      if (!si.cursor) return null;
      const { x, y } = si.cursor;
      const size = lineScan ? 1 : cursorSize;
      return {
        source: { type: "pixel", x, y, size },
        label: lineScan ? `Cursor ${x}` : `Cursor ${x}, ${y}${size > 1 ? ` (${size}×${size})` : ""}`,
      };
    }
    const roi = si.rois.find((r) => r.id === c);
    return roi ? { source: { type: "region", region: shapeToRegion(roi.shape) }, label: roi.name } : null;
  };

  // Fits every chosen spectrum; the ones that fail are named in the error.
  const runFit = async (patch: Partial<typeof options> = {}) => {
    const srcs = choices.flatMap((c) => {
      const src = sourceFor(c);
      return src ? [{ choice: c, ...src }] : [];
    });
    if (!srcs.length) {
      setError("Choose a spectrum to fit.");
      return;
    }
    const used = { ...options, ...patch };
    setBusy(true);
    setError(null);
    const done = await Promise.allSettled(srcs.map((s) => api.eelsFit(ds.id, s.source, used)));
    const ok = srcs.flatMap((s, i) => {
      const r = done[i];
      return r.status === "fulfilled" ? [{ source: s.choice, label: s.label, result: r.value }] : [];
    });
    const failed = srcs.flatMap((s, i) => {
      const r = done[i];
      return r.status === "rejected" ? [`${s.label}: ${(r.reason as Error).message}`] : [];
    });
    if (ok.length) setFits(key, ok);
    if (failed.length) setError(failed.join("\n"));
    setBusy(false);
  };

  // A typed onset is kept with the settings and refitted at once.
  const setOnset = (element: string, value: number | null) => {
    const onsets = { ...options.onsets };
    if (value === null) delete onsets[element];
    else onsets[element] = value;
    setOptions(ds.id, { onsets });
    void runFit({ onsets });
  };

  // Every spectrum fitted, a row per element of each.
  const saveTable = () =>
    fit &&
    downloadCsv(
      [
        ["cross-sections", fit.result.gos],
        ["convolved with the low loss", fit.result.convolved ? "yes" : "no"],
        [],
        ["spectrum", "pixels", "element", "edge", "onset (eV)", "intensity", "error", "at%"],
        ...fits!.flatMap((f) =>
          f.result.rows.map((r) => [f.label, f.result.pixels, r.element, r.edge, r.onset, r.intensity, r.std, r.atomic_percent]),
        ),
      ],
      `${safeFileName(ds.title)}_eels_fit.csv`,
    );

  return (
    <div className="quantify">
      <section>
        <h3 className="q-title">{ds.title}</h3>
        <p className="muted small q-elements">Elements: {elements.length ? elements.join(", ") : "none"}</p>
      </section>

      <section className="q-form">
        {isSi && (
          <div className="q-sources" role="group" aria-label="Spectra to fit">
            <span>Spectra</span>
            <div className="q-source-list">
              {[
                { id: "all", name: lineScan ? "All positions" : "All pixels", color: null, disabled: false },
                {
                  id: "cursor",
                  name: `Cursor${si.cursor ? ` (${lineScan ? si.cursor.x : `${si.cursor.x}, ${si.cursor.y}`})` : ""}`,
                  color: null,
                  disabled: !si.cursor,
                },
                ...si.rois.map((r) => ({ id: r.id, name: r.name, color: r.color, disabled: false })),
              ].map((c) => (
                <label key={c.id} className="q-check">
                  <input
                    type="checkbox"
                    checked={choices.includes(c.id)}
                    onChange={(e) => toggleChoice(c.id, e.target.checked)}
                    disabled={busy || c.disabled}
                  />
                  {c.color && <span className="series-swatch" style={{ background: c.color }} />}
                  {c.name}
                </label>
              ))}
            </div>
          </div>
        )}
        <label title="Generalised oscillator strengths the edges are computed from">
          Cross-sections
          <select value={options.gos} onChange={(e) => setOptions(ds.id, { gos: e.target.value as EelsGos })} disabled={busy}>
            {GOS.map((g) => (
              <option key={g.id} value={g.id} title={g.title}>
                {g.label}
              </option>
            ))}
          </select>
        </label>
        <label title="Energy range fitted; empty: to the end of the spectrum">
          <span>Range</span>
          <span className="q-inline bg-range">
            <OptionalEnergy
              value={options.lo}
              placeholder={formatValue(e0)}
              label="Fit from"
              disabled={busy}
              onCommit={(lo) => setOptions(ds.id, { lo })}
            />
            –
            <OptionalEnergy
              value={options.hi}
              placeholder={formatValue(e1)}
              label="Fit to"
              disabled={busy}
              onCommit={(hi) => setOptions(ds.id, { hi })}
            />
            eV
          </span>
        </label>
        <label className="q-check" title="Fit the fine structure after each onset with a spline">
          <input
            type="checkbox"
            checked={options.fine_structure}
            onChange={(e) => setOptions(ds.id, { fine_structure: e.target.checked })}
            disabled={busy}
          />
          Fine structure (spline)
        </label>
        {options.fine_structure && (
          <label title="Width of the fine structure after each onset; narrowed where edges overlap">
            Width
            <span className="q-inline">
              <input
                type="number"
                min={1}
                step={1}
                value={options.fine_structure_width}
                onChange={(e) => setOptions(ds.id, { fine_structure_width: Math.max(1, Number(e.target.value) || 30) })}
                disabled={busy}
              />
              eV
            </span>
          </label>
        )}
        {lowLoss && (
          <label className="q-check" title={`Convolve the model with ${lowLoss.title}`}>
            <input
              type="checkbox"
              checked={options.convolve}
              onChange={(e) => setOptions(ds.id, { convolve: e.target.checked })}
              disabled={busy}
            />
            Convolve with the low loss
          </label>
        )}
        <button className="primary" onClick={() => runFit()} disabled={busy || !elements.length || !choices.length}>
          {busy ? "Fitting…" : fit ? "Refit" : choices.length > 1 ? `Fit ${choices.length} spectra` : "Fit spectrum"}
        </button>
        {error && <p className="field-error">{error}</p>}
      </section>

      {fit && (
        <>
          <section className="q-summary small">
            <div>
              <span className="muted">Fitted:</span>{" "}
              {fits!.length > 1 ? (
                <select value={fit.source} onChange={(e) => setListed(e.target.value)} aria-label="Fit listed">
                  {fits!.map((f) => (
                    <option key={f.source} value={f.source}>
                      {f.label}
                    </option>
                  ))}
                </select>
              ) : (
                fit.label
              )}
              {fit.result.pixels > 1 && (
                <span className="muted">
                  {" "}
                  · {fit.result.pixels.toLocaleString()} {lineScan ? "positions" : "px"}
                </span>
              )}
            </div>
            <div>
              {fit.result.residual_ratio !== null && (
                <>
                  <span className="muted">Residual:</span> {(100 * fit.result.residual_ratio).toFixed(1)}%
                </>
              )}
              {fit.result.background_r !== null && (
                <>
                  {" "}
                  · <span className="muted">r:</span> {fit.result.background_r.toFixed(2)}
                </>
              )}
              {fit.result.convolved && <span className="muted"> · convolved</span>}
            </div>
            {fit.result.skipped.length > 0 && (
              <div className="muted">No edge in range: {fit.result.skipped.join(", ")}</div>
            )}
            <div className="q-toggles">
              <label>
                <input type="checkbox" checked={q.showFit} onChange={(e) => q.setShowFit(e.target.checked)} />
                Show fit
              </label>
              <label>
                <input type="checkbox" checked={q.showComponents} onChange={(e) => q.setShowComponents(e.target.checked)} />
                Show elements
              </label>
            </div>
          </section>

          <section>
            <table className="q-table">
              <thead>
                <tr>
                  <th>Edge</th>
                  <th title="Onset of the edge (eV); type one where the database's is off">Onset</th>
                  <th>Intensity</th>
                  <th>±</th>
                  <th>at%</th>
                </tr>
              </thead>
              <tbody>
                {fit.result.rows.map((r) => (
                  <tr key={r.element}>
                    <td>
                      {r.element} {edgeLabel(r.edge.split("_")[1] ?? "")}
                    </td>
                    <td>
                      <OptionalEnergy
                        value={options.onsets[r.element] ?? null}
                        placeholder={formatValue(r.onset)}
                        label={`Onset of ${r.element}`}
                        disabled={busy}
                        onCommit={(v) => v !== (options.onsets[r.element] ?? null) && setOnset(r.element, v)}
                      />
                    </td>
                    <td className="num">{formatValue(r.intensity)}</td>
                    <td className="num">{r.std === null ? "–" : formatValue(r.std)}</td>
                    <td className="num">{r.atomic_percent === null ? "–" : r.atomic_percent.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted small">
              <button className="link-button" onClick={saveTable} title="Save this table as CSV">
                Save CSV
              </button>
            </p>
          </section>
        </>
      )}

      {isSi && !lineScan && <EelsFitMaps ds={ds} />}
    </div>
  );
}

/** Fit every pixel as a background job: a fitted map per element (and at% maps). */
function EelsFitMaps({ ds }: { ds: DatasetSummary }) {
  const key = datasetKey(ds);
  const options = useEelsFit((s) => s.options[ds.id]) ?? EELS_FIT_DEFAULTS;
  const job = useEelsFit((s) => s.jobs[key]);
  const composition = useEelsFit((s) => s.composition);
  const { setJob, setComposition } = useEelsFit.getState();
  const running = !!job && ACTIVE.has(job.status);
  const showFit = useQuant((s) => s.showFit);
  const showComponents = useQuant((s) => s.showComponents);
  const { setShowFit, setShowComponents } = useQuant.getState();

  // Poll the job while it runs; on completion bring in the new maps.
  useEffect(() => {
    if (!job || !ACTIVE.has(job.status)) return;
    const timer = setInterval(async () => {
      try {
        const next = await api.getJob(job.id);
        useEelsFit.getState().setJob(key, next);
        if (next.status === "done" && next.result) useDatasets.getState().replaceFile(next.result.file);
      } catch {
        /* backend busy or restarted: try again */
      }
    }, 300);
    return () => clearInterval(timer);
  }, [job?.id, job?.status, key]); // eslint-disable-line react-hooks/exhaustive-deps

  const start = async () => {
    try {
      setJob(key, await api.startEelsFitMaps(ds.id, options, composition));
    } catch (e) {
      useLayers.getState().notify(`Could not start: ${(e as Error).message}`);
    }
  };

  return (
    <section className="q-maps">
      <h3>Maps of every pixel</h3>
      <label className="q-check-line">
        <input type="checkbox" checked={composition} onChange={(e) => setComposition(e.target.checked)} disabled={running} />
        at% maps
      </label>
      {running ? (
        <div className="q-job">
          <div className="q-progress" role="progressbar" aria-valuenow={Math.round(job.progress * 100)}>
            <div style={{ width: `${Math.round(job.progress * 100)}%` }} />
          </div>
          <div className="q-job-row small">
            <span>
              {job.message || "Starting…"} {Math.round(job.progress * 100)}%
            </span>
            <button className="link-button" onClick={() => api.cancelJob(job.id).then((j) => setJob(key, j))}>
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <button className="secondary" onClick={start} disabled={!ds.edges?.elements.length}>
          {job?.status === "done" ? "Fit every pixel again" : "Fit every pixel"}
        </button>
      )}
      {job?.status === "done" && job.result?.skipped?.length ? (
        <p className="muted small">No edge in range: {job.result.skipped.join(", ")}</p>
      ) : null}
      {job?.status === "cancelled" && <p className="muted small">Cancelled.</p>}
      {job?.status === "failed" && <p className="field-error">{job.error}</p>}
      {ds.pixel_fit && !running && (
        <div className="q-toggles small">
          <label title="The cursor spectrum shows the fit of its pixels">
            <input type="checkbox" checked={showFit} onChange={(e) => setShowFit(e.target.checked)} />
            Show fit
          </label>
          <label>
            <input type="checkbox" checked={showComponents} onChange={(e) => setShowComponents(e.target.checked)} />
            Show elements
          </label>
        </div>
      )}
    </section>
  );
}
