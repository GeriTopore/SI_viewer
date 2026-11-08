import { useMemo, useState } from "react";
import { api, type DatasetSummary, type FitSource } from "../api/client";
import { allDatasets, useDatasets } from "../state/datasets";
import { cliffLorimer, datasetKey, useQuant, type SourceChoice } from "../state/quant";
import { shapeToRegion, useSi, useSiState } from "../state/si";
import { lineLabel, useElementTable } from "../views/si/xray";
import { filterLabel, formatValue, hasTechnique, isLineScan } from "../format";
import { downloadCsv, safeFileName } from "../download";
import { FitMapsSection } from "./FitMapsSection";

/** The EDS spectrum image or spectrum shown in the viewer, if any, and
 * whether what is shown is a line scan (which can't be fitted). */
function useEdsTarget(): { ds?: DatasetSummary; lineScan: boolean; other?: DatasetSummary } {
  const files = useDatasets((s) => s.files);
  const selection = useDatasets((s) => s.selection);
  return useMemo(() => {
    const byId = new Map(allDatasets(files).map((d) => [d.id, d]));
    const shown = selection.map((id) => byId.get(id));
    const lineScan = shown.some((d) => d && isLineScan(d));
    // A shown spectrum or SI that isn't EDS (to say why there is nothing to fit).
    const other = shown.find((d) => d && !d.eds && hasTechnique(d));
    return {
      other,
      ds: shown.find((d) => d && d.eds && !isLineScan(d) && (d.kind === "spectrum_image" || d.kind === "spectrum")),
      lineScan,
    };
  }, [files, selection]);
}

const BACKGROUND_ORDERS = [2, 3, 4, 5, 6, 7, 8, 10];

/** Quantify tab: fit one spectrum with exSpy's model and quantify it
 * (Cliff-Lorimer with editable k-factors). */
export function QuantifyPanel() {
  const { ds, lineScan, other } = useEdsTarget();
  if (!ds && other?.technique === "EELS") {
    return <p className="muted">EDS fitting doesn't apply to EELS data. EELS tools come later.</p>;
  }
  if (!ds && lineScan) {
    return <p className="muted">Fitting and quantification need a 2D spectrum image or a spectrum; line scans can't be quantified yet.</p>;
  }
  if (!ds && other && !other.technique) {
    return <p className="muted">Set the signal type to EDS in the Info tab to fit and quantify this data.</p>;
  }
  if (!ds) {
    return <p className="muted">Show an EDS spectrum image or spectrum in the viewer to fit and quantify it.</p>;
  }
  return <QuantifyTarget key={datasetKey(ds)} ds={ds} />;
}

function QuantifyTarget({ ds }: { ds: DatasetSummary }) {
  const key = datasetKey(ds);
  const q = useQuant();
  const si = useSiState(ds.id);
  const cursorSize = useSi((s) => s.cursorSize);
  const table = useElementTable();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isSi = ds.kind === "spectrum_image";
  const choice: SourceChoice = q.source[key] ?? "all";

  const sourceFor = (c: SourceChoice): { source: FitSource; label: string } | null => {
    if (!isSi || c === "all") return { source: { type: "all" }, label: isSi ? "All pixels" : ds.title };
    if (c === "cursor") {
      if (!si.cursor) return null;
      const { x, y } = si.cursor;
      return {
        source: { type: "pixel", x, y, size: cursorSize },
        label: `Cursor ${x}, ${y}${cursorSize > 1 ? ` (${cursorSize}×${cursorSize})` : ""}`,
      };
    }
    const roi = si.rois.find((r) => r.id === c);
    return roi ? { source: { type: "region", region: shapeToRegion(roi.shape) }, label: roi.name } : null;
  };

  const runFit = async () => {
    const src = sourceFor(choice);
    if (!src) {
      setError("That spectrum source no longer exists.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await api.fit(ds.id, src.source, q.options);
      q.setFit(key, { label: src.label, result });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const fit = q.fits[key];
  const atomicWeights = useMemo(() => Object.fromEntries(table.map((e) => [e.symbol, e.atomic_weight])), [table]);
  const composition = useMemo(() => {
    if (!fit) return [];
    const byElement = new Map<string, typeof fit.result.lines>();
    for (const l of fit.result.lines) byElement.set(l.element, [...(byElement.get(l.element) ?? []), l]);
    const rows = [...byElement.entries()].map(([element, lines]) => {
      const chosen = q.lineChoice[key]?.[element];
      const line = lines.find((l) => l.line === chosen) ?? lines.find((l) => l.analysis) ?? lines[0];
      return { element, line: line.line, intensity: line.intensity, included: !(q.excluded[key] ?? []).includes(element), lines };
    });
    return cliffLorimer(rows, q.kfactors, atomicWeights).map((r, i) => ({ ...r, lines: rows[i].lines }));
  }, [fit, q.lineChoice, q.excluded, q.kfactors, atomicWeights, key]);
  const uncalibrated = composition.some((r) => r.included && r.uncalibrated);

  // The quantification table, with what it was fitted from.
  const saveTable = () =>
    fit &&
    downloadCsv(
      [
        ["source", fit.label],
        ...(filterLabel(ds.filters) ? [["processing", filterLabel(ds.filters)]] : []),
        [],
        ["element", "line", "intensity", "k-factor", "k-factor set", "included", "wt%", "at%"],
        ...composition.map((r) => [
          r.element, r.line, r.intensity, r.k, r.uncalibrated ? "no (default 1)" : "yes", r.included ? "yes" : "no",
          r.wt, r.at,
        ]),
      ],
      `${safeFileName(ds.title)}_quantification.csv`,
    );

  return (
    <div className="quantify">
      <section>
        <h3 className="q-title">{ds.title}</h3>
        <p className="muted small q-elements">
          Elements: {ds.eds?.elements.length ? ds.eds.elements.join(", ") : "none"} — change them with
          the Elements button above the spectrum.
        </p>
        {filterLabel(ds.filters) && (
          <p className="muted small q-elements" title="Set in the Process tab">
            Fits use the processed data: {filterLabel(ds.filters)}.
          </p>
        )}
      </section>

      <section className="q-form">
        {isSi && (
          <label>
            Spectrum
            <select value={choice} onChange={(e) => q.setSource(key, e.target.value)} disabled={busy}>
              <option value="all">All pixels</option>
              <option value="cursor" disabled={!si.cursor}>
                Cursor{si.cursor ? ` (${si.cursor.x}, ${si.cursor.y})` : ""}
              </option>
              {si.rois.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name}
                </option>
              ))}
            </select>
          </label>
        )}
        <label title="Order of the polynomial fitted to the bremsstrahlung background">
          Background
          <select
            value={q.options.background_order}
            onChange={(e) => q.setOptions({ background_order: Number(e.target.value) })}
            disabled={busy}
          >
            {BACKGROUND_ORDERS.map((n) => (
              <option key={n} value={n}>
                polynomial, order {n}
              </option>
            ))}
          </select>
        </label>
        <label title="Channels below this energy (the noise peak) are not fitted">
          Ignore below
          <span className="q-inline">
            <input
              type="number"
              min={0}
              step={0.1}
              value={q.options.min_energy}
              onChange={(e) => q.setOptions({ min_energy: Math.max(0, Number(e.target.value) || 0) })}
              disabled={busy}
            />
            keV
          </span>
        </label>
        <label className="q-check">
          <input
            type="checkbox"
            checked={q.options.calibrate_resolution}
            onChange={(e) => q.setOptions({ calibrate_resolution: e.target.checked })}
            disabled={busy}
          />
          Calibrate detector resolution
        </label>
        <label className="q-check">
          <input
            type="checkbox"
            checked={q.options.calibrate_energy}
            onChange={(e) => q.setOptions({ calibrate_energy: e.target.checked })}
            disabled={busy}
          />
          Calibrate line energies
        </label>
        <button className="primary" onClick={runFit} disabled={busy || !ds.eds?.elements.length}>
          {busy ? "Fitting…" : fit ? "Refit" : "Fit spectrum"}
        </button>
        {error && <p className="field-error">{error}</p>}
      </section>

      {fit && (
        <>
          <section className="q-summary small">
            <div>
              <span className="muted">Fitted:</span> {fit.label}
              {fit.result.pixels > 1 && <span className="muted"> · {fit.result.pixels.toLocaleString()} px</span>}
            </div>
            <div>
              {fit.result.resolution_MnKa !== null && (
                <>
                  <span className="muted">Resolution (Mn Kα):</span> {fit.result.resolution_MnKa.toFixed(1)} eV ·{" "}
                </>
              )}
              {fit.result.residual_ratio !== null && (
                <>
                  <span className="muted">Residual:</span> {(100 * fit.result.residual_ratio).toFixed(1)}%
                </>
              )}
            </div>
            {fit.result.skipped.length > 0 && (
              <div className="muted">
                Not fitted (no line above {q.options.min_energy} keV): {fit.result.skipped.join(", ")}
              </div>
            )}
            <div className="q-toggles">
              <label>
                <input type="checkbox" checked={q.showFit} onChange={(e) => q.setShowFit(e.target.checked)} />
                Show fit
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={q.showComponents}
                  onChange={(e) => q.setShowComponents(e.target.checked)}
                />
                Show elements
              </label>
            </div>
          </section>

          <section>
            <table className="q-table">
              <thead>
                <tr>
                  <th title="Include in the composition">✓</th>
                  <th>Line</th>
                  <th>Intensity</th>
                  <th>k-factor</th>
                  <th>wt%</th>
                  <th>at%</th>
                </tr>
              </thead>
              <tbody>
                {composition.map((r) => (
                  <tr key={r.element} className={r.included ? undefined : "q-excluded"}>
                    <td>
                      <input
                        type="checkbox"
                        checked={r.included}
                        onChange={() => q.toggleExcluded(key, r.element)}
                        aria-label={`Include ${r.element}`}
                      />
                    </td>
                    <td>
                      <select value={r.line} onChange={(e) => q.setLine(key, r.element, e.target.value)}>
                        {r.lines.map((l) => (
                          <option key={l.line} value={l.line}>
                            {r.element} {lineLabel(l.line.split("_")[1])}
                          </option>
                        ))}
                      </select>
                    </td>
                    <td className="num">{formatValue(Math.round(r.intensity))}</td>
                    <td>
                      <input
                        type="number"
                        min={0}
                        step={0.01}
                        value={q.kfactors[r.line] ?? ""}
                        placeholder="1"
                        className={r.uncalibrated ? "q-k-default" : undefined}
                        onChange={(e) => q.setKFactor(r.line, e.target.value === "" ? null : Number(e.target.value))}
                        aria-label={`k-factor for ${r.line}`}
                      />
                    </td>
                    <td className="num">{r.wt === null ? "–" : r.wt.toFixed(1)}</td>
                    <td className="num">{r.at === null ? "–" : r.at.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {uncalibrated && (
              <p className="q-warning small">
                Uncalibrated: some k-factors are the default 1, so these are intensity ratios rather than a
                composition. Enter k-factors for your microscope (or load a preset).
              </p>
            )}
            <p className="muted small">
              Cliff-Lorimer, normalised over the ticked elements; no absorption correction.{" "}
              <button className="link-button" onClick={saveTable} title="Save this table as CSV">
                Save CSV
              </button>
            </p>
          </section>

          <KFactorPresets />
        </>
      )}

      {isSi && <FitMapsSection ds={ds} />}
    </div>
  );
}

/** Save, load and delete named k-factor sets (stored in this browser). */
function KFactorPresets() {
  const { presets, applyPreset, savePreset, deletePreset } = useQuant();
  const [name, setName] = useState("");
  const [current, setCurrent] = useState("");
  const names = Object.keys(presets).sort();
  return (
    <section className="q-presets">
      <h3>k-factor presets</h3>
      <div className="q-preset-row">
        <select
          value={names.includes(current) ? current : ""}
          onChange={(e) => {
            setCurrent(e.target.value);
            if (e.target.value) applyPreset(e.target.value);
          }}
        >
          <option value="">{names.length ? "Load preset…" : "No presets yet"}</option>
          {names.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
        {names.includes(current) && (
          <button className="link-button" onClick={() => deletePreset(current)} title={`Delete preset "${current}"`}>
            Delete
          </button>
        )}
      </div>
      <div className="q-preset-row">
        <input
          type="text"
          placeholder="Name, e.g. Spectra 300 SuperX"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <button
          className="secondary small-button"
          disabled={!name.trim()}
          onClick={() => {
            savePreset(name.trim());
            setCurrent(name.trim());
            setName("");
          }}
        >
          Save current
        </button>
      </div>
    </section>
  );
}
