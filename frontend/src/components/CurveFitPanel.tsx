import { Fragment, useEffect, useState, type KeyboardEvent } from "react";
import type { DatasetSummary } from "../api/client";
import { energyRange } from "../state/background";
import { snapToAxis } from "../state/pca";
import { EnergyInput } from "./BackgroundSection";
import { formatValue, isLineScan } from "../format";
import {
  PLACED, addComponent, drawModel, duplicateComponent, estimateComponent, fitModel, keyOf, modeOf, paramRole,
  removeComponent, renameComponent, setAllFree, setParamLimits, setParamMode, setTwin, updateComponent, useCurveFit,
  useCurveFitState, type ModelComponent, type ParamMode,
} from "../state/curveFit";
import { useSiState } from "../state/si";
import { useCurveSource } from "../views/useCurveFitSeries";

/** A number typed in a table cell: committed on Enter or leaving it; empty
 * gives null where ``optional``. */
function NumberCell({ value, onCommit, label, optional, disabled }: {
  value: number | null; onCommit: (v: number | null) => void; label: string; optional?: boolean; disabled?: boolean;
}) {
  const shown = value === null ? "" : formatValue(value);
  const [text, setText] = useState(shown);
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    if (!editing) setText(shown);
  }, [shown, editing]);
  const commit = () => {
    setEditing(false);
    const t = text.trim();
    if (t === "" && optional) return value !== null && onCommit(null);
    const v = Number(t);
    if (t === "" || !Number.isFinite(v)) return setText(shown);
    if (v !== value) onCommit(v);
  };
  return (
    <input
      className="cell-input"
      value={text}
      inputMode="decimal"
      aria-label={label}
      placeholder={optional ? "–" : undefined}
      disabled={disabled}
      onFocus={() => setEditing(true)}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e: KeyboardEvent<HTMLInputElement>) => {
        e.stopPropagation();
        if (e.key === "Enter") e.currentTarget.blur();
        else if (e.key === "Escape") {
          setText(shown);
          setEditing(false);
        }
      }}
    />
  );
}

/** A text typed in a table cell (a twin's formula): committed on Enter or
 * leaving it; Escape reverts. */
function TextCell({ value, onCommit, label, title }: { value: string; onCommit: (v: string) => void; label: string; title?: string }) {
  const [text, setText] = useState(value);
  useEffect(() => setText(value), [value]);
  return (
    <input
      className="cell-input curve-twin-expr"
      value={text}
      aria-label={label}
      title={title}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => text.trim() !== value && onCommit(text.trim())}
      onKeyDown={(e) => {
        e.stopPropagation();
        if (e.key === "Enter") e.currentTarget.blur();
        else if (e.key === "Escape") setText(value);
      }}
    />
  );
}

/** A component's name; a double-click turns it into a field (Enter or
 * leaving it renames, Escape cancels). */
function ComponentName({ ds, c }: { ds: DatasetSummary; c: ModelComponent }) {
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(c.name);
  const done = (save: boolean) => {
    setEditing(false);
    if (save) renameComponent(ds, c.name, text);
  };
  if (editing)
    return (
      <input
        className="rename-input"
        value={text}
        autoFocus
        onFocus={(e) => e.currentTarget.select()}
        aria-label="Component name"
        onClick={(e) => e.stopPropagation()}
        onChange={(e) => setText(e.target.value)}
        onBlur={() => done(true)}
        onKeyDown={(e) => {
          e.stopPropagation();
          if (e.key === "Enter") done(true);
          else if (e.key === "Escape") done(false);
        }}
      />
    );
  return (
    <span
      className="editable-title"
      onDoubleClick={(e) => {
        e.stopPropagation();
        setText(c.name);
        setEditing(true);
      }}
    >
      {c.name}
    </span>
  );
}

/** Process tab, any spectrum or spectrum image: a model of HyperSpy's
 * components fitted to one spectrum, over its whole range (crop the data to
 * fit part of it) (M21a, M21b). While this is open the model and its
 * components are drawn on the spectrum with their current values, as they
 * are edited. A component is
 * selected here or by clicking its curve (its table unfolds, its handles show
 * on the plot); peaks are placed with a click on the spectrum. */
export function CurveFitPanel({ ds }: { ds: DatasetSummary }) {
  const st = useCurveFitState(ds);
  const catalogue = useCurveFit((s) => s.catalogue);
  const si = useSiState(ds.id);
  const source = useCurveSource(ds, st.source);
  const isSi = ds.kind === "spectrum_image";
  const lineScan = isLineScan(ds);
  const { patch } = useCurveFit.getState();

  useEffect(() => {
    void useCurveFit.getState().loadCatalogue().catch(() => {});
    useCurveFit.getState().setShown(ds.id);
    return () => {
      if (useCurveFit.getState().shown === ds.id) useCurveFit.getState().setShown(null);
    };
  }, [ds.id]);

  // The model drawn again (with its current values) after an edit.
  const key = keyOf(st);
  const sourceKey = JSON.stringify(source);
  useEffect(() => {
    if (!source || !st.components.length || st.busy || st.result?.key === key) return;
    const t = setTimeout(() => void drawModel(ds, source), 250);
    return () => clearTimeout(t);
  }, [key, sourceKey, st.busy]); // eslint-disable-line react-hooks/exhaustive-deps

  const setParam = (c: ModelComponent, name: string, p: Partial<ModelComponent["params"][string]>) =>
    updateComponent(ds, c.name, { params: { ...c.params, [name]: { ...c.params[name], ...p } } });
  const kindOf = (kind: string) => catalogue?.components.find((k) => k.kind === kind);
  const fitted = st.result?.fitted && st.result.key === key ? st.result : null;
  const stdOf = (c: ModelComponent, name: string) =>
    fitted?.components.find((f) => f.name === c.name)?.params.find((p) => p.name === name)?.std ?? null;
  const groups = catalogue ? [...new Set(catalogue.components.map((c) => c.group))] : [];

  return (
    <div className="curve-fit">

      <section className="q-form">
        {isSi && (
          <label title="The spectrum the model is fitted to">
            <span>Spectrum</span>
            <select value={st.source} onChange={(e) => patch(ds, { source: e.target.value })} disabled={st.busy}>
              <option value="all">{lineScan ? "All positions" : "All pixels"}</option>
              <option value="cursor" disabled={!si.cursor}>
                Cursor{si.cursor ? ` (${lineScan ? si.cursor.x : `${si.cursor.x}, ${si.cursor.y}`})` : ""}
              </option>
              {si.rois.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name}
                </option>
              ))}
            </select>
          </label>
        )}
        <label
          className="q-check"
          title="Fit the background components (PowerLaw, Exponential, Polynomial, Offset) over a window alone, e.g. the pre-edge, then keep them fixed while the rest are fitted"
        >
          <input
            type="checkbox"
            checked={st.bgLo !== null && st.bgHi !== null}
            onChange={(e) => {
              if (!e.target.checked) return patch(ds, { bgLo: null, bgHi: null });
              const r = energyRange(ds);
              if (r) patch(ds, { bgLo: r.e0, bgHi: snapToAxis(ds, r.e0 + (r.e1 - r.e0) * 0.15) });
            }}
          />
          Background window
        </label>
        {st.bgLo !== null && st.bgHi !== null && (
          <label title="The window the background is fitted over (drag the hatched band on the spectrum)">
            <span>
              <i className="bg-swatch" style={{ background: CURVE_BG_COLOUR }} />
              Window
            </span>
            <span className="q-inline bg-range">
              <EnergyInput value={st.bgLo} step={energyRange(ds)?.step ?? 1} label="Background from" onCommit={(v) => patch(ds, { bgLo: snapToAxis(ds, v) })} />
              –
              <EnergyInput value={st.bgHi} step={energyRange(ds)?.step ?? 1} label="Background to" onCommit={(v) => patch(ds, { bgHi: snapToAxis(ds, v) })} />
              {energyRange(ds)?.units}
            </span>
          </label>
        )}
        <label title="Add one of HyperSpy's components (peaks are placed with a click; the others start from their estimate)">
          <span>Add</span>
          <select
            value=""
            disabled={!catalogue || st.busy}
            onChange={(e) => {
              const kind = e.target.value;
              if (!kind) return;
              // Peaks are placed with a click on the spectrum; the others start from their estimate.
              if (PLACED.has(kind) && source) patch(ds, { placing: kind, selected: null });
              else void addComponent(ds, kind, source);
            }}
          >
            <option value="">Component…</option>
            {groups.map((g) => (
              <optgroup key={g} label={g}>
                {catalogue!.components
                  .filter((c) => c.group === g)
                  .map((c) => (
                    <option key={c.kind} value={c.kind} title={c.summary}>
                      {c.kind}
                    </option>
                  ))}
              </optgroup>
            ))}
          </select>
        </label>
        {st.placing && (
          <p className="muted small curve-placing">
            Click on the spectrum where the {st.placing}'s top should be ·{" "}
            <button className="link-button" onClick={() => patch(ds, { placing: null })}>
              Cancel
            </button>
          </p>
        )}
      </section>

      {st.components.map((c) => (
        <section
          key={c.name}
          className={`curve-component${c.active ? "" : " off"}${c.name === st.selected ? " selected" : ""}`}
        >
          <div
            className="section-header curve-component-header"
            onClick={() => patch(ds, { selected: c.name === st.selected ? null : c.name })}
            title="Click to select (its handles show on the spectrum); double-click the name to rename"
          >
            <span className="curve-component-title">
              <input
                type="checkbox"
                checked={c.active}
                onClick={(e) => e.stopPropagation()}
                onChange={(e) => updateComponent(ds, c.name, { active: e.target.checked })}
                aria-label={`Use ${c.name}`}
              />
              <span className="series-swatch" style={{ background: c.color }} />
              <ComponentName ds={ds} c={c} />
              {c.name !== c.kind && !c.name.startsWith(`${c.kind} `) && <span className="muted small">{c.kind}</span>}
            </span>
            <span className="curve-component-actions" onClick={(e) => e.stopPropagation()}>
              {c.kind === "Polynomial" && (
                <select
                  value={c.order ?? 2}
                  aria-label="Order"
                  title="Order of the polynomial"
                  onChange={(e) => {
                    const order = Number(e.target.value);
                    const params = Object.fromEntries(
                      Array.from({ length: order + 1 }, (_, i) => [`a${i}`, c.params[`a${i}`] ?? { value: 0, free: true, bmin: null, bmax: null }]),
                    );
                    updateComponent(ds, c.name, { order, params });
                  }}
                >
                  {Array.from({ length: (catalogue?.max_order ?? 10) + 1 }, (_, i) => (
                    <option key={i} value={i}>
                      order {i}
                    </option>
                  ))}
                </select>
              )}
              <button
                className="icon-button"
                title={`Duplicate ${c.name}`}
                aria-label={`Duplicate ${c.name}`}
                onClick={() => duplicateComponent(ds, c.name, source)}
              >
                ⧉
              </button>
              <button className="icon-button" title={`Remove ${c.name}`} onClick={() => removeComponent(ds, c.name)}>
                ×
              </button>
            </span>
          </div>
          {c.name === st.selected && (
            <div className="curve-component-tools">
              {kindOf(c.kind)?.estimate && (
                <button
                  className="link-button"
                  disabled={!source || st.busy}
                  onClick={() => source && void estimateComponent(ds, c.name, source)}
                  title="Starting values from the spectrum"
                >
                  Estimate
                </button>
              )}
              <button className="link-button" onClick={() => setAllFree(ds, c.name, false)} title="Fix every parameter">
                Fix all
              </button>
              <button className="link-button" onClick={() => setAllFree(ds, c.name, true)} title="Fit every parameter">
                Free all
              </button>
            </div>
          )}
          {c.name === st.selected && (
          <table className="axes curve-params">
            <thead>
              <tr>
                <th>Name</th>
                <th>Value</th>
                <th>±</th>
                <th title="Free: fitted; Limit: fitted within Min–Max (shown on the spectrum when the row is selected); Fixed: kept as it is">Fit</th>
                <th>Min</th>
                <th>Max</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(c.params).map(([name, p]) => {
                const std = stdOf(c, name);
                const mode = modeOf(p);
                const shown = paramRole(c.kind, name) !== "other";
                // A twin's value is its target's through the formula: as last drawn.
                const drawn = st.result?.components.find((f) => f.name === c.name)?.params.find((q) => q.name === name)?.value;
                const others = st.components.filter((x) => x.name !== c.name);
                return (
                  <Fragment key={name}>
                  <tr
                    className={st.selectedParam === name ? "selected" : undefined}
                    onClick={() => patch(ds, { selectedParam: name })}
                    title={shown ? "Click to show its limits on the spectrum" : undefined}
                  >
                    <td>{name}</td>
                    <td>
                      <NumberCell
                        value={mode === "twin" && drawn !== undefined ? drawn : p.value}
                        disabled={mode === "twin"}
                        label={`${c.name} ${name}`}
                        onCommit={(v) => setParam(c, name, { value: v ?? 0 })}
                      />
                    </td>
                    <td className="muted">{std !== null && p.free && mode !== "twin" ? formatValue(std) : ""}</td>
                    <td>
                      <select
                        className="cell-input"
                        value={mode}
                        aria-label={`${name}: free, limited or fixed`}
                        onChange={(e) => setParamMode(ds, c.name, name, e.target.value as ParamMode)}
                      >
                        <option value="free">Free</option>
                        <option value="limited">Limit</option>
                        <option value="fixed">Fixed</option>
                        <option value="twin" disabled={!others.length}>{others.length ? "Twin" : "Twin (add a component)"}</option>
                      </select>
                    </td>
                    <td>
                      <NumberCell
                        value={p.bmin}
                        optional
                        disabled={mode !== "limited"}
                        label={`${name} minimum`}
                        onCommit={(v) => setParamLimits(ds, c.name, name, v, p.bmax)}
                      />
                    </td>
                    <td>
                      <NumberCell
                        value={p.bmax}
                        optional
                        disabled={mode !== "limited"}
                        label={`${name} maximum`}
                        onCommit={(v) => setParamLimits(ds, c.name, name, p.bmin, v)}
                      />
                    </td>
                  </tr>
                  {p.twin && (
                    <tr className="curve-twin-row">
                      <td colSpan={6}>
                        <span className="curve-twin">
                          <span className="muted">=</span>
                          <TextCell
                            value={p.twin.expr}
                            label={`${name}: formula of x`}
                            title="A formula of x, the value it follows, e.g. x + 5.4 or 0.5*x (x alone: the same value)"
                            onCommit={(expr) => setTwin(ds, c.name, name, { expr: expr || "x" })}
                          />
                          <span className="muted" title="x is the value of">x:</span>
                          <select
                            className="cell-input"
                            value={`${p.twin.component}|${p.twin.param}`}
                            aria-label={`${name} follows`}
                            onChange={(e) => {
                              const [component, param] = e.target.value.split("|");
                              setTwin(ds, c.name, name, { component, param });
                            }}
                          >
                            {others.map((o) => (
                              <optgroup key={o.name} label={o.name}>
                                {Object.keys(o.params).map((q) => (
                                  <option key={q} value={`${o.name}|${q}`}>
                                    {o.name} {q}
                                  </option>
                                ))}
                              </optgroup>
                            ))}
                          </select>
                        </span>
                      </td>
                    </tr>
                  )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
          )}
        </section>
      ))}

      <section className="q-form">
        {st.error && <p className="field-error">{st.error}</p>}
        <button
          className="primary"
          disabled={!source || !st.components.length || st.busy}
          onClick={() => source && void fitModel(ds, source)}
          title="Fit the free parameters (HyperSpy's default optimizer; bounded where a limit is set)"
        >
          {st.busy ? "Fitting…" : "Fit"}
        </button>
        {fitted && (
          <p className="muted small curve-fit-goodness">
            {fitted.success === false && <span className="field-warning">Did not converge · </span>}
            reduced χ² {fitted.red_chisq !== null && fitted.red_chisq !== undefined ? formatValue(fitted.red_chisq) : "–"}
            {fitted.poisson ? "" : " (no noise estimate)"} · χ² {fitted.chisq != null ? formatValue(fitted.chisq) : "–"} ·{" "}
            {fitted.dof} free parameters
            {fitted.residual_ratio !== null && ` · residual ${formatValue(fitted.residual_ratio * 100)} % of the signal's spread`}
          </p>
        )}
      </section>
    </div>
  );
}


export const CURVE_BG_COLOUR = "#f59e0b";
