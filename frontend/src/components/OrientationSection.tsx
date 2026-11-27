import { useState } from "react";
import { api, type DatasetSummary } from "../api/client";
import { formatValue } from "../format";
import { allDatasets, useDatasets } from "../state/datasets";
import { useLayers } from "../state/layers";
import { PHASE_COLOURS, chosenFor, isPolar, matchKey, othersFor, perAngstrom, phasesFor, polarFor, polarStale, polarsFor, templatesOf, useOrientation } from "../state/orientation";
import { useFourDState } from "../views/FourDView";
import { done } from "./FourDCalibrationInfo";
import { ContextMenu, type MenuItem } from "./ContextMenu";
import { templatesLabel } from "./TemplatesView";
import { JobProgress } from "./StrainSection";

/** Orientation and phase mapping as pyxem does it (its SPED-Ag tutorial):
 * the Templates object (made on a Structure) drawn on the patterns and
 * matched against. */
export function OrientationSection({ d }: { d: DatasetSummary }) {
  const files = useDatasets((s) => s.files);
  const all = allDatasets(files);
  const sets = templatesOf(all);
  const chosenId = useOrientation((s) => chosenFor(s, d));
  const polar = isPolar(d);
  const chosen = sets.find((x) => x.id === chosenId) ?? null;
  const { choose } = useOrientation.getState();
  const calibrated = perAngstrom(d) !== null;
  const structureOf = (x: DatasetSummary) => all.find((s) => s.id === x.parent_id);
  const polars = polar ? [] : polarsFor(all, d);
  const matchedOn = useOrientation((s) => polarFor(s, all, d));
  const stale = matchedOn ? polarStale(matchedOn, all) : false;
  // The cursors (shared by the data and its polar patterns): matched where they are.
  const { st } = useFourDState(d);
  // The phases matched: the Templates chosen, and those of other Structures ticked (a phase map).
  const phases = useOrientation((s) => phasesFor(s, d, all));
  const others = useOrientation((s) => othersFor(s, d));
  const nameOf = (x: DatasetSummary) => `${structureOf(x)?.title ?? x.templates!.name} · ${x.title.replace(/^Templates · /, "")} (${x.templates!.templates})`;
  // The list: the Templates chosen (drawn), then the other phases' (matched with it).
  const listed = [chosen, ...others.filter((id) => id !== chosen?.id).map((id) => sets.find((x) => x.id === id))]
    .filter((x): x is DatasetSummary => !!x);
  const setList = (ids: string[]) => {
    choose(d.id, ids[0] ?? null);
    useOrientation.getState().setOthers(d.id, ids.slice(1));
  };
  const [addMenu, setAddMenu] = useState<{ x: number; y: number } | null>(null);
  const addItems: MenuItem[] = sets.map((x) => {
    const sameStructure = listed.find((y) => y.parent_id === x.parent_id);
    return {
      label: nameOf(x),
      title: sameStructure ? `${structureOf(x)?.title ?? "Its Structure"} is listed already` : templatesLabel(x),
      disabled: !!sameStructure,
      onSelect: () => setList([...listed.map((y) => y.id), x.id]),
    };
  });
  const key = phases && matchedOn ? matchKey(matchedOn, phases) : "";
  const busy = useOrientation((s) => (key ? s.matchBusy[key] ?? false : false));
  const matchNow = () => phases && matchedOn && useOrientation.getState().runMatch(
    key, d.id, phases, polar ? null : matchedOn.id, st.cursors.map((c) => [c.x, c.y]),
  );

  return (
    <section className="q-form virtual-form">
      <h4 className="process-heading">Templates{done(!!chosen)}</h4>
      {/* The phases matched, as the detector groups: the highlighted one drawn on the
          patterns (and in the Structure tab), the others matched with it (a phase map). */}
      {listed.length > 0 && (
        <ul className="group-list">
          {listed.map((x, i) => (
            <li key={x.id} className={`layer-row no-eye${i === 0 ? " active" : ""}`}
              title={`${templatesLabel(x)}${i === 0 ? "" : " · click to draw it on the patterns"}`}
              onClick={() => i > 0 && setList([x.id, ...listed.filter((y) => y.id !== x.id).map((y) => y.id)])}>
              <span className="layer-swatch" style={{ background: PHASE_COLOURS[i % PHASE_COLOURS.length] }} />
              <span className="layer-label">{nameOf(x)}</span>
              <button className="icon-button" title={`Remove ${nameOf(x)}`}
                onClick={(e) => {
                  e.stopPropagation();
                  setList(listed.filter((y) => y.id !== x.id).map((y) => y.id));
                }}>
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
      {sets.length ? (
        <div className="virtual-row">
          <button className="secondary" aria-haspopup="menu"
            onMouseDown={(e) => addMenu && e.stopPropagation()}
            onClick={(e) => {
              if (addMenu) return setAddMenu(null);
              const r = e.currentTarget.getBoundingClientRect();
              setAddMenu({ x: r.left, y: r.bottom + 4 });
            }}>
            + Template ▾
          </button>
          {addMenu && <ContextMenu x={addMenu.x} y={addMenu.y} onClose={() => setAddMenu(null)} items={addItems} />}
        </div>
      ) : (
        <p className="muted small">Make templates on a Structure first (open a .cif, then its Templates process).</p>
      )}
      {chosen && !calibrated && !polar && <p className="field-error">Calibrate the pattern's pixel size to draw them.</p>}
      {/* Not on polar patterns: their axes are radius and angle. */}
      {chosen && calibrated && !polar && <TemplateScale d={d} />}
      {chosen && !polar && (
        polars.length ? (
          <label title="The polar patterns matched against">
            <span>Polar patterns</span>
            <select value={matchedOn?.id ?? ""} onChange={(e) => useOrientation.getState().choosePolar(d.id, e.target.value)}>
              {polars.map((x) => <option key={x.id} value={x.id}>{x.title}</option>)}
            </select>
          </label>
        ) : (
          <p className="muted small">Make polar patterns first (Polar transform).</p>
        )
      )}
      {chosen && stale && (
        <p className="field-error">Made at another pixel size than {all.find((x) => x.id === matchedOn!.polar!.source)?.title}'s now: transform it again.</p>
      )}
      {chosen && matchedOn && (
        <div className="process-actions">
          <button className="primary" disabled={busy} onClick={() => void matchNow()}
            title="Match the templates at each cursor">
            Match current position
          </button>
        </div>
      )}
      {phases && matchedOn && <MapScan d={d} templates={phases} polar={polar ? null : matchedOn.id} />}
    </section>
  );
}

// The slider's range, logarithmic (1× in the middle): ¼× to 4×. The box takes any factor.
const SCALE_LOG2 = 2;

/** The template drawn larger or smaller (about the axes' 0) to line it up
 * with the pattern by eye: the pattern's pixel size that implies, and kept
 * as its calibration on request (scale and offset, the centre kept). */
function TemplateScale({ d }: { d: DatasetSummary }) {
  const f = useOrientation((s) => s.scale[d.id] ?? 1);
  const setScale = (v: number) => useOrientation.getState().setScale(d.id, v);
  const updateDataset = useDatasets((s) => s.updateDataset);
  const [error, setError] = useState<string | null>(null);
  const k = d.axes.map((a, index) => ({ ...a, index })).filter((a) => !a.navigate);
  const units = k[0]?.units ?? "";
  // Spots drawn f× further out: each pattern pixel is 1/f as large.
  const implied = k.length ? k[0].scale / f : 0;
  const apply = async () => {
    setError(null);
    try {
      updateDataset(await api.calibrate(d.id, k.map((a) => ({ index: a.index, scale: a.scale / f, offset: a.offset / f }))));
      setScale(1);
      useLayers.getState().notify(`Pattern pixel size set to ${formatValue(implied)} ${units}.`);
    } catch (e) {
      setError((e as Error).message);
    }
  };
  return (
    <>
      <label title="Draw the template larger or smaller to fit the pattern">
        <span>Scale</span>
        <span className="size-control">
          <input type="range" min={-SCALE_LOG2} max={SCALE_LOG2} step={0.0005} value={Math.log2(f)}
            onChange={(e) => setScale(+(2 ** Number(e.target.value)).toPrecision(4))} />
          <input key={f} type="number" className="cell-input" step="any" defaultValue={formatValue(f)}
            onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
            onBlur={(e) => {
              const v = Number(e.target.value);
              if (Number.isFinite(v) && v > 0 && Math.abs(v - f) > 1e-9) setScale(v);
            }} />
          <span className="muted small">×</span>
        </span>
      </label>
      <p className="muted small">
        Pixel size {formatValue(implied)} {units}
        {f !== 1 && <span> (now {formatValue(k[0].scale)})</span>}
      </p>
      {error && <p className="field-error">{error}</p>}
      <div className="process-actions">
        <button className="secondary" disabled={f === 1} onClick={() => setScale(1)} title="Back to the pattern's calibrated pixel size">
          Reset
        </button>
        <button className="primary" disabled={f === 1} onClick={() => void apply()}
          title="Set the pattern's pixel size to the template's">
          Use as pixel size
        </button>
      </div>
    </>
  );
}

/** The whole scan matched as the tutorial does (the best match kept, the
 * full match on a fraction of the templates): an Orientation map under the
 * data, with its correlation and IPF-x, y, z images. */
function MapScan({ d, templates, polar }: { d: DatasetSummary; templates: string; polar: string | null }) {
  const job = useOrientation((s) => s.mapJobs[d.id]);
  const error = useOrientation((s) => s.mapErrors[d.id] ?? null);
  const frac = useOrientation((s) => s.fracKeep[d.id] ?? 0.5);
  const running = !!job && (job.status === "queued" || job.status === "running");
  return (
    <>
      <h4 className="process-heading">Orientation map</h4>
      <label title="Fraction of the templates fully matched">
        <span>Fraction kept</span>
        <input type="number" min={0.01} max={1} step={0.05} value={frac}
          onChange={(e) => {
            const v = Number(e.target.value);
            if (e.target.value !== "" && v > 0 && v <= 1) useOrientation.setState((s) => ({ fracKeep: { ...s.fracKeep, [d.id]: v } }));
          }} />
      </label>
      {error && <p className="field-error">{error}</p>}
      {running ? (
        <JobProgress job={job!} cancel={() => void useOrientation.getState().cancelMap(d.id)} />
      ) : (
        <div className="process-actions">
          <button className="primary" onClick={() => void useOrientation.getState().runMap(d.id, templates, polar)}
            title="Match every pattern: an Orientation map with IPF maps">
            Map
          </button>
        </div>
      )}
      {job?.status === "failed" && !running && <p className="field-error">{job.error}</p>}
    </>
  );
}
