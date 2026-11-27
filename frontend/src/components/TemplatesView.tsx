import { useEffect, useState } from "react";
import { api, type DatasetSummary, type TemplatePattern, type TemplatesIpf } from "../api/client";
import { useOrientation } from "../state/orientation";
import { useViewTemplate } from "../state/viewTemplate";
import { formatValue } from "../format";
import { TemplatePatternPlot, TemplatesIpfPlot } from "./OrixPlots";

const uvwLabel = (uvw: number[]) => `[${uvw.map((x) => formatValue(x)).join(" ")}]`;

/** A Templates object's settings in a line. */
export const templatesLabel = (d: DatasetSummary) => {
  const o = d.templates!.options;
  return `${d.templates!.templates} templates · ${formatValue(o.resolution)}° · ${formatValue(o.reciprocal_radius)} 1/Å · s ${formatValue(o.max_excitation_error)} 1/Å · ${formatValue(o.precession_angle)}° precession · ${formatValue(o.kv)} kV`;
};

/** A Structure's templates (the bottom pane's Templates tab): their beam
 * directions in the inverse pole figure (orix, as pyxem's IPF markers); the
 * one clicked (or nearest a zone axis) as simulated, and drawn on 4D-STEM
 * patterns while Orientation mapping is shown. */
export function TemplatesPanel({ sets, chosen, onChoose }: { sets: DatasetSummary[]; chosen: string; onChoose: (id: string) => void }) {
  const d = sets.find((x) => x.id === chosen) ?? sets[0];
  const linked = useOrientation((s) => s.linked[d.id] ?? false);
  const view = useViewTemplate(d.id, d.parent_id ?? null, linked);
  const clicked = useOrientation((s) => s.picked[d.id] ?? null);
  const picked = linked ? (view?.index ?? null) : clicked;
  const pick = (i: number) => useOrientation.getState().pick(d.id, i);
  const [ipf, setIpf] = useState<{ id: string; ipf: TemplatesIpf } | null>(null);
  const [t, setT] = useState<TemplatePattern | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uvw, setUvw] = useState<[number, number, number]>([1, 0, 1]);

  useEffect(() => {
    let stale = false;
    setError(null);
    api.templatesIpf(d.id).then((r) => !stale && setIpf({ id: d.id, ipf: r }), (e: Error) => !stale && setError(e.message));
    return () => {
      stale = true;
    };
  }, [d.id]);
  useEffect(() => {
    if (linked || picked === null) return setT(null);
    let stale = false;
    api.template(d.id, picked).then((r) => !stale && setT(r), (e: Error) => !stale && setError(e.message));
    return () => {
      stale = true;
    };
  }, [d.id, picked, linked]);
  const find = () => api.nearestTemplate(d.id, uvw).then((r) => pick(r.index), (e: Error) => setError(e.message));

  return (
    <div className="templates-tab">
      <div className="templates-bar small">
        {sets.length > 1 ? (
          <select value={d.id} onChange={(e) => onChoose(e.target.value)}>
            {sets.map((x) => <option key={x.id} value={x.id}>{x.title}</option>)}
          </select>
        ) : (
          <span>{d.title}</span>
        )}
        <span className="muted">{templatesLabel(d)}</span>
        <label className="q-check-line" title="The template nearest the way the Structure is seen (its 3D view, or a pattern's Structure tab), turned as the crystal is">
          <input type="checkbox" checked={linked} onChange={(e) => useOrientation.getState().setLinked(d.id, e.target.checked)} />
          Link to structure view
        </label>
        <span className="templates-find" hidden={linked}>
          <span className="uvw-input">
            {uvw.map((v, i) => (
              <input key={i} type="number" step={1} value={v} aria-label={"uvw"[i]}
                onChange={(e) => {
                  if (e.target.value === "") return;
                  const next = [...uvw] as [number, number, number];
                  next[i] = Math.round(Number(e.target.value));
                  setUvw(next);
                }} />
            ))}
          </span>
          <button className="secondary" disabled={uvw.every((x) => x === 0)} onClick={() => void find()}
            title="Pick the template nearest this zone axis">
            Nearest
          </button>
        </span>
      </div>
      {error && <p className="field-error">{error}</p>}
      <div className="templates-panes">
        <figure>
          {ipf?.id === d.id ? <TemplatesIpfPlot ipf={ipf.ipf} picked={picked} onPick={pick} /> : <p className="muted small">Loading…</p>}
        </figure>
        <figure>
          {linked ? (
            view ? (
              <>
                <TemplatePatternPlot t={{ ...view, euler: [], reciprocal_radius: d.templates!.options.reciprocal_radius }} />
                <figcaption className="small">
                  {uvwLabel(view.uvw)}<span className="muted"> · template #{view.index + 1}, {formatValue(view.angle)}° off</span>
                </figcaption>
              </>
            ) : (
              <p className="muted small">Turn the structure (its 3D view above).</p>
            )
          ) : t && picked !== null ? (
            <>
              <TemplatePatternPlot t={t} />
              <figcaption className="small">
                #{t.index + 1} · {uvwLabel(t.uvw)}
                <span className="muted"> · φ₁ {formatValue(t.euler[0])}°, Φ {formatValue(t.euler[1])}°, φ₂ {formatValue(t.euler[2])}°</span>
              </figcaption>
            </>
          ) : (
            <p className="muted small">Click a template.</p>
          )}
        </figure>
      </div>
    </div>
  );
}
