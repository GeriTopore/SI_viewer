import { useEffect, useMemo, useState } from "react";
import type { DatasetSummary, MaskOptions } from "../api/client";
import { formatValue } from "../format";
import { useDatasets } from "../state/datasets";
import { loadValues, makeMask, maskSources, masksOf, selectMask, setOtsu, useMasks, useMaskState } from "../state/masks";
import { EditableTitle } from "./RenameInput";
import { ThresholdHistogram } from "./ThresholdHistogram";

const OUT = "var(--muted)";
const IN = "var(--accent)";

/** Process tab, images: masks made from the image, or from an image made
 * from it (a Bragg-filtered grating, an amplitude, …), by thresholding it
 * (smoothed first, if asked) and cleaning up the result; each a dataset
 * under the image it was made from, used by other processes (e.g.
 * sublattices). Listed here to select (its settings come back), rename or
 * delete. */
export function MaskSection({ d: target }: { d: DatasetSummary }) {
  const files = useDatasets((s) => s.files);
  const focusId = useDatasets((s) => s.focusId);
  const sources = useMemo(() => maskSources(files, target), [files, target]);
  const [chosen, setChosen] = useState<string | null>(null);
  // The image clicked (or its mask's image) to start with.
  const focused = sources.find((x) => x.id === focusId) ?? sources.find((x) => masksOf(files, x.id).some((m) => m.id === focusId));
  const d = sources.find((x) => x.id === chosen) ?? focused ?? target;
  useEffect(() => setChosen(null), [focusId]);
  const st = useMaskState(d.id);
  const list = useMemo(() => masksOf(files, d.id), [files, d.id]);
  const selected = list.find((m) => m.id === st.selected);
  const o = st.draft;
  const set = (p: Partial<MaskOptions>) => useMasks.getState().patch(d.id, { draft: { ...o, ...p } });
  const num = (v: string, fallback: number) => (v.trim() !== "" && Number.isFinite(Number(v)) ? Number(v) : fallback);
  const busy = !!st.busy;

  // The histogram follows the smoothing (and the image's data).
  const values = st.values?.smooth === o.smooth ? st.values : null;
  useEffect(() => {
    if (values) return;
    const t = setTimeout(() => loadValues(d.id), 250);
    return () => clearTimeout(t);
  }, [values, d.id, d.revision, o.smooth]);

  // Thresholds largest first, and which groups are in the mask.
  const thresholds = o.keep === "above" ? [o.lo] : o.keep === "below" ? [o.hi] : [o.hi, o.lo];
  const kept = o.keep === "above" ? [true, false] : o.keep === "below" ? [false, true] : [false, true, false];
  const inMask = kept.map((k) => k !== o.invert);
  const ready = thresholds.every((t) => t != null);
  const setThresholds = (t: number[]) => set(o.keep === "above" ? { lo: t[0] } : o.keep === "below" ? { hi: t[0] } : { hi: t[0], lo: t[1] });

  return (
    <section className="q-form">
      {sources.length > 1 && (
        <label title="The image the mask is made from">
          <span>From</span>
          <select value={d.id} onChange={(e) => setChosen(e.target.value)}>
            {sources.map((x) => (
              <option key={x.id} value={x.id}>
                {x.title}
              </option>
            ))}
          </select>
        </label>
      )}
      <h4 className="process-heading">Masks</h4>
      {list.length > 0 ? (
        <ul className="nb-list">
          {list.map((m) => (
            <li
              key={m.id}
              className={`nb-row${m.id === selected?.id ? " active" : ""}`}
              onClick={() => selectMask(d.id, m)}
              title="Work on this one (its settings)"
            >
              <span className="nb-row-name" onClick={(e) => m.id === selected?.id && e.stopPropagation()}>
                {m.id === selected?.id ? <EditableTitle d={m} /> : m.title}
              </span>
              <span className="muted small">{m.mask ? `${formatValue(m.mask.fraction * 100)} %` : ""}</span>
              <button
                className="icon-button"
                title="Delete it"
                onClick={(e) => {
                  e.stopPropagation();
                  useDatasets.getState().deleteDataset(m);
                }}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted small">None yet.</p>
      )}
      <label title="Gaussian blur (px) before thresholding; 0: none">
        <span>Smooth (px)</span>
        <input type="number" min={0} step={0.5} value={o.smooth} onChange={(e) => set({ smooth: Math.max(0, num(e.target.value, o.smooth)) })} />
      </label>
      <label title="The pixels in the mask">
        <span>Keep</span>
        <select value={o.keep} onChange={(e) => set({ keep: e.target.value as MaskOptions["keep"] })}>
          <option value="above">Above</option>
          <option value="below">Below</option>
          <option value="between">Between</option>
        </select>
      </label>
      {values && ready ? (
        <ThresholdHistogram
          values={values.values}
          thresholds={thresholds as number[]}
          colours={inMask.map((k) => (k ? IN : OUT))}
          labels={inMask.map((k) => (k ? "in" : "out"))}
          share
          onChange={setThresholds}
        />
      ) : (
        <p className="muted small">Loading the values…</p>
      )}
      {o.keep !== "below" && (
        <label>
          <span>Lower</span>
          <input type="number" step="any" value={o.lo ?? ""} onChange={(e) => set({ lo: num(e.target.value, o.lo ?? 0) })} />
        </label>
      )}
      {o.keep !== "above" && (
        <label>
          <span>Upper</span>
          <input type="number" step="any" value={o.hi ?? ""} onChange={(e) => set({ hi: num(e.target.value, o.hi ?? 0) })} />
        </label>
      )}
      <label className="q-check-line small">
        <input type="checkbox" checked={o.invert} onChange={(e) => set({ invert: e.target.checked })} />
        Invert
      </label>
      <label title="Remove specks and fill gaps up to this radius (px); 0: none">
        <span>Clean (px)</span>
        <input type="number" min={0} step={1} value={o.clean} onChange={(e) => set({ clean: Math.max(0, Math.round(num(e.target.value, o.clean))) })} />
      </label>
      <label title="Regions and holes under this size (px) are removed; 0: none">
        <span>Min area (px)</span>
        <input type="number" min={0} step={10} value={o.min_area} onChange={(e) => set({ min_area: Math.max(0, Math.round(num(e.target.value, o.min_area))) })} />
      </label>
      <div className="process-actions">
        <button disabled={!values} onClick={() => setOtsu(d.id)} title="The threshold by Otsu's method">
          Otsu
        </button>
        <button className="primary" disabled={busy || !ready} onClick={() => makeMask(d.id)} title="A new mask with these settings">
          {st.busy === "Adding" ? "Adding…" : "Add"}
        </button>
        {selected && (
          <button disabled={busy || !ready} onClick={() => makeMask(d.id, selected)} title={`Make ${selected.title} again with these settings`}>
            {st.busy === "Updating" ? "Updating…" : "Update"}
          </button>
        )}
      </div>
      {st.error && <p className="field-error">{st.error}</p>}
    </section>
  );
}

export const canMakeMask = (d: DatasetSummary) => d.kind === "image" && d.origin !== "mask" && d.axes.length === 2;
