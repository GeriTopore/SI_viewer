import { useEffect, useMemo, useRef, useState } from "react";
import { api, type DatasetSummary, type Filters, type SpatialFilter, type SpectralFilter } from "../api/client";
import { allDatasets, useDatasets } from "../state/datasets";
import { useLayers } from "../state/layers";
import { filterLabel } from "../format";
import { RebinDialog, canRebin } from "./RebinDialog";
import { AlignSection } from "./AlignSection";

const SPATIAL: { id: SpatialFilter; label: string; title: string }[] = [
  { id: "none", label: "None", title: "No spatial filter" },
  { id: "gaussian", label: "Gaussian", title: "Gaussian blur" },
  { id: "median", label: "Median", title: "Median filter: removes outliers, keeps edges" },
  { id: "mean", label: "Mean", title: "Mean (box) filter" },
];

const SPECTRAL: { id: SpectralFilter; label: string; title: string }[] = [
  { id: "none", label: "None", title: "No spectral smoothing" },
  { id: "gaussian", label: "Gaussian", title: "Gaussian smoothing along the energy axis" },
  { id: "savgol", label: "Savitzky–Golay", title: "Savitzky–Golay smoothing: keeps peak heights better" },
];

const ODD_SIZES = [3, 5, 7, 9, 11, 15, 21];
const SAVGOL_WINDOWS = [5, 7, 9, 11, 15, 21, 31];
// Filters are sent this long after the last change (e.g. while dragging σ).
const DEBOUNCE_MS = 250;

/** What the Process tab works on: the spectrum image in the viewer, whatever
 * is overlaid on it (its processing covers all its maps); else the focused
 * dataset, or for a map made from a spectrum image, that spectrum image. */
function useProcessTarget(): DatasetSummary | undefined {
  const files = useDatasets((s) => s.files);
  const selection = useDatasets((s) => s.selection);
  const focusId = useDatasets((s) => s.focusId);
  return useMemo(() => {
    const byId = new Map(allDatasets(files).map((d) => [d.id, d]));
    const shownSi = selection.map((id) => byId.get(id)).find((d) => d?.kind === "spectrum_image");
    if (shownSi) return shownSi;
    const focused = focusId ? byId.get(focusId) : undefined;
    return (focused?.parent_id && byId.get(focused.parent_id)) || focused;
  }, [files, selection, focusId]);
}

/** Process tab, in workflow order: rebinning (spectrum images), then
 * non-destructive filters. */
export function ProcessPanel() {
  const d = useProcessTarget();
  if (!d) return <p className="muted">Select a dataset to filter it.</p>;
  if (d.kind === "other") return <p className="muted">{d.title} can't be filtered.</p>;
  return <ProcessTarget key={d.id} d={d} />;
}

function ProcessTarget({ d }: { d: DatasetSummary }) {
  const replaceFile = useDatasets((s) => s.replaceFile);
  const files = useDatasets((s) => s.files);
  const notify = useLayers((s) => s.notify);
  const [draft, setDraft] = useState<Filters>(d.filters);
  const [error, setError] = useState<string | null>(null);
  const [rebinOpen, setRebinOpen] = useState(false);

  // Follow changes made elsewhere (e.g. "Apply to all maps" from another map).
  const sent = useRef(JSON.stringify(d.filters));
  useEffect(() => {
    const now = JSON.stringify(d.filters);
    if (now !== sent.current) {
      sent.current = now;
      setDraft(d.filters);
    }
  }, [d.filters]);

  const send = async (ids: string[], filters: Filters) => {
    setError(null);
    try {
      let notes: string[] = [];
      for (const id of ids) {
        const r = await api.setFilters(id, filters);
        replaceFile(r.file);
        notes = [...notes, ...r.notes];
      }
      if (notes.length) notify(notes.join(" "));
    } catch (e) {
      setError((e as Error).message);
    }
  };

  // Changes apply live, debounced.
  useEffect(() => {
    const json = JSON.stringify(draft);
    if (json === sent.current) return;
    const t = setTimeout(() => {
      sent.current = json;
      send([d.id], draft);
    }, DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [draft]); // eslint-disable-line react-hooks/exhaustive-deps

  // A change still waiting when another dataset is picked is sent at once.
  const latest = useRef(draft);
  latest.current = draft;
  useEffect(
    () => () => {
      const json = JSON.stringify(latest.current);
      if (json !== sent.current) {
        sent.current = json;
        send([d.id], latest.current);
      }
    },
    [], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const set = (patch: Partial<Filters>) => setDraft((f) => ({ ...f, ...patch }));

  const isSi = d.kind === "spectrum_image";
  const spatial = d.kind === "image" || d.kind === "stack" || isSi;
  const spectral = d.kind === "spectrum" || isSi;
  // A median isn't linear, so it can't be carried over to an SI's maps and spectra.
  const spatialChoices = isSi ? SPATIAL.filter((s) => s.id !== "median") : SPATIAL;
  // Maps loaded from a file (no spectrum image to follow) can share a filter.
  const otherMaps = d.is_element_map
    ? allDatasets(files).filter((x) => x.file_id === d.file_id && x.is_element_map && !x.parent_id && x.id !== d.id)
    : [];
  const unfiltered: Filters = { ...draft, spatial: "none", spectral: "none" };
  const active = filterLabel(draft);
  // The sections that apply, numbered in workflow order.
  const sections = [d.kind === "stack" && "align", canRebin(d) && "rebin", "filters"].filter(Boolean);
  const num = (s: string) => `${sections.indexOf(s) + 1}. `;

  return (
    <div className="quantify process">
      <section>
        <h3 className="q-title">{d.title}</h3>
        {isSi && (
          <p className="muted small q-elements">
            Processing the spectrum image: it applies to all its spectra, fits and element maps, whichever are
            overlaid in the viewer.
          </p>
        )}
      </section>

      {d.kind === "stack" && (
        <section className="q-maps">
          <h3>{num("align")}Align frames</h3>
          <AlignSection d={d} />
        </section>
      )}

      {canRebin(d) && (
        <section className="q-maps">
          <h3>{num("rebin")}Rebin</h3>
          <p className="muted small q-elements">
            Sums neighbouring pixels or channels of the spectrum image and its whole scan (HyperSpy rebin). This
            overwrites the loaded data; the file on disk is untouched.
          </p>
          <button className="process-rebin" onClick={() => setRebinOpen(true)}>
            Rebin…
          </button>
        </section>
      )}

      <section className="q-maps">
        <h3>{num("filters")}Filters</h3>
        <p className="muted small q-elements">
          Filters apply whenever the data is shown; the loaded data is unchanged and they can be removed at any
          time.
        </p>
      </section>

      {spatial && (
        <section className="q-form">
          <h4 className="process-heading">Spatial filter</h4>
          <div className="segmented process-choice" role="group" aria-label="Spatial filter">
            {spatialChoices.map((s) => (
              <button
                key={s.id}
                className={draft.spatial === s.id ? "active" : undefined}
                aria-pressed={draft.spatial === s.id}
                title={s.title}
                onClick={() => set({ spatial: s.id })}
              >
                {s.label}
              </button>
            ))}
          </div>
          {draft.spatial === "gaussian" && (
            <label title="Standard deviation of the Gaussian, in pixels">
              σ (px)
              <span className="q-inline process-slider">
                <input
                  type="range"
                  min={0.3}
                  max={10}
                  step={0.1}
                  value={draft.sigma}
                  onChange={(e) => set({ sigma: Number(e.target.value) })}
                />
                <input
                  type="number"
                  min={0.1}
                  max={50}
                  step={0.1}
                  value={draft.sigma}
                  onChange={(e) => Number(e.target.value) > 0 && set({ sigma: Number(e.target.value) })}
                />
              </span>
            </label>
          )}
          {(draft.spatial === "median" || draft.spatial === "mean") && (
            <label title="Width of the square window, in pixels">
              Size
              <select value={draft.size} onChange={(e) => set({ size: Number(e.target.value) })}>
                {ODD_SIZES.map((n) => (
                  <option key={n} value={n}>
                    {n} × {n} px
                  </option>
                ))}
              </select>
            </label>
          )}
          {isSi && (
            <p className="muted small process-note">
              Blurs the whole spectrum image: cursor and region spectra, the total-counts and energy-window maps,
              and every element map (also ones added later from the periodic table). Fits and quantification use
              the blurred data; fitted maps are removed when the processing changes, so fit again.
            </p>
          )}
        </section>
      )}

      {spectral && (
        <section className="q-form">
          <h4 className="process-heading">Spectral smoothing</h4>
          <div className="segmented process-choice" role="group" aria-label="Spectral smoothing">
            {SPECTRAL.map((s) => (
              <button
                key={s.id}
                className={draft.spectral === s.id ? "active" : undefined}
                aria-pressed={draft.spectral === s.id}
                title={s.title}
                onClick={() => set({ spectral: s.id })}
              >
                {s.label}
              </button>
            ))}
          </div>
          {draft.spectral === "gaussian" && (
            <label title="Standard deviation of the Gaussian, in channels">
              σ (channels)
              <span className="q-inline process-slider">
                <input
                  type="range"
                  min={0.3}
                  max={10}
                  step={0.1}
                  value={draft.spectral_sigma}
                  onChange={(e) => set({ spectral_sigma: Number(e.target.value) })}
                />
                <input
                  type="number"
                  min={0.1}
                  max={50}
                  step={0.1}
                  value={draft.spectral_sigma}
                  onChange={(e) => Number(e.target.value) > 0 && set({ spectral_sigma: Number(e.target.value) })}
                />
              </span>
            </label>
          )}
          {draft.spectral === "savgol" && (
            <>
              <label title="Number of channels in each local polynomial fit">
                Window
                <select value={draft.window} onChange={(e) => set({ window: Number(e.target.value) })}>
                  {SAVGOL_WINDOWS.map((n) => (
                    <option key={n} value={n} disabled={n <= draft.order}>
                      {n} channels
                    </option>
                  ))}
                </select>
              </label>
              <label title="Order of the local polynomial">
                Order
                <select value={draft.order} onChange={(e) => set({ order: Number(e.target.value) })}>
                  {[0, 1, 2, 3, 4, 5, 6].map((n) => (
                    <option key={n} value={n} disabled={n >= draft.window}>
                      {n}
                    </option>
                  ))}
                </select>
              </label>
            </>
          )}
          <p className="muted small process-note">
            {isSi
              ? "Smooths every spectrum of the spectrum image, and the maps and fits made from them."
              : "Smooths the spectrum; fits use the smoothed spectrum."}
          </p>
        </section>
      )}

      <section className="process-actions">
        {otherMaps.length > 0 && (
          <button
            onClick={() => {
              sent.current = JSON.stringify(draft);
              send([d.id, ...otherMaps.map((m) => m.id)], draft);
            }}
            title="Give every element map of this file the same filter"
          >
            Apply to all {otherMaps.length + 1} maps
          </button>
        )}
        <button onClick={() => setDraft(unfiltered)} disabled={!active}>
          Remove filters
        </button>
        {error && <p className="field-error">{error}</p>}
      </section>

      {rebinOpen && <RebinDialog d={d} onClose={() => setRebinOpen(false)} />}
    </div>
  );
}
