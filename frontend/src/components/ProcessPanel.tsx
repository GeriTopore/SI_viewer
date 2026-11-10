import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { DatasetSummary, Filters, SpatialFilter, SpectralFilter } from "../api/client";
import { allDatasets, useDatasets } from "../state/datasets";
import { applyFilters } from "../state/filters";
import { useAlignment } from "../state/alignment";
import { useUi } from "../state/ui";
import { filterLabel, isInsituView, isLineScan } from "../format";
import { RebinDialog, canRebin } from "./RebinDialog";
import { AlignSection } from "./AlignSection";
import { BackgroundSection } from "./BackgroundSection";
import { canRemoveBackground } from "../state/background";
import { ThicknessSection } from "./ThicknessSection";
import { DeconvolutionSection } from "./DeconvolutionSection";
import { canDeconvolve } from "../state/deconvolution";
import { canEstimateThickness } from "../state/thickness";
import { ZlpSection } from "./ZlpSection";

const SPATIAL: { id: SpatialFilter; label: string; title: string }[] = [
  { id: "none", label: "None", title: "No spatial filter" },
  { id: "gaussian", label: "Gaussian", title: "Gaussian blur" },
  { id: "median", label: "Median", title: "Median filter: removes outliers, keeps edges" },
  { id: "mean", label: "Mean", title: "Mean (box) filter" },
  { id: "bandpass", label: "Bandpass", title: "Fourier bandpass: removes structures smaller than Small and larger than Large" },
];

const SPECTRAL: { id: SpectralFilter; label: string; title: string }[] = [
  { id: "none", label: "None", title: "No spectral smoothing" },
  { id: "gaussian", label: "Gaussian", title: "Gaussian smoothing along the energy axis" },
  { id: "savgol", label: "Savitzky–Golay", title: "Savitzky–Golay smoothing: keeps peak heights better" },
];

/** What is wrong with the bandpass sizes (not sent until fixed), if anything. */
const bandpassError = (f: Filters) =>
  f.spatial === "bandpass" && f.small >= f.large ? "Small must be less than Large" : null;

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

/** One process of the tab: its title (what it does as a tooltip) folds it
 * away, remembered per browser; ``summary`` shows beside a folded title. */
function ProcessGroup({ id, title, tip, summary, children }: {
  id: string; title: string; tip: string; summary?: string | null; children: ReactNode;
}) {
  const collapsed = useUi((s) => s.processCollapsed.includes(id));
  const toggle = useUi((s) => s.toggleProcess);
  return (
    <div className={`q-maps process-group${collapsed ? " collapsed" : ""}`}>
      <h3 className="process-group-title">
        <button className="process-group-toggle" aria-expanded={!collapsed} onClick={() => toggle(id)} title={tip}>
          <span className="process-chevron" aria-hidden="true" />
          {title}
        </button>
        {collapsed && summary && <span className="muted small">{summary}</span>}
      </h3>
      {!collapsed && children}
    </div>
  );
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
  const files = useDatasets((s) => s.files);
  const [draft, setDraft] = useState<Filters>(d.filters);
  const [error, setError] = useState<string | null>(null);
  const [rebinOpen, setRebinOpen] = useState(false);
  // No dataset name while its alignment workspace is open.
  const aligning = useAlignment((s) => s.openId) === d.id;

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
      await applyFilters(ids, filters);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  // Changes apply live, debounced.
  useEffect(() => {
    const json = JSON.stringify(draft);
    if (json === sent.current || bandpassError(draft)) return;
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
      if (json !== sent.current && !bandpassError(latest.current)) {
        sent.current = json;
        send([d.id], latest.current);
      }
    },
    [], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const set = (patch: Partial<Filters>) => setDraft((f) => ({ ...f, ...patch }));

  const isSi = d.kind === "spectrum_image";
  // A line scan has no 2D scan to blur, and no maps to follow.
  const lineScan = isLineScan(d);
  const spatial = d.kind === "image" || d.kind === "stack" || (isSi && !lineScan);
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
  const sections = [
    d.kind === "stack" && "align", canRebin(d) && "rebin", canEstimateThickness(d) && "zlp", "filters",
    canRemoveBackground(d) && "background",
    canEstimateThickness(d) && "thickness",
    canDeconvolve(d) && "deconvolution",
  ].filter(Boolean);
  const num = (s: string) => `${sections.indexOf(s) + 1}. `;

  return (
    <div className="quantify process">
      {!aligning && (
        <section>
          <h3 className="q-title">{d.title}</h3>
        </section>
      )}

      {d.kind === "stack" && (
        <ProcessGroup id="align" title={`${num("align")}Align frames`} tip="Register the frames to remove drift, then average them">
          <AlignSection d={d} />
        </ProcessGroup>
      )}

      {canRebin(d) && (
        <ProcessGroup id="rebin" title={`${num("rebin")}Rebin`} tip="Sum neighbouring pixels or channels; overwrites the loaded data">
          <button
            className="process-rebin"
            onClick={() => setRebinOpen(true)}
            disabled={isInsituView(d)}
            title={isInsituView(d) ? "Not for in-situ frames: the next frame would undo it" : undefined}
          >
            Rebin…
          </button>
        </ProcessGroup>
      )}

      {canEstimateThickness(d) && (
        <ProcessGroup
          id="zlp"
          title={`${num("zlp")}Zero-loss alignment`}
          tip="Shift every spectrum so its zero-loss peak is at 0 eV; linked core losses follow"
        >
          <ZlpSection d={d} />
        </ProcessGroup>
      )}

      <ProcessGroup
        id="filters"
        title={`${num("filters")}Filters`}
        tip="Applied whenever the data is shown; the loaded data is unchanged"
        summary={active}
      >
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
            {draft.spatial === "bandpass" && (
              <>
                <label title="Structures smaller than this are removed, in pixels">
                  Small (px)
                  <input
                    type="number"
                    min={0.1}
                    max={10000}
                    step={0.5}
                    value={draft.small}
                    onChange={(e) => Number(e.target.value) > 0 && set({ small: Number(e.target.value) })}
                  />
                </label>
                <label title="Structures larger than this are removed, in pixels">
                  Large (px)
                  <input
                    type="number"
                    min={0.1}
                    max={10000}
                    step={1}
                    value={draft.large}
                    onChange={(e) => Number(e.target.value) > 0 && set({ large: Number(e.target.value) })}
                  />
                </label>
                {bandpassError(draft) && <p className="field-error">{bandpassError(draft)}</p>}
              </>
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
      </ProcessGroup>

      {canRemoveBackground(d) && (
        <ProcessGroup
          id="background"
          title={`${num("background")}Background removal`}
          tip="Fit and subtract the background under an edge; makes a new spectrum image"
        >
          <BackgroundSection d={d} />
        </ProcessGroup>
      )}

      {canEstimateThickness(d) && (
        <ProcessGroup
          id="thickness"
          title={`${num("thickness")}Thickness`}
          tip="Thickness of the low loss from the share of its counts in the zero-loss peak; makes a new map"
        >
          <ThicknessSection d={d} />
        </ProcessGroup>
      )}

      {canDeconvolve(d) && (
        <ProcessGroup
          id="deconvolution"
          title={`${num("deconvolution")}Deconvolution`}
          tip="Remove plural scattering (or sharpen) with the low loss; makes a new spectrum image"
        >
          <DeconvolutionSection d={d} />
        </ProcessGroup>
      )}

      {rebinOpen && <RebinDialog d={d} onClose={() => setRebinOpen(false)} />}
    </div>
  );
}
