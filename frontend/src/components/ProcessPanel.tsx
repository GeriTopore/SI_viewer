import { useEffect, useMemo, useRef, useState, type DragEvent, type ReactNode } from "react";
import { create } from "zustand";
import type { DatasetSummary, Filters, SpatialFilter, SpectralFilter } from "../api/client";
import { allDatasets, useDatasets } from "../state/datasets";
import { applyFilters } from "../state/filters";
import { useAlignment } from "../state/alignment";
import { PANE_SIZES, useUi, type SplitSide } from "../state/ui";
import { Splitter } from "./Splitter";
import { followProcess } from "./ProcessLink";
import { filterLabel, isInsituView, isLineScan } from "../format";
import { InsituAlignSection } from "./InsituAlignSection";
import { RebinDialog, canRebin } from "./RebinDialog";
import { BadPixelSection } from "./BadPixelSection";
import { PatternBackgroundSection } from "./PatternBackgroundSection";
import { StrainSection } from "./StrainSection";
import { CalibrationSection } from "./CalibrationSection";
import { CentringSection } from "./CentringSection";
import { DiskDetectionSection } from "./DiskDetectionSection";
import { VirtualImageSection } from "./VirtualImageSection";
import { AlignSection } from "./AlignSection";
import { BackgroundSection } from "./BackgroundSection";
import { canRemoveBackground } from "../state/background";
import { ThicknessSection } from "./ThicknessSection";
import { DeconvolutionSection } from "./DeconvolutionSection";
import { canDeconvolve } from "../state/deconvolution";
import { canEstimateThickness } from "../state/thickness";
import { ZlpSection } from "./ZlpSection";
import { DpcSection, canMakeDpc } from "./DpcSection";
import { PcaSection, hasPcaSection } from "./PcaSection";
import { NormaliseSection } from "./NormaliseSection";
import { AtomicSection, canAnalyseAtomic } from "./AtomicSection";
import { MaskSection, canMakeMask } from "./MaskSection";
import { ColumnsSection } from "./ColumnsSection";
import { SublatticesSection } from "./SublatticesSection";
import { NeighbourhoodSection } from "./NeighbourhoodSection";
import { PldSection } from "./PldSection";
import { CurveFitPanel } from "./CurveFitPanel";
import { canNormalise } from "../state/normalise";
import { ProcessIcon } from "./ProcessIcons";
import { GridIcon, ListIcon } from "./DatasetList";

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

/** Drag-and-drop type for a process (its icon or title), to split it off. */
const PROCESS_MIME = "application/x-si-viewer-process";
/** The process being dragged, if any (the tab shows where it would go). */
const useProcessDrag = create<{ id: string | null }>(() => ({ id: null }));
const dragProps = (id: string) => ({
  draggable: true,
  onDragStart: (e: DragEvent) => {
    e.dataTransfer.setData(PROCESS_MIME, id);
    e.dataTransfer.effectAllowed = "move";
    useProcessDrag.setState({ id });
  },
  onDragEnd: () => useProcessDrag.setState({ id: null }),
});

/** What the Process tab works on: the spectrum image in the viewer (or the
 * focused member of its linked group), whatever is overlaid on it (its
 * processing covers all its maps); else the focused
 * dataset, or for a map made from a spectrum image, that spectrum image. */
function useProcessTarget(): DatasetSummary | undefined {
  const files = useDatasets((s) => s.files);
  const selection = useDatasets((s) => s.selection);
  const focusId = useDatasets((s) => s.focusId);
  return useMemo(() => {
    const byId = new Map(allDatasets(files).map((d) => [d.id, d]));
    const shownSi = selection.map((id) => byId.get(id)).find((d) => d?.kind === "spectrum_image");
    const focused = focusId ? byId.get(focusId) : undefined;
    // A column item (kemstem): the image it came from.
    if (focused && ["columns", "sublattice", "neighbourhood", "bonds", "displacements"].includes(focused.atomic?.output ?? "")) {
      let up: DatasetSummary | undefined = focused;
      while (up && up.atomic?.output !== "columns") up = byId.get(up.parent_id ?? "");
      const image = up?.parent_id ? byId.get(up.parent_id) : undefined;
      if (image) return image;
    }
    // A member of the linked group in the viewer is processed when focused.
    if (focused && shownSi?.eels?.linked.includes(focused.id)) return focused;
    if (shownSi) return shownSi;
    // A map made from a spectrum image is processed with it; anything else
    // (a crop, an aligned average, a filtered or DPC image, a stack) is
    // processed itself, as highlighted.
    const parent = focused?.parent_id ? byId.get(focused.parent_id) : undefined;
    if (focused && focused.kind !== "stack" && (parent?.kind === "spectrum_image" || parent?.kind === "spectrum")) return parent;
    return focused;
  }, [files, selection, focusId]);
}

/** One process of the tab: what it does (``tip``), its settings (``body``),
 * and ``summary`` beside its folded title. */
interface Process {
  id: string;
  title: string;
  tip: string;
  /** Its section of the tab (default "General"). */
  section?: string;
  summary?: string | null;
  body: ReactNode;
}

/** Sections that come first, in this order (4D-STEM data: cleaning the
 * patterns before analysing them); the others follow in order of their
 * first process. */
const SECTION_ORDER = ["Pre-processing", "Analysis"];

/** ``processes`` sorted by section (``SECTION_ORDER``, then as they come),
 * keeping their order within each, so the numbers run through. */
function bySection(processes: Process[]): Process[] {
  const first = new Map<string, number>();
  processes.forEach((p, i) => first.set(p.section ?? "General", first.get(p.section ?? "General") ?? i));
  const rank = (p: Process) => {
    const name = p.section ?? "General";
    const k = SECTION_ORDER.indexOf(name);
    return k >= 0 ? k - SECTION_ORDER.length : first.get(name)!;
  };
  return processes.map((p, i) => ({ p, i })).sort((a, b) => rank(a.p) - rank(b.p) || a.i - b.i).map(({ p }) => p);
}

/** The processes by section, in order of their first process; numbered
 * through. */
function sectionsOf(processes: Process[]): { name: string; items: { p: Process; n: number }[] }[] {
  const out: { name: string; items: { p: Process; n: number }[] }[] = [];
  processes.forEach((p, i) => {
    const name = p.section ?? "General";
    let s = out.find((x) => x.name === name);
    if (!s) out.push((s = { name, items: [] }));
    s.items.push({ p, n: i + 1 });
  });
  return out;
}

/** A section's header (when the tab has more than one); it folds the
 * section, remembered per browser. */
function SectionHeader({ name, count }: { name: string; count: number }) {
  const collapsed = useUi((s) => s.processCollapsed.includes(`section:${name}`));
  const toggle = useUi((s) => s.toggleProcess);
  return (
    <h3 className={`process-section-title${collapsed ? " collapsed" : ""}`}>
      <button className="process-group-toggle" aria-expanded={!collapsed} onClick={() => toggle(`section:${name}`)}>
        <span className="process-chevron" aria-hidden="true" />
        {name}
      </button>
      {collapsed && <span className="muted small">{count}</span>}
    </h3>
  );
}

/** The processes in sections (headed when there are several). */
function Sections({ processes, render }: { processes: Process[]; render: (items: { p: Process; n: number }[]) => ReactNode }) {
  const sections = sectionsOf(processes);
  const collapsed = useUi((s) => s.processCollapsed);
  if (sections.length < 2) return <>{render(sections[0]?.items ?? [])}</>;
  return (
    <>
      {sections.map((s) => (
        <div key={s.name} className="process-section">
          <SectionHeader name={s.name} count={s.items.length} />
          {!collapsed.includes(`section:${s.name}`) && render(s.items)}
        </div>
      ))}
    </>
  );
}

/** List view: every process as a card whose title folds it away,
 * remembered per browser. */
function ProcessList({ processes }: { processes: Process[] }) {
  return (
    <Sections
      processes={processes}
      render={(items) => (
        <div className="process-list">
          {items.map(({ p, n }) => (
            <ProcessGroup key={p.id} p={p} n={n} />
          ))}
        </div>
      )}
    />
  );
}

function ProcessGroup({ p, n }: { p: Process; n: number }) {
  const collapsed = useUi((s) => s.processCollapsed.includes(p.id));
  const toggle = useUi((s) => s.toggleProcess);
  return (
    <div className={`process-group${collapsed ? " collapsed" : ""}`} data-process={p.id}>
      <h3 className="process-group-title">
        <button
          className="process-group-toggle"
          aria-expanded={!collapsed}
          onClick={() => toggle(p.id)}
          title={`${p.tip}\nDrag onto the tab to show it beside the others`}
          {...dragProps(p.id)}
        >
          <span className="process-chevron" aria-hidden="true" />
          <ProcessIcon id={p.id} size={20} />
          {n}. {p.title}
        </button>
        {collapsed && p.summary && <span className="muted small">{p.summary}</span>}
      </h3>
      {!collapsed && (
        <div className="process-group-body" onPointerDownCapture={() => followProcess(p.id)}>
          {p.body}
        </div>
      )}
    </div>
  );
}

/** Icon view: a tile per process; a tile opens its process in the tab. */
function ProcessGrid({ processes }: { processes: Process[] }) {
  const openId = useUi((s) => s.processOpen);
  const setOpen = useUi((s) => s.setProcessOpen);
  const i = processes.findIndex((p) => p.id === openId);
  const open = processes[i];
  if (open)
    return (
      <div className="process-group process-open">
        <h3 className="process-group-title">
          <button className="link-button process-back" onClick={() => setOpen(null)} title="All processes">
            ‹ All
          </button>
          <span className="process-drag-handle" title={`${open.tip}\nDrag onto the tab to show it beside the others`} {...dragProps(open.id)}>
            <ProcessIcon id={open.id} size={20} />
            {i + 1}. {open.title}
          </span>
        </h3>
        <div className="process-group-body" onPointerDownCapture={() => followProcess(open.id)}>
          {open.body}
        </div>
      </div>
    );
  return (
    <Sections
      processes={processes}
      render={(items) => (
        <div className="process-grid">
          {items.map(({ p, n }) => (
            <button
              key={p.id}
              className="process-tile"
              onClick={() => {
                setOpen(p.id);
                followProcess(p.id);
              }}
              title={`${p.tip}\nDrag onto the tab to show it beside the others`}
              {...dragProps(p.id)}
            >
              <span className="process-tile-n">{n}</span>
              <ProcessIcon id={p.id} />
              <span className="process-tile-name">{p.title}</span>
              {p.summary && <span className="process-tile-summary">{p.summary}</span>}
            </button>
          ))}
        </div>
      )}
    />
  );
}

function ProcessModeToggle() {
  const mode = useUi((s) => s.processMode);
  const setMode = useUi((s) => s.setProcessMode);
  return (
    <div className="segmented" role="group" aria-label="Process view">
      <button className={mode === "list" ? "active" : undefined} aria-pressed={mode === "list"} title="List view" onClick={() => setMode("list")}>
        <ListIcon />
      </button>
      <button className={mode === "grid" ? "active" : undefined} aria-pressed={mode === "grid"} title="Icon view" onClick={() => setMode("grid")}>
        <GridIcon />
      </button>
    </div>
  );
}

/** Process tab, in workflow order: rebinning (spectrum images), then
 * non-destructive filters. */
export function ProcessPanel() {
  const d = useProcessTarget();
  const split = useUi((s) => s.processSplit);
  const files = useDatasets((s) => s.files);
  // Other kinds have no processes, except Bragg vectors (their Calibration).
  const processable = (x: DatasetSummary) => x.kind !== "other" || x.origin === "disks";
  const main = !d ? (
    <p className="muted">Select a dataset to filter it.</p>
  ) : !processable(d) ? (
    <p className="muted">{d.title} can't be filtered.</p>
  ) : (
    <ProcessTarget key={d.id} d={d} />
  );
  // The split-off process: on its pinned dataset (while it is open), else on the tab's.
  const pinned = split?.image ? allDatasets(files).find((x) => x.id === split.image) : undefined;
  const target = pinned ?? d;
  const second =
    split && target && processable(target) ? (
      <ProcessTarget key={`split:${target.id}`} d={target} only={split.id} pinned={!!pinned} />
    ) : split ? (
      <SplitHeaderOnly id={split.id} />
    ) : null;
  return <ProcessSplitView main={main} second={second} target={d} />;
}

/** The Process tab, split or not: dragging a process's icon or title onto it
 * shows that process beside (or above, below) the rest, as the bottom
 * pane's tabs split; the line between them resizes them. */
function ProcessSplitView({ main, second, target }: { main: ReactNode; second: ReactNode; target: DatasetSummary | undefined }) {
  const split = useUi((s) => s.processSplit);
  const fraction = useUi((s) => s.processSplitFraction);
  const setFraction = useUi((s) => s.setProcessSplitFraction);
  const dragging = useProcessDrag((s) => s.id);
  const [hover, setHover] = useState<SplitSide | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const isDrag = (e: DragEvent) => e.dataTransfer.types.includes(PROCESS_MIME);
  // Left / right third: side by side; middle third: above or below.
  const sideOf = (e: DragEvent): SplitSide => {
    const r = bodyRef.current!.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width;
    if (x < 1 / 3) return "left";
    if (x > 2 / 3) return "right";
    return e.clientY < r.top + r.height / 2 ? "top" : "bottom";
  };
  const drop = (e: DragEvent) => {
    if (!isDrag(e)) return;
    e.preventDefault();
    const id = e.dataTransfer.getData(PROCESS_MIME) || dragging;
    const side = sideOf(e);
    if (id) {
      const ui = useUi.getState();
      // Kept on the dataset the tab is working on (or the one it was pinned to).
      const image = split?.id === id ? split.image : (target?.id ?? null);
      ui.setProcessSplit({ id, side, image });
      if (ui.processOpen === id) ui.setProcessOpen(null);
      followProcess(id);
    }
    setHover(null);
    useProcessDrag.setState({ id: null });
  };
  const first = split && (split.side === "left" || split.side === "top");
  const rows = split ? split.side === "top" || split.side === "bottom" : false;
  const halves = split
    ? [
        <div key="split" className="process-half split-off" style={{ flexBasis: `${fraction * 100}%` }}>
          {second}
        </div>,
        <div key="main" className="process-half">
          {main}
        </div>,
      ]
    : [];
  if (split && !first) halves.reverse();
  return (
    <div ref={bodyRef} className={`process-split${split ? ` split ${rows ? "rows" : "cols"}` : ""}`}>
      {split ? (
        <>
          {halves[0]}
          <Splitter
            direction={rows ? "row" : "col"}
            className={rows ? "split-row-resizer" : "split-resizer"}
            label="Resize the two processes"
            onDrag={(pos) => {
              const r = bodyRef.current!.getBoundingClientRect();
              const f = rows ? (pos - r.top) / r.height : (pos - r.left) / r.width;
              setFraction(first ? f : 1 - f);
            }}
            onReset={() => setFraction(PANE_SIZES.processSplit.initial)}
          />
          {halves[1]}
        </>
      ) : (
        <div className="process-half">{main}</div>
      )}
      {dragging && (
        <div
          className="split-drop"
          onDragOver={(e) => {
            if (!isDrag(e)) return;
            e.preventDefault();
            e.dataTransfer.dropEffect = "move";
            setHover(sideOf(e));
          }}
          onDragLeave={() => setHover(null)}
          onDrop={drop}
        >
          {hover && <div className={`hot ${hover}`} />}
        </div>
      )}
    </div>
  );
}

/** The split-off process's title: its icon (dragged to move it), the
 * dataset it works on, a pin keeping it there, and a close button. */
function SplitHeader({ id, title, tip, on, pinned }: {
  id: string;
  title: string;
  tip?: string;
  /** The dataset it works on. */
  on?: DatasetSummary;
  pinned?: boolean;
}) {
  const split = useUi((s) => s.processSplit);
  return (
    <h3 className="process-group-title process-split-title">
      <span className="process-drag-handle" title={`${tip ? `${tip}\n` : ""}Drag to move it`} {...dragProps(id)}>
        <ProcessIcon id={id} size={20} />
        {title}
      </span>
      <span className="process-split-on" />
      {split && on && (
        <button
          className={`icon-toggle${pinned ? " active" : ""}`}
          aria-pressed={!!pinned}
          title={pinned ? `Stays on ${on.title} (click to follow the dataset clicked)` : "Follows the dataset clicked (click to keep it on this one)"}
          onClick={() => {
            const ui = useUi.getState();
            const cur = ui.processSplit;
            if (cur) ui.setProcessSplit({ ...cur, image: pinned ? null : on.id });
          }}
        >
          <PinIcon />
        </button>
      )}
      <button className="icon-button" title="Close" onClick={() => useUi.getState().setProcessSplit(null)}>
        ×
      </button>
    </h3>
  );
}
/** The split-off process when there is nothing to work on. */
function SplitHeaderOnly({ id }: { id: string }) {
  return (
    <div className="process-group process-open">
      <SplitHeader id={id} title={id} />
      <p className="muted small">Select a dataset to work on.</p>
    </div>
  );
}

const PinIcon = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M9.5 1.5l5 5-2 .5-3 3 .5 3-1.5 1.5-3.5-3.5L2 15M5 7.5L8.5 11M6.5 5.5l3-3" />
  </svg>
);

function ProcessTarget({ d, only, pinned }: { d: DatasetSummary; only?: string; pinned?: boolean }) {
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
  // Bragg vectors: their 4D data (their Calibration and Strain processes work on the data's records).
  const braggCube = d.origin === "disks" ? allDatasets(files).find((x) => x.id === d.parent_id && x.kind === "4dstem") : undefined;
  const unfiltered: Filters = { ...draft, spatial: "none", spectral: "none" };
  const active = filterLabel(draft);
  const mode = useUi((s) => s.processMode);
  // The processes that apply, in workflow order.
  const rebinBody = (
    <button className="process-rebin" onClick={() => setRebinOpen(true)}>
      Rebin…
    </button>
  );
  const processes = bySection(([
    d.kind === "stack" && d.origin !== "pca" && {
      id: "align", title: "Align frames", tip: "Register the frames to remove drift, then average them",
      body: <AlignSection d={d} />,
    },
    canMakeDpc(d) && {
      id: "dpc", title: "DPC / iDPC",
      tip: "Differential phase contrast from the four quadrants of a segmented detector, and its integral",
      body: <DpcSection d={d} />,
    },
    // 4D-STEM data: Pre-processing (Rebin first), then Analysis.
    canRebin(d) && d.kind === "4dstem" && {
      id: "rebin", title: "Rebin", section: "Pre-processing", tip: "Sum neighbouring scan positions or pattern pixels; overwrites the loaded data",
      body: rebinBody,
    },
    d.kind === "4dstem" && {
      id: "virtual-image", title: "Virtual imaging", section: "Analysis",
      tip: "The virtual detector drawn on the pattern: its shape, response, apertures; keep the image as a dataset",
      body: <VirtualImageSection key={d.id} d={d} />,
    },
    d.kind === "4dstem" && {
      id: "bad-pixels", title: "Remove bad pixels", section: "Pre-processing",
      tip: "Hot and dead detector pixels, found in the mean pattern and repaired in every pattern",
      body: <BadPixelSection key={d.id} d={d} />,
    },
    d.kind === "4dstem" && {
      id: "background", title: "Background", section: "Pre-processing",
      tip: "Diffuse scattering subtracted from every pattern (pyxem: median kernel, difference of Gaussians, radial median, h-dome), tried on the patterns shown",
      body: <PatternBackgroundSection key={d.id} d={d} />,
    },
    d.kind === "4dstem" && {
      id: "centring", title: "Fix descan", section: "Pre-processing",
      tip: "The direct beam found in every pattern, the descan as a plane, and every pattern shifted so the beam is at its centre",
      body: <CentringSection key={d.id} d={d} />,
    },
    d.kind === "4dstem" && {
      id: "disks", title: "Disk detection", section: "Analysis",
      tip: "Bragg disks in every pattern: matched with the probe, tuned on the patterns shown, kept as Bragg vectors and their map",
      body: <DiskDetectionSection key={d.id} d={d} />,
    },
    braggCube && {
      id: "calibration", title: "Calibration",
      tip: "The Bragg vectors' calibration: each position's origin (descan), the QR rotation (least curl), the pattern's pixel size from a ring of a standard (or the camera length), its centre and ellipticity",
      body: <CalibrationSection key={braggCube.id} d={braggCube} />,
    },
    braggCube && {
      id: "strain", title: "Strain mapping",
      tip: "Strain from the Bragg vectors: the lattice (O, g1, g2) placed on the Bragg vector map, every pattern's disks indexed and fitted to it, then εxx, εyy, εxy and ωxy against a reference",
      body: <StrainSection key={braggCube.id} d={braggCube} />,
    },
    canRebin(d) && d.origin !== "pca" && d.kind !== "4dstem" && {
      id: "rebin", title: "Rebin", tip: "Sum neighbouring pixels or channels; overwrites the loaded data",
      body: rebinBody,
    },
    isInsituView(d) && d.kind === "spectrum_image" && !isLineScan(d) && {
      id: "aligned-sum", title: "Aligned sum",
      tip: "Register every frame with the measured drift and sum them; makes a new spectrum image",
      body: <InsituAlignSection d={d} />,
    },
    canEstimateThickness(d) && {
      id: "zlp", title: "ZLP alignment",
      tip: "Shift every spectrum so its zero-loss peak is at 0 eV; linked core losses follow. Without a low loss: shift by hand",
      body: <ZlpSection d={d} />,
    },
    // Not on Bragg vectors (a table: nothing to filter).
    d.origin !== "disks" && {
      id: "filters", title: "Filters", tip: "Applied whenever the data is shown; the loaded data is unchanged",
      ...(d.kind === "4dstem" && { section: "Pre-processing" }),
      summary: active,
      body: (
        <>
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
        </>
      ),
    },
    canNormalise(d) && {
      id: "normalise", title: "Normalisation",
      tip: "Normalise the spectra on the plot: to 0–1, a value at an energy or a mean over a window",
      body: <NormaliseSection d={d} />,
    },
    canRemoveBackground(d) && {
      id: "background", title: "Background removal",
      tip: "Fit and subtract the background under an edge; makes a new spectrum image",
      body: <BackgroundSection d={d} />,
    },
    canEstimateThickness(d) && {
      id: "thickness", title: "Thickness",
      tip: "Thickness of the low loss from the share of its counts in the zero-loss peak; makes a new map",
      body: <ThicknessSection d={d} />,
    },
    canDeconvolve(d) && {
      id: "deconvolution", title: "Deconvolution",
      tip: "Remove plural scattering (or sharpen) with the low loss; makes a new spectrum image",
      body: <DeconvolutionSection d={d} />,
    },
    (d.kind === "spectrum" || d.kind === "spectrum_image") && {
      id: "curvefit", title: "Curve fitting",
      tip: "Fit a model of HyperSpy's components (peaks, backgrounds, steps) to a spectrum",
      body: <CurveFitPanel key={d.id} ds={d} />,
    },
    canMakeMask(d) && {
      id: "mask", title: "Masks",
      tip: "Masks of the image's pixels between thresholds, for other processes (e.g. sublattices)",
      body: <MaskSection key={d.id} d={d} />,
    },
    canAnalyseAtomic(d) && {
      id: "fourier", title: "Fourier filtering", section: "kemstem",
      tip: "Pick FFT peaks, then filter round them: amplitude, phase and strain maps (kemstem)",
      body: <AtomicSection key={d.id} d={d} />,
    },
    canAnalyseAtomic(d) && {
      id: "columns", title: "Atomic columns", section: "kemstem",
      tip: "Find the atomic columns and fit each with a 2D Gaussian (kemstem)",
      body: <ColumnsSection key={d.id} d={d} />,
    },
    canAnalyseAtomic(d) && {
      id: "sublattices", title: "Sublattices", section: "kemstem",
      tip: "Split the fitted atomic columns into sublattices",
      body: <SublatticesSection key={d.id} d={d} />,
    },
    canAnalyseAtomic(d) && {
      id: "neighbourhoods", title: "Neighbourhoods", section: "kemstem",
      tip: "Vectors to each column's nearest neighbours, their clusters, and bonds along lattice vectors picked from them",
      body: <NeighbourhoodSection key={d.id} d={d} />,
    },
    canAnalyseAtomic(d) && {
      id: "pld", title: "Lattice displacements", section: "kemstem",
      tip: "Periodic lattice displacements: damp the superlattice peaks into a reference, then measure how far each column moved",
      body: <PldSection key={d.id} d={d} />,
    },
    hasPcaSection(d) && {
      id: "pca", title: "Decomposition and denoising",
      tip: d.origin === "pca"
        ? "Rebuild the spectrum image from its first components; makes a new spectrum image"
        : "Decompose over an energy window (HyperSpy): the components' maps and spectra",
      body: <PcaSection d={d} />,
    },
  ] as (Process | false)[]).filter((x): x is Process => !!x));

  if (only) {
    const k = processes.findIndex((p) => p.id === only);
    const p = processes[k];
    return (
      <div className="quantify process">
        <div className="process-group process-open">
          <SplitHeader id={only} title={p ? `${k + 1}. ${p.title}` : only} tip={p?.tip} on={d} pinned={pinned} />
          {p ? (
            <div className="process-group-body" onPointerDownCapture={() => followProcess(only)}>
              {p.body}
            </div>
          ) : (
            <p className="muted small">Not a process of {d.title}.</p>
          )}
        </div>
        {rebinOpen && <RebinDialog d={d} onClose={() => setRebinOpen(false)} />}
      </div>
    );
  }
  return (
    <div className="quantify process">
      {!aligning && (
        <section className="process-header">
          <span className="panel-tabs-spacer" />
          <ProcessModeToggle />
        </section>
      )}
      {mode === "grid" ? <ProcessGrid processes={processes} /> : <ProcessList processes={processes} />}
      {rebinOpen && <RebinDialog d={d} onClose={() => setRebinOpen(false)} />}
    </div>
  );
}
