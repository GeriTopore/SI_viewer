import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from "react";
import { api, type DatasetSummary, type EelsRole, type Technique } from "../api/client";
import { useExportFormats } from "../download";
import { allDatasets, useDatasets, useFocusedDataset } from "../state/datasets";
import { useLayers } from "../state/layers";
import { EditableTitle } from "./RenameInput";
import { ExportDialog } from "./ExportDialog";
import { bytesLabel, filterLabel, formatValue, hasTechnique, isLineScan, isThickness, isZlpShift, kindLabel, shapeLabel } from "../format";

export function Inspector() {
  const d = useFocusedDataset();
  if (!d) return <p className="muted">Select a dataset to see its details.</p>;

  return (
    <div className="inspector">
      <dl className="props">
        <dt>Title</dt>
        <dd>
          <EditableTitle key={d.id} d={d} />
        </dd>
        <dt>Type</dt>
        <dd>
          {isThickness(d)
            ? d.kind === "spectrum" ? "Thickness profile" : "Thickness map"
            : isZlpShift(d)
              ? d.kind === "spectrum" ? "ZLP shift profile" : "ZLP shift map"
              : isLineScan(d) ? "LineScan" : d.kind === "spectrum_image" ? "Spectrum image" : d.kind === "spectrum" ? "Spectrum" : kindLabel(d)}
        </dd>
        {hasTechnique(d) && (
          <>
            <dt>Signal</dt>
            <dd>
              <TechniqueControl d={d} />
            </dd>
          </>
        )}
        {d.eels && (
          <>
            <dt>Loss</dt>
            <dd>
              <LossControl d={d} />
            </dd>
          </>
        )}
        <dt>Shape</dt>
        <dd>{shapeLabel(d)}</dd>
        <dt>Data</dt>
        <dd>
          {d.dtype}, {bytesLabel(d.nbytes)}
          <span className="muted"> · {d.lazy ? "on disk" : "in memory"}</span>
        </dd>
        {d.rebinned.length > 0 && (
          <>
            <dt>Rebinned</dt>
            <dd title="Applied to the loaded data; the file on disk is unchanged">{d.rebinned.join("; then ")}</dd>
          </>
        )}
        {d.zlp_aligned.length > 0 && (
          <>
            <dt>ZLP aligned</dt>
            <dd title="Applied to the loaded data; the file on disk is unchanged">{d.zlp_aligned.join("; then ")}</dd>
          </>
        )}
        {filterLabel(d.filters) && (
          <>
            <dt>Filtered</dt>
            <dd title="Applied when the data is shown (Process tab); the loaded data is unchanged">
              {filterLabel(d.filters)}
            </dd>
          </>
        )}
      </dl>
      {d.insitu?.view && <SeriesInfo d={d} />}
      <AxesTable key={d.id} d={d} />
      <SaveButtons d={d} />
    </div>
  );
}

/** The in-situ series behind a spectrum image: its frames, times and files. */
function SeriesInfo({ d }: { d: DatasetSummary }) {
  const s = d.insitu!;
  const [h, w] = s.frame_shape;
  const channels = d.signal_shape.join("×");
  const duration = s.duration_s ?? s.last_time_s - s.first_time_s;
  return (
    <>
      <h3 className="dl-section-title">In-situ series</h3>
      <dl className="props">
        <dt>Frames</dt>
        <dd>
          {s.frames}
          {s.recorded_frames != null && s.recorded_frames !== s.frames && ` of ${s.recorded_frames} recorded`}
        </dd>
        {s.note && (
          <>
            <dt>Note</dt>
            <dd className="field-error">{s.note}</dd>
          </>
        )}
        <dt>Time span</dt>
        <dd title={s.time_source === "recorded" ? "Time each frame was finished" : `Times from the ${s.time_source}`}>
          {formatValue(s.first_time_s)}–{formatValue(s.last_time_s)} s
        </dd>
        {s.frame_step_s != null && (
          <>
            <dt>Frame time</dt>
            <dd>{formatValue(s.frame_step_s)} s</dd>
          </>
        )}
        <dt>Duration</dt>
        <dd>{formatValue(duration)} s</dd>
        {s.start_time && (
          <>
            <dt>Started</dt>
            <dd>{s.start_time}</dd>
          </>
        )}
        <dt>Frame size</dt>
        <dd>
          {w}×{h} px{channels && ` · ${channels} ch`}
          {s.frame_bytes != null && ` · ${bytesLabel(s.frame_bytes)}`}
        </dd>
        {s.raw_bytes != null && (
          <>
            <dt>Raw files</dt>
            <dd title={s.raw_files?.join(", ")}>
              {s.raw_files?.length ?? 0} · {bytesLabel(s.raw_bytes)}
            </dd>
          </>
        )}
      </dl>
    </>
  );
}

/** The signal type as text, with where it came from, and a "Change…" popover. */
function TechniqueControl({ d }: { d: DatasetSummary }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => setOpen(false), [d.id]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    const close = () => setOpen(false);
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    window.addEventListener("resize", close);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", close);
    };
  }, [open]);

  return (
    <span className="technique-control" ref={ref}>
      <span>{d.technique ?? "Unknown"}</span>
      <button
        className="link-button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        title="Change the signal type"
      >
        Change…
      </button>
      {open && <TechniquePopover d={d} anchor={ref} onClose={() => setOpen(false)} />}
    </span>
  );
}

const LOSS_OPTIONS: [EelsRole, string, string][] = [
  ["low_loss", "LL", "Low loss"],
  ["core_loss", "HL", "Core (high) loss"],
];

/** Low or core loss of an EELS spectrum image, with where that came from;
 * a linked group takes one low loss at most (the backend refuses a second). */
function LossControl({ d }: { d: DatasetSummary }) {
  const [busy, setBusy] = useState(false);
  const role = d.eels!.role;
  const set = async (r: EelsRole) => {
    if (r === role) return;
    setBusy(true);
    try {
      await useDatasets.getState().setEelsRole(d, r);
    } catch (e) {
      useLayers.getState().notify(`Could not change ${d.title}: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };
  return (
    <span className="technique-control">
      <span className="segmented" role="group" aria-label="Low or core loss">
        {LOSS_OPTIONS.map(([r, label, title]) => (
          <button
            key={r}
            className={role === r ? "active" : undefined}
            aria-pressed={role === r}
            disabled={busy}
            onClick={() => set(r)}
            title={title}
          >
            {label}
          </button>
        ))}
      </span>
    </span>
  );
}

const TECHNIQUE_OPTIONS: [Technique | null, string][] = [["EDS", "EDS"], ["EELS", "EELS"], [null, "Unknown"]];

/** Choose another signal type; nothing is sent until "Change". */
function TechniquePopover({
  d,
  anchor,
  onClose,
}: {
  d: DatasetSummary;
  anchor: RefObject<HTMLElement>;
  onClose: () => void;
}) {
  const replaceFile = useDatasets((s) => s.replaceFile);
  const files = useDatasets((s) => s.files);
  const [choice, setChoice] = useState<Technique | null>(d.technique);
  const [busy, setBusy] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);

  // Below the "Change…" button, kept inside the window.
  useLayoutEffect(() => {
    const a = anchor.current!.getBoundingClientRect();
    const r = boxRef.current!.getBoundingClientRect();
    setPos({
      left: Math.max(4, Math.min(a.left, window.innerWidth - r.width - 4)),
      top: Math.max(4, Math.min(a.bottom + 4, window.innerHeight - r.height - 4)),
    });
  }, [anchor]);

  const leavingEds = d.technique === "EDS" && choice !== "EDS";
  const maps = leavingEds
    ? allDatasets(files).filter((x) => x.file_id === d.file_id && x.parent_id === d.id && x.origin !== "file").length
    : 0;
  const elements = leavingEds ? (d.eds?.elements.length ?? 0) : 0;
  const changed = choice !== d.technique;

  const apply = async () => {
    setBusy(true);
    try {
      replaceFile((await api.setTechnique(d.id, choice)).file);
      // A changed signal type can unlink a spectrum image from its group (the others' files too).
      if (d.eels?.linked.length) await useDatasets.getState().syncFiles();
      onClose();
    } catch (e) {
      useLayers.getState().notify(`Could not set the signal type: ${(e as Error).message}`);
      setBusy(false);
    }
  };

  return (
    <div
      className="technique-popover"
      ref={boxRef}
      role="dialog"
      aria-label="Change the signal type"
      style={pos ?? { visibility: "hidden" }}
    >
      <div className="technique-options" role="radiogroup" aria-label="Signal type">
        {TECHNIQUE_OPTIONS.map(([value, label]) => (
          <label key={label}>
            <input type="radio" name="technique" checked={choice === value} onChange={() => setChoice(value)} />
            {label}
          </label>
        ))}
      </div>
      {maps > 0 && (
        <p className="field-error">
          Its {maps} computed and fitted {maps === 1 ? "map" : "maps"} will be deleted.
        </p>
      )}
      {elements > 0 && <p className="muted small">Its elements and fits won't be used.</p>}
      <div className="technique-actions">
        <button className="link-button" onClick={onClose} disabled={busy}>
          Cancel
        </button>
        <button className="secondary small-button" onClick={apply} disabled={!changed || busy}>
          Change
        </button>
      </div>
    </div>
  );
}

/** Export… opens the dialog saving the dataset in a format it supports. */
function SaveButtons({ d }: { d: DatasetSummary }) {
  const formats = useExportFormats()[d.kind] ?? [];
  const [open, setOpen] = useState(false);
  if (!formats.length) return null;
  return (
    <section className="inspector-save">
      <h3>Save</h3>
      <div className="inspector-save-buttons">
        <button onClick={() => setOpen(true)}>Export…</button>
      </div>
      {open && <ExportDialog datasets={[d]} onClose={() => setOpen(false)} />}
    </section>
  );
}

interface Draft {
  scale: string;
  offset: string;
  units: string;
}

/** Axis calibrations, editable (e.g. to set the pixel size of a bare TIFF). */
function AxesTable({ d }: { d: DatasetSummary }) {
  const updateDataset = useDatasets((s) => s.updateDataset);
  const [editing, setEditing] = useState(false);
  const [drafts, setDrafts] = useState<Draft[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (editing) {
      setDrafts(d.axes.map((a) => ({ scale: String(a.scale), offset: String(a.offset), units: a.units })));
      setError(null);
    }
  }, [editing, d.axes]);

  const setDraft = (i: number, patch: Partial<Draft>) =>
    setDrafts((ds) => ds.map((x, j) => (j === i ? { ...x, ...patch } : x)));

  const save = async () => {
    const body = drafts.map((x, index) => ({
      index,
      scale: Number(x.scale),
      offset: Number(x.offset),
      units: x.units.trim(),
    }));
    if (body.some((b) => !(b.scale > 0) || !Number.isFinite(b.offset))) {
      setError("Scale must be a positive number and offset a number.");
      return;
    }
    try {
      updateDataset(await api.calibrate(d.id, body));
      setEditing(false);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  // The x/y partner of a 2D plane's axis (both navigation or both signal axes).
  const partnerOf = (i: number) => {
    const plane = d.axes.map((_, j) => j).filter((j) => d.axes[j].navigate === d.axes[i].navigate).slice(-2);
    return plane.length === 2 && plane.includes(i) ? plane.find((j) => j !== i)! : null;
  };

  // Square pixels are the norm: while x and y match, editing one edits both.
  const setLinked = (i: number, key: "scale" | "units", value: string) => {
    const partner = partnerOf(i);
    setDrafts((ds) =>
      ds.map((x, j) =>
        j === i || (j === partner && x[key] === ds[i][key]) ? { ...x, [key]: value } : x,
      ),
    );
  };

  return (
    <>
      <div className="section-header">
        <h3>Axes</h3>
        {editing ? (
          <span>
            <button className="link-button" onClick={() => setEditing(false)}>
              Cancel
            </button>
            <button className="secondary small-button" onClick={save}>
              Save
            </button>
          </span>
        ) : (
          <button className="link-button" onClick={() => setEditing(true)} title="Set axis calibration">
            Edit calibration
          </button>
        )}
      </div>
      {error && <p className="field-error">{error}</p>}
      <table className="axes">
        <thead>
          <tr>
            <th>Name</th>
            <th>Size</th>
            <th>Scale</th>
            <th>Offset</th>
            <th>Units</th>
          </tr>
        </thead>
        <tbody>
          {d.axes.map((ax, i) => (
            <tr key={i} className={ax.navigate ? "nav-axis" : undefined}>
              <td>{ax.name || <span className="muted">axis {i}</span>}</td>
              <td>{ax.size}</td>
              {editing && drafts[i] ? (
                <>
                  <td>
                    <input
                      className="cell-input"
                      value={drafts[i].scale}
                      onChange={(e) => setLinked(i, "scale", e.target.value)}
                      inputMode="decimal"
                    />
                  </td>
                  <td>
                    <input
                      className="cell-input"
                      value={drafts[i].offset}
                      onChange={(e) => setDraft(i, { offset: e.target.value })}
                      inputMode="decimal"
                    />
                  </td>
                  <td>
                    <input
                      className="cell-input"
                      value={drafts[i].units}
                      onChange={(e) => setLinked(i, "units", e.target.value)}
                      placeholder="nm"
                    />
                  </td>
                </>
              ) : (
                <>
                  <td>{ax.scale.toPrecision(4)}</td>
                  <td>{ax.offset.toPrecision(4)}</td>
                  <td>{ax.units}</td>
                </>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
