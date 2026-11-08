import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from "react";
import { api, exportUrl, mapsZipUrl, type DatasetSummary, type Technique } from "../api/client";
import { downloadUrl, useExportFormats } from "../download";
import { allDatasets, useDatasets, useFocusedDataset } from "../state/datasets";
import { useLayers } from "../state/layers";
import { TECHNIQUE_SOURCE_NOTE, bytesLabel, filterLabel, hasTechnique, isLineScan, kindLabel, shapeLabel } from "../format";

export function Inspector() {
  const d = useFocusedDataset();
  if (!d) return <p className="muted">Select a dataset to see its details.</p>;

  return (
    <div className="inspector">
      <dl className="props">
        <dt>Title</dt>
        <dd>{d.title}</dd>
        <dt>Type</dt>
        <dd>
          {isLineScan(d) ? "Line scan" : d.kind === "spectrum_image" ? "Spectrum image" : d.kind === "spectrum" ? "Spectrum" : kindLabel(d)}
        </dd>
        {hasTechnique(d) && (
          <>
            <dt>Signal</dt>
            <dd>
              <TechniqueControl d={d} />
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
        {filterLabel(d.filters) && (
          <>
            <dt>Filtered</dt>
            <dd title="Applied when the data is shown (Process tab); the loaded data is unchanged">
              {filterLabel(d.filters)}
            </dd>
          </>
        )}
      </dl>
      <AxesTable key={d.id} d={d} />
      <SaveButtons d={d} />
    </div>
  );
}

/** The signal type as text, with where it came from, and a "Change…" popover. */
function TechniqueControl({ d }: { d: DatasetSummary }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  const note = d.technique_source ? TECHNIQUE_SOURCE_NOTE[d.technique_source] : undefined;

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
      {note && d.technique && <span className="muted small">{note}</span>}
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

const TECHNIQUE_OPTIONS: [Technique | null, string][] = [["EDS", "EDS"], ["EELS", "EELS"], [null, "Unknown"]];
const TOOLS_OF: Record<string, string> = {
  EDS: "X-ray lines, element maps and Quantify",
  EELS: "the EELS tools",
};

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
      // A changed signal type can unlink a low/core loss pair (the partner's file too).
      if (d.eels?.partner_id) await useDatasets.getState().syncFiles();
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
      <p className="muted small">
        The tools offered change with the type
        {choice && ` (${choice}: ${TOOLS_OF[choice]})`}.
      </p>
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

/** Buttons saving the dataset in each format it supports. */
function SaveButtons({ d }: { d: DatasetSummary }) {
  const formats = useExportFormats()[d.kind] ?? [];
  if (!formats.length) return null;
  return (
    <section className="inspector-save">
      <h3>Save</h3>
      <p className="muted small">
        {d.kind === "spectrum_image"
          ? "Saved as loaded (rebinned, if it was); its processing settings are recorded in the metadata."
          : "Saved as shown, with any filters applied."}
      </p>
      <div className="inspector-save-buttons">
        {formats.map((f) => (
          <button key={f.format} onClick={() => downloadUrl(exportUrl(d.id, f.format))} title={`Save as ${f.label}`}>
            {f.label.split(" (")[0]}
          </button>
        ))}
        {d.is_element_map && (
          <button onClick={() => downloadUrl(mapsZipUrl(d.file_id))} title="Every element map of this file as TIFFs in a zip">
            All maps (.zip)
          </button>
        )}
      </div>
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
