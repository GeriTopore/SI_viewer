import { useEffect, useState } from "react";
import { api, type DatasetSummary } from "../api/client";
import { useDatasets, useFocusedDataset } from "../state/datasets";
import { bytesLabel, filterLabel, kindLabel, shapeLabel } from "../format";

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
          {kindLabel(d)}
          {d.signal_type && <span className="muted"> · {d.signal_type}</span>}
        </dd>
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
    </div>
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
