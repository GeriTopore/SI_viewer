import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import type { Axis, DatasetSummary } from "../api/client";
import { formatValue } from "../format";
import { shapeBounds, useSi, type Roi, type Shape } from "../state/si";

type View = { key: string; width: number; height: number };

/** The region clicked or drawn last (of the view in ``useSi.regionView``), or null. */
export function useSelectedRegion() {
  const view = useSi((s) => s.regionView);
  const roi = useSi((s) => {
    const st = view ? s.bySi[view.key] : undefined;
    return st?.rois.find((r) => r.id === st.selected) ?? null;
  });
  return view && roi ? { view, roi } : null;
}

// Keys stay in the field (not the viewer's shortcuts, e.g. Delete).
const keys = (e: KeyboardEvent<HTMLInputElement>, done: (save: boolean) => void) => {
  e.stopPropagation();
  if (e.key === "Enter") done(true);
  else if (e.key === "Escape") done(false);
};

/** The region's name; a double-click turns it into a field (Enter or leaving it saves, Escape cancels). */
function EditableName({ roi, onRename }: { roi: Roi; onRename: (name: string) => void }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(roi.name);
  const ref = useRef<HTMLInputElement>(null);
  const finished = useRef(false);
  useEffect(() => {
    if (editing) ref.current?.select();
    finished.current = false;
  }, [editing]);
  const done = (save: boolean) => {
    if (finished.current) return;
    finished.current = true;
    if (save && value.trim() && value.trim() !== roi.name) onRename(value.trim());
    setEditing(false);
  };
  if (editing)
    return (
      <input ref={ref} className="rename-input" value={value} aria-label="Region name"
        onChange={(e) => setValue(e.target.value)} onBlur={() => done(true)} onKeyDown={(e) => keys(e, done)} />
    );
  return (
    <span className="editable-title" role="button" tabIndex={0} title="Double-click to rename"
      onDoubleClick={() => { setValue(roi.name); setEditing(true); }}
      onKeyDown={(e) => (e.key === "Enter" || e.key === "F2") && (setValue(roi.name), setEditing(true))}>
      {roi.name}
    </span>
  );
}

/** A table cell's number field: commits on Enter or leaving it; Escape reverts. */
function Cell({ value, onCommit, label }: { value: string; onCommit: (v: number) => void; label: string }) {
  const [text, setText] = useState(value);
  useEffect(() => setText(value), [value]);
  const cancelled = useRef(false);
  const done = (save: boolean) => {
    if (cancelled.current) {
      cancelled.current = false;
      return;
    }
    if (!save) cancelled.current = true; // the blur that follows
    const v = Number(text);
    if (save && text.trim() && Number.isFinite(v) && text !== value) onCommit(v);
    else setText(value);
  };
  return (
    <input className="cell-input" value={text} inputMode="decimal" aria-label={label}
      onChange={(e) => setText(e.target.value)} onBlur={() => done(true)}
      onKeyDown={(e) => keys(e, (save) => { done(save); e.currentTarget.blur(); })} />
  );
}

/** ``shape`` moved and resized so its bounds are ``[x, y, w, h]``. */
function placed(shape: Shape, [x, y, w, h]: number[]): Shape {
  if (shape.kind !== "polygon") return { ...shape, x0: x, y0: y, x1: x + w, y1: y + h };
  const [x0, y0, x1, y1] = shapeBounds(shape);
  const sx = x1 > x0 ? w / (x1 - x0) : 1;
  const sy = y1 > y0 ? h / (y1 - y0) : 1;
  return { ...shape, points: shape.points.map(([px, py]) => [x + (px - x0) * sx, y + (py - y0) * sy]) };
}

// Length units, in nm.
const LENGTHS: Record<string, number> = { pm: 1e-3, "Å": 0.1, nm: 1, "µm": 1e3, um: 1e3, mm: 1e6 };
const LENGTH_CHOICES = ["pm", "Å", "nm", "µm"];

/** The image's x and y axes in ``d`` (a spectrum image's scan, an image's or stack's frame). */
function planeAxes(d: DatasetSummary | null | undefined): [Axis, Axis] | null {
  if (!d) return null;
  const axes = d.kind === "spectrum_image" ? d.axes.filter((a) => a.navigate) : d.axes.filter((a) => !a.navigate);
  if (axes.length < 2) return null;
  const [y, x] = axes.slice(-2);
  return [x, y];
}

const KIND = { rect: "Rectangle", ellipse: "Ellipse", polygon: "Polygon" } as const;

/** The selected region, as the dataset's info: its name (double-click to
 * rename), type, and position and size in a table, in pixels or the image's
 * calibrated units. */
export function RegionInfo({ view, roi, dataset }: { view: View; roi: Roi; dataset?: DatasetSummary | null }) {
  const update = (patch: Partial<Roi>) => useSi.getState().updateRoi(view.key, roi.id, patch);
  const chosen = useSi((s) => s.regionUnits);
  const setUnits = useSi((s) => s.setRegionUnits);

  // Calibration: only when the dataset's frame is the region's image.
  const axes = planeAxes(dataset);
  const cal = axes && axes[0].size === view.width && axes[1].size === view.height && axes[0].units && axes[0].units !== "<undefined>"
    ? axes : null;
  const base = cal?.[0].units ?? "";
  const choices = ["px", ...(cal ? (base in LENGTHS ? LENGTH_CHOICES : [base]) : [])];
  const units = choices.includes(chosen) ? chosen : chosen in LENGTHS && base in LENGTHS ? chosen : "px";
  // Pixels → shown units, for axis 0 (x) or 1 (y); positions add the offset.
  const factor = cal && units !== "px" ? (base in LENGTHS ? LENGTHS[base] / LENGTHS[units] : 1) : 1;
  const toUnits = (px: number, axis: 0 | 1, position: boolean) =>
    !cal || units === "px" ? px : ((position ? cal[axis].offset : 0) + px * cal[axis].scale) * factor;
  const toPx = (v: number, axis: 0 | 1, position: boolean) =>
    !cal || units === "px" ? v : (v / factor - (position ? cal[axis].offset : 0)) / cal[axis].scale;

  const [x0, y0, x1, y1] = shapeBounds(roi.shape);
  const box = [x0, y0, x1 - x0, y1 - y0];
  const rows: { name: string; axis: 0 | 1; position: boolean }[] = [
    { name: "x", axis: 0, position: true },
    { name: "y", axis: 1, position: true },
    { name: "Width", axis: 0, position: false },
    { name: "Height", axis: 1, position: false },
  ];
  const set = (i: number, v: number) => {
    const b = [...box];
    b[i] = Math.round(toPx(v, rows[i].axis, rows[i].position));
    // Kept on the image, at least a pixel across.
    const [W, H] = [view.width, view.height];
    b[2] = Math.max(1, Math.min(b[2], W));
    b[3] = Math.max(1, Math.min(b[3], H));
    b[0] = Math.max(0, Math.min(b[0], W - b[2]));
    b[1] = Math.max(0, Math.min(b[1], H - b[3]));
    update({ shape: placed(roi.shape, b) });
  };
  const show = (v: number) => (units === "px" ? String(Math.round(v)) : formatValue(v));

  return (
    <div className="inspector">
      <dl className="props">
        <dt>Title</dt>
        <dd>
          <EditableName key={roi.id} roi={roi} onRename={(name) => update({ name })} />
        </dd>
        <dt>Type</dt>
        <dd>
          <span className="region-type">
            <span className="region-swatch" style={{ background: roi.color }} />
            {KIND[roi.shape.kind]} region
          </span>
        </dd>
        {roi.shape.kind === "polygon" && (
          <>
            <dt>Vertices</dt>
            <dd>{roi.shape.points.length}</dd>
          </>
        )}
      </dl>
      <div className="section-header">
        <h3>Dimensions</h3>
        {choices.length > 1 && (
          <select className="region-units" value={units} aria-label="Units" onChange={(e) => setUnits(e.target.value)}>
            {choices.map((u) => <option key={u} value={u}>{u}</option>)}
          </select>
        )}
      </div>
      <table className="axes">
        <thead>
          <tr>
            <th>Name</th>
            <th>Value</th>
            <th>Units</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.name}>
              <td>{r.name}</td>
              <td>
                <Cell label={r.name} value={show(toUnits(box[i], r.axis, r.position))} onCommit={(v) => set(i, v)} />
              </td>
              <td>{units}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
