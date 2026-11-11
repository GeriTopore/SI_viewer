import { useEffect, useState } from "react";
import type { DatasetSummary } from "../api/client";
import { allDatasets, groupOf, planeSize, useDatasets } from "../state/datasets";
import { formatValue, isLineScan } from "../format";
import { InPlaceChoice, useInPlace } from "./InPlaceChoice";

/** Spectrum images are rebinned (a 2D one with everything on its scan, a
 * line scan along its positions or times and energy), and stacks (along
 * their frames, y and x). */
export const canRebin = (d: DatasetSummary) => d.kind === "spectrum_image" || d.kind === "stack";

/** Indices (in ``d.axes``) of the y and x axes: a scan's, or a stack's frame. */
function planeAxes(d: DatasetSummary): [number, number] | null {
  const inPlane = (a: DatasetSummary["axes"][number]) => (d.kind === "stack" ? !a.navigate : a.navigate);
  const idx = d.axes.map((a, i) => [a, i] as const).filter(([a]) => inPlane(a)).map(([, i]) => i);
  return idx.length >= 2 ? [idx[idx.length - 2], idx[idx.length - 1]] : null;
}

/** An axis' name; a stack's unnamed frame axis is "Frames". */
const axisName = (d: DatasetSummary, i: number) =>
  d.axes[i].name || (d.kind === "stack" && d.axes[i].navigate ? "Frames" : `axis ${i}`);

/** Dialog choosing an integer binning factor per axis of a spectrum image
 * (HyperSpy rebin). Images on the same scan are binned with it. An in-situ
 * series also has its time axis: each new frame sums that many frames. */
export function RebinDialog({ d, onClose }: { d: DatasetSummary; onClose: () => void }) {
  const plane = planeAxes(d);
  // Start with 2 × 2 on the scan (a line scan: 2 along it).
  const [factors, setFactors] = useState<number[]>(() => d.axes.map((a, i) => (plane ? plane.includes(i) : a.navigate) ? 2 : 1));
  // Loaded images and stacks of the same scan, binned with it (maps are remade).
  const files = useDatasets((s) => s.files);
  const [sw, sh] = planeSize(d);
  const onScan = allDatasets(files).filter(
    (x) =>
      d.kind === "spectrum_image" &&
      !isLineScan(d) &&
      x.file_id === d.file_id &&
      x.origin === "file" &&
      (x.kind === "image" || x.kind === "stack") &&
      planeSize(x)[0] === sw &&
      planeSize(x)[1] === sh,
  );
  // Linked EELS spectrum images get the same factors (energy too).
  const byId = new Map(allDatasets(files).map((x) => [x.id, x]));
  const group = groupOf(d, byId).filter((x) => x.id !== d.id);
  const [linked, setLinked] = useState(true);
  const series = d.insitu?.view ? d.insitu : null;
  const [time, setTime] = useState(1);
  // A series is binned as it is read: in place only.
  const [inPlaceChoice, setInPlace] = useInPlace("rebin");
  const inPlace = series ? true : inPlaceChoice;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && !busy && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [busy, onClose]);

  const setFactor = (i: number, raw: string) => {
    const f = Math.max(1, Math.min(d.axes[i].size, Math.round(Number(raw) || 1)));
    setFactors((cur) =>
      cur.map((v, j) => (j === i || (linked && plane?.includes(i) && plane.includes(j)) ? f : v)),
    );
  };

  const unchanged = factors.every((f) => f === 1) && time === 1;
  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      await useDatasets.getState().rebin(d, factors, time, inPlace);
      onClose();
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  };

  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && !busy && onClose()}>
      <div className="modal rebin-dialog" role="dialog" aria-label={`Rebin ${d.title}`}>
        <h2>Rebin</h2>
        <p className="muted small rebin-subtitle">{d.title}</p>
        <table className="rebin-table">
          <thead>
            <tr>
              <th>Axis</th>
              <th>Size</th>
              <th>Factor</th>
              <th>New size</th>
              <th>New scale</th>
            </tr>
          </thead>
          <tbody>
            {series && (
              <tr>
                <td>Time</td>
                <td>{series.frames}</td>
                <td>
                  <input
                    type="number"
                    min={1}
                    max={series.frames}
                    step={1}
                    value={time}
                    onChange={(e) => setTime(Math.max(1, Math.min(series.frames, Math.round(Number(e.target.value) || 1))))}
                    disabled={busy}
                    aria-label="Binning factor for Time"
                  />
                </td>
                <td>
                  {Math.floor(series.frames / time)}
                  {series.frames % time > 0 && <span className="muted"> (crops {series.frames % time})</span>}
                </td>
                <td>{series.frame_step_s ? `${formatValue(series.frame_step_s * time)} s` : "–"}</td>
              </tr>
            )}
            {d.axes.map((a, i) => {
              const f = factors[i];
              const size = Math.floor(a.size / f);
              const cropped = a.size - size * f;
              return (
                <tr key={i}>
                  <td>{axisName(d, i)}</td>
                  <td>{a.size}</td>
                  <td>
                    <input
                      type="number"
                      min={1}
                      max={a.size}
                      step={1}
                      value={f}
                      onChange={(e) => setFactor(i, e.target.value)}
                      disabled={busy}
                      aria-label={`Binning factor for ${axisName(d, i)}`}
                    />
                  </td>
                  <td>
                    {size}
                    {cropped > 0 && <span className="muted"> (crops {cropped})</span>}
                  </td>
                  <td>
                    {formatValue(a.scale * f)} {a.units}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {plane && (
          <label className="rebin-link">
            <input type="checkbox" checked={linked} onChange={(e) => setLinked(e.target.checked)} />
            Same factor for x and y
          </label>
        )}
        {!series && (
          <div className="rebin-in-place">
            <InPlaceChoice inPlace={inPlace} onChange={setInPlace} disabled={busy} what={d.title} />
          </div>
        )}
        {inPlace && group.length + onScan.length > 0 && (
          <p className="muted small rebin-also">
            Also binned: {[...group.map((x) => `${x.title} (linked)`), ...onScan.map((x) => x.title)].join(", ")}
          </p>
        )}
        <div className="rebin-warning" role="note" hidden={!inPlace}>
          {series
            ? `Every frame is binned from now on${series.view?.mode !== "frame" ? "; the frame range becomes one frame" : ""}; the files are unchanged.`
            : `Overwrites the loaded data${d.alignment ? " and removes its drift correction" : ""}; the file is unchanged.`}
        </div>
        {error && <p className="field-error">{error}</p>}
        <div className="rebin-actions">
          <button className="secondary" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button className="primary" onClick={submit} disabled={busy || unchanged}>
            {busy ? "Rebinning…" : "Rebin"}
          </button>
        </div>
      </div>
    </div>
  );
}
