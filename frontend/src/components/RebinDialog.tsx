import { useEffect, useState } from "react";
import type { DatasetSummary } from "../api/client";
import { useDatasets } from "../state/datasets";
import { formatValue } from "../format";

/** Indices (in ``d.axes``) of the 2D plane's y and x axes, if it has one. */
function planeAxes(d: DatasetSummary): [number, number] | null {
  const navigate = d.kind === "spectrum_image";
  if (d.kind !== "spectrum_image" && d.kind !== "image" && d.kind !== "stack") return null;
  const idx = d.axes.map((a, i) => [a, i] as const).filter(([a]) => a.navigate === navigate).map(([, i]) => i);
  return idx.length >= 2 ? [idx[idx.length - 2], idx[idx.length - 1]] : null;
}

const axisName = (d: DatasetSummary, i: number) => {
  const a = d.axes[i];
  if (a.name) return a.name;
  if (d.kind === "stack" && a.navigate) return "frames";
  return `axis ${i}`;
};

/** Dialog choosing an integer binning factor per axis (HyperSpy rebin). */
export function RebinDialog({ d, onClose }: { d: DatasetSummary; onClose: () => void }) {
  const plane = planeAxes(d);
  // Start with 2 × 2 on the image plane (or ×2 for a lone spectrum).
  const [factors, setFactors] = useState<number[]>(() =>
    d.axes.map((_, i) => (plane ? (plane.includes(i) ? 2 : 1) : d.axes.length === 1 ? 2 : 1)),
  );
  const [linked, setLinked] = useState(true);
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

  const unchanged = factors.every((f) => f === 1);
  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      await useDatasets.getState().rebin(d, factors);
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
        <div className="rebin-warning" role="note">
          This will overwrite the currently loaded data (the original file remains untouched).
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
