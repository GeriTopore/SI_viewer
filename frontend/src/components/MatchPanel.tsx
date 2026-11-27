import { useEffect, useMemo, useState } from "react";
import { api, type MatchResult, type TemplatesIpf } from "../api/client";
import { colormapGradient, colormapLut } from "../viewer/colormaps";
import { formatValue } from "../format";
import { TemplatesIpfPlot } from "./OrixPlots";

interface CursorLike {
  id: string;
  x: number;
  y: number;
  color: string;
}

const uvwLabel = (uvw: number[]) => `[${uvw.map((x) => formatValue(x)).join(" ")}]`;

/** pyxem's template matching at the cursors (the pattern's Match tab): the
 * correlation of every template at the active cursor on the inverse pole
 * figure (inferno, as the tutorial's heat map), each cursor's best ringed
 * in its colour; each cursor's best: [uvw], correlation, in-plane rotation,
 * Euler angles. */
export function MatchPanel({ templatesId, cursors, activeId, at, error }: {
  templatesId: string;
  cursors: CursorLike[];
  activeId: string;
  at: Record<string, MatchResult>;
  error: string | null;
}) {
  const [ipf, setIpf] = useState<{ id: string; ipf: TemplatesIpf } | null>(null);
  useEffect(() => {
    let stale = false;
    api.templatesIpf(templatesId).then((r) => !stale && setIpf({ id: templatesId, ipf: r }), () => undefined);
    return () => {
      stale = true;
    };
  }, [templatesId]);
  const active = cursors.find((c) => c.id === activeId) ?? cursors[0];
  const m = active ? at[`${active.x},${active.y}`] : undefined;
  const lut = colormapLut("inferno");
  const colours = useMemo(() => {
    if (!m) return undefined;
    const vals = m.correlations.filter((c): c is number => c !== null);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    return m.correlations.map((c) => {
      if (c === null) return "rgba(128,128,128,0.25)";
      const k = Math.round(((c - lo) / (hi - lo || 1)) * 255) * 4;
      return `rgb(${lut[k]},${lut[k + 1]},${lut[k + 2]})`;
    });
  }, [m, lut]);
  const rings = cursors.flatMap((c) => {
    const r = at[`${c.x},${c.y}`];
    return r ? [{ index: r.best.index, colour: c.color }] : [];
  });

  return (
    <div className="templates-tab">
      {error && <p className="field-error">{error}</p>}
      <div className="templates-panes">
        <figure>
          {ipf?.id === templatesId && m ? (
            <>
              <TemplatesIpfPlot ipf={ipf.ipf} picked={null} colours={colours} rings={rings} />
              <figcaption className="small match-key">
                <span className="muted">Correlation at ({active.x}, {active.y})</span>
                <span className="match-scale" style={{ background: colormapGradient("inferno") }} />
              </figcaption>
            </>
          ) : (
            <p className="muted small">{error ? "" : "Matching…"}</p>
          )}
        </figure>
        <figure>
          <table className="axes match-table">
            <thead>
              <tr>
                <th />
                <th>Cursor</th>
                <th>[uvw]</th>
                <th>Corr.</th>
                <th>In-plane</th>
                <th>Euler (φ₁, Φ, φ₂)</th>
              </tr>
            </thead>
            <tbody>
              {cursors.map((c) => {
                const r = at[`${c.x},${c.y}`];
                return (
                  <tr key={c.id} className={c.id === active?.id ? "selected" : undefined}>
                    <td><span className="match-dot" style={{ background: c.color }} /></td>
                    <td>({c.x}, {c.y})</td>
                    <td>{r ? uvwLabel(r.best.uvw) : "…"}</td>
                    <td>{r ? formatValue(r.best.correlation) : ""}</td>
                    <td>{r ? `${formatValue(r.best.rotation)}°${r.best.mirror ? " (mirrored)" : ""}` : ""}</td>
                    <td>{r ? r.best.euler.map((a) => `${formatValue(a)}°`).join(", ") : ""}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </figure>
      </div>
    </div>
  );
}
