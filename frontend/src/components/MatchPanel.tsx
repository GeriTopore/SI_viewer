import { useEffect, useMemo, useState } from "react";
import { api, type MatchResult, type TemplatesIpf } from "../api/client";
import { colormapGradient, colormapLut } from "../viewer/colormaps";
import { formatValue } from "../format";
import { pickKey, useOrientation } from "../state/orientation";
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
 * Euler angles. Under it, every template matched at the active cursor, best
 * first: picking one draws it on the patterns instead of the best. */
export function MatchPanel({ templatesId, cursors, activeId, at, busy, error, matchKey }: {
  templatesId: string;
  cursors: CursorLike[];
  activeId: string;
  at: Record<string, MatchResult>;
  /** Matching now (a cursor without a result yet may get one). */
  busy: boolean;
  error: string | null;
  /** The matches' key in ``useOrientation.matchCache`` (picks are kept by it). */
  matchKey: string;
}) {
  const active = cursors.find((c) => c.id === activeId) ?? cursors[0];
  const m = active ? at[`${active.x},${active.y}`] : undefined;
  // The IPF of the phase matched at the active cursor (several phases: its own).
  const shown = m?.templates ?? templatesId.split(",")[0];
  const phased = templatesId.includes(",");
  const [ipf, setIpf] = useState<{ id: string; ipf: TemplatesIpf } | null>(null);
  useEffect(() => {
    let stale = false;
    api.templatesIpf(shown).then((r) => !stale && setIpf({ id: shown, ipf: r }), () => undefined);
    return () => {
      stale = true;
    };
  }, [shown]);
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
  const ids = templatesId.split(",");
  // The match shown at each position (its rank in the ranked list; 0 the best).
  const picks = useOrientation((s) => s.matchPick);
  const rankAt = (x: number, y: number) => picks[pickKey(matchKey, `${x},${y}`)] ?? 0;
  const rings = cursors.flatMap((c) => {
    const r = at[`${c.x},${c.y}`];
    const row = r?.ranked?.[rankAt(c.x, c.y)];
    const id = row ? ids[row.phase] : r?.templates ?? shown;
    return r && id === shown ? [{ index: row?.index ?? r.best.index, colour: c.color }] : [];
  });
  const activeRank = active ? rankAt(active.x, active.y) : 0;

  return (
    <div className="templates-tab">
      {error && <p className="field-error">{error}</p>}
      <div className="templates-panes match-panes">
        <figure>
          {ipf?.id === shown && m ? (
            <>
              <TemplatesIpfPlot ipf={ipf.ipf} picked={null} colours={colours} rings={rings} />
              <figcaption className="small match-key">
                <span className="muted">{m.phase ? `${m.phase} · ` : ""}Correlation at ({active.x}, {active.y})</span>
                <span className="match-scale" style={{ background: colormapGradient("inferno") }} />
              </figcaption>
            </>
          ) : (
            <p className="muted small">{error ? "" : busy ? "Matching…" : `Not matched at (${active?.x}, ${active?.y})`}</p>
          )}
        </figure>
        <figure>
          <table className="axes match-table">
            <thead>
              <tr>
                <th />
                <th>Cursor</th>
                {phased && <th>Phase</th>}
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
                    {phased && (
                      <td title={r?.phases?.map((p) => `${p.name}: ${formatValue(p.correlation)}`).join(" · ")}>{r?.phase ?? ""}</td>
                    )}
                    <td>{r ? uvwLabel(r.best.uvw) : busy ? "…" : "–"}</td>
                    <td>{r ? formatValue(r.best.correlation) : ""}</td>
                    <td>{r ? `${formatValue(r.best.rotation)}°${r.best.mirror ? " (mirrored)" : ""}` : ""}</td>
                    <td>{r ? r.best.euler.map((a) => `${formatValue(a)}°`).join(", ") : ""}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {m?.ranked && m.ranked.length > 0 && (
            <details className="match-ranked">
              <summary className="small muted">All matches at ({active.x}, {active.y}) · {m.ranked.length}</summary>
              <div className="match-ranked-scroll">
                <table className="axes match-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      {phased && <th>Phase</th>}
                      <th>[uvw]</th>
                      <th>Corr.</th>
                      <th>In-plane</th>
                    </tr>
                  </thead>
                  <tbody>
                    {m.ranked.map((row, i) => (
                      <tr key={i} className={i === activeRank ? "selected" : undefined}
                        onClick={() => void useOrientation.getState().pickMatch(matchKey, `${active.x},${active.y}`, i, templatesId)}>
                        <td>{i + 1}</td>
                        {phased && <td>{m.phases?.[row.phase]?.name ?? ""}</td>}
                        <td>{uvwLabel(row.uvw)}</td>
                        <td>{formatValue(row.correlation)}</td>
                        <td>{`${formatValue(row.rotation)}°${row.mirror ? " (mirrored)" : ""}`}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}
        </figure>
      </div>
    </div>
  );
}
