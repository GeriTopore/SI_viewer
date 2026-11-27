import { useId, type MouseEvent } from "react";
import type { IpfSector, StructurePlots, TemplatePattern, TemplatesIpf } from "../api/client";
import { formatValue } from "../format";

/** orix's plots drawn by the app: orix gives the stereographic coordinates
 * (y up), colour keys and sectors; these draw them as SVG. */

const PAD = 0.06;

/** SVG y down from orix's y up. */
const fy = (y: number) => -y;

/** The sector's box with a margin for the corner labels. */
function sectorBox(s: IpfSector): [number, number, number, number] {
  const xs = s.outline.x, ys = s.outline.y;
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const m = PAD * Math.max(x1 - x0, y1 - y0);
  // Wider at the sides: the corners' labels are centred on them.
  return [x0 - 2.6 * m, fy(y1) - m * 1.6, x1 - x0 + 5.2 * m, y1 - y0 + 3.2 * m];
}

const outlinePath = (s: IpfSector) =>
  s.outline.x.map((x, i) => `${i ? "L" : "M"}${x},${fy(s.outline.y[i])}`).join(" ") + " Z";

function SectorFrame({ sector, keyOpacity, clip }: { sector: IpfSector; keyOpacity: number; clip: string }) {
  const [x0, x1, y0, y1] = sector.extent;
  const unit = Math.max(x1 - x0, y1 - y0);
  return (
    <>
      <defs>
        <clipPath id={clip}>
          <path d={outlinePath(sector)} />
        </clipPath>
      </defs>
      {keyOpacity > 0 && (
        <image href={sector.key} x={x0} y={fy(y1)} width={x1 - x0} height={y1 - y0} preserveAspectRatio="none"
          clipPath={`url(#${clip})`} opacity={keyOpacity} />
      )}
      <path d={outlinePath(sector)} fill="none" stroke="var(--text)" strokeWidth={unit * 0.006} />
      {sector.corners.map((c) => (
        <text key={c.label} x={c.x} y={fy(c.y) + (c.y > (y0 + y1) / 2 ? -unit * 0.035 : unit * 0.075)}
          textAnchor="middle" fontSize={unit * 0.06} fill="var(--text)">[{c.label}]</text>
      ))}
    </>
  );
}

/** orix's inverse pole figure colour key, clipped to the sector. */
export function IpfKey({ sector }: { sector: IpfSector }) {
  const clip = useId().replace(/:/g, "");
  return (
    <svg className="orix-plot" viewBox={sectorBox(sector).join(" ")} role="img" aria-label="IPF colour key">
      <SectorFrame sector={sector} keyOpacity={1} clip={clip} />
    </svg>
  );
}

/** The templates' beam directions in the inverse pole figure (as pyxem's IPF
 * markers), each in its IPF colour; click picks the nearest. */
export function TemplatesIpfPlot({ ipf, picked, onPick, colours, rings = [] }: {
  ipf: TemplatesIpf;
  picked: number | null;
  onPick?: (i: number) => void;
  /** Each template's colour instead of its IPF colour (e.g. a correlation). */
  colours?: string[];
  /** Templates ringed in a colour (e.g. each cursor's best match). */
  rings?: { index: number; colour: string }[];
}) {
  const clip = useId().replace(/:/g, "");
  const { sector, points } = ipf;
  const box = sectorBox(sector);
  const unit = Math.max(sector.extent[1] - sector.extent[0], sector.extent[3] - sector.extent[2]);
  const r = unit * 0.008;
  const click = (e: MouseEvent<SVGSVGElement>) => {
    const svg = e.currentTarget;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    let best = -1, d = Infinity;
    points.x.forEach((x, i) => {
      const dd = (x - p.x) ** 2 + (fy(points.y[i]) - p.y) ** 2;
      if (dd < d) [best, d] = [i, dd];
    });
    if (best >= 0) onPick?.(best);
  };
  return (
    <svg className={`orix-plot${onPick ? " orix-clickable" : ""}`} viewBox={box.join(" ")} onClick={onPick ? click : undefined} role="img" aria-label="Templates in the inverse pole figure">
      <SectorFrame sector={sector} keyOpacity={colours ? 0 : 0.18} clip={clip} />
      {points.x.map((x, i) => (
        <circle key={i} cx={x} cy={fy(points.y[i])} r={r} fill={colours?.[i] ?? `rgb(${points.rgb[i].join(",")})`} />
      ))}
      {rings.filter((g) => g.index < points.x.length).map((g, k) => (
        <circle key={`ring${k}`} cx={points.x[g.index]} cy={fy(points.y[g.index])} r={r * (3.2 + 1.4 * k)} fill="none" stroke={g.colour} strokeWidth={r * 0.9} />
      ))}
      {picked !== null && picked < points.x.length && (
        <circle cx={points.x[picked]} cy={fy(points.y[picked])} r={r * 3.2} fill="none" stroke="var(--text)" strokeWidth={r * 0.9} />
      )}
    </svg>
  );
}

/** orix's stereographic projection of a point group (``Symmetry.plot``):
 * upper-hemisphere poles as dots, lower ones reprojected as crosses. */
export function SymmetryPlot({ symmetry }: { symmetry: StructurePlots["symmetry"] }) {
  const s = 0.045;
  return (
    <svg className="orix-plot" viewBox="-1.2 -1.2 2.4 2.4" role="img" aria-label={`Point group ${symmetry.name}`}>
      <circle cx={0} cy={0} r={1} fill="none" stroke="var(--text)" strokeWidth={0.012} />
      <line x1={-1} x2={1} y1={0} y2={0} stroke="var(--border)" strokeWidth={0.006} />
      <line x1={0} x2={0} y1={-1} y2={1} stroke="var(--border)" strokeWidth={0.006} />
      <text x={1.06} y={0.03} fontSize={0.09} fill="var(--muted)">e₁</text>
      <text x={-0.03} y={-1.06} fontSize={0.09} fill="var(--muted)">e₂</text>
      {symmetry.upper.x.map((x, i) => <circle key={`u${i}`} cx={x} cy={fy(symmetry.upper.y[i])} r={s * 0.6} fill="var(--accent)" />)}
      {symmetry.lower.x.map((x, i) => {
        const y = fy(symmetry.lower.y[i]);
        return (
          <g key={`l${i}`} stroke="var(--accent)" strokeWidth={0.014}>
            <line x1={x - s} x2={x + s} y1={y} y2={y} />
            <line x1={x} x2={x} y1={y - s} y2={y + s} />
          </g>
        );
      })}
    </svg>
  );
}

/** One template's reflections out to its reciprocal radius (1/Å), each
 * sized by its intensity; hkl on hover. */
export function TemplatePatternPlot({ t }: { t: TemplatePattern }) {
  const R = t.reciprocal_radius;
  const m = R * 1.08;
  return (
    <svg className="orix-plot template-pattern" viewBox={`${-m} ${-m} ${2 * m} ${2 * m}`} role="img" aria-label="Template">
      <rect x={-m} y={-m} width={2 * m} height={2 * m} fill="#000" />
      <circle cx={0} cy={0} r={R} fill="none" stroke="#555" strokeWidth={R * 0.004} strokeDasharray={`${R * 0.02} ${R * 0.015}`} />
      <circle cx={0} cy={0} r={R * 0.012} fill="#888" />
      {t.x.map((x, i) => (
        <circle key={i} cx={x} cy={fy(t.y[i])} r={R * (0.008 + 0.025 * Math.sqrt(t.intensity[i]))} fill="#fff">
          <title>{t.hkl[i].join(" ")} · {formatValue(t.intensity[i])}</title>
        </circle>
      ))}
    </svg>
  );
}
