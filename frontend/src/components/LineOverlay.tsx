import { useViewProfiles, type ProfileLine } from "../state/profiles";
import { bandPolygon, calibratedLength, pointAt } from "../viewer/profile";
import { formatValue } from "../format";
import type { AxisCal, Transform } from "./ImageView";

interface Props {
  t: Transform;
  view: string;
  calX: AxisCal;
  calY: AxisCal;
}

/** SVG drawing of a view's profile lines: the integration band, endpoint
 * handles of the selected line, its length, the position hovered on the
 * profile plot and the part picked on it. */
export function LineOverlay({ t, view, calX, calY }: Props) {
  const { lines, selected, hover, pick } = useViewProfiles(view);
  return (
    <g className="line-overlay">
      {lines
        .filter((l) => l.visible)
        .map((l) => {
          const length = calibratedLength(l, calX, calY);
          return (
            <LineShape
              key={l.id}
              t={t}
              line={l}
              selected={l.id === selected}
              label={`${formatValue(length)} ${calX.units || "px"}`}
              hoverT={hover !== null && length > 0 && hover <= length ? hover / length : null}
              pick={pick?.line === l.id ? pick : null}
            />
          );
        })}
    </g>
  );
}

const screenPoints = (t: Transform, pts: [number, number][]) => pts.map(([x, y]) => t.toScreen(x, y).join(",")).join(" ");

function LineShape({
  t, line, selected, label, hoverT, pick,
}: {
  t: Transform;
  line: ProfileLine;
  selected: boolean;
  label: string;
  hoverT: number | null;
  pick: { t0: number; t1: number } | null;
}) {
  const [ax, ay] = t.toScreen(line.x0, line.y0);
  const [bx, by] = t.toScreen(line.x1, line.y1);
  const showBand = line.width > 1;
  // At least a pixel long so a picked point shows.
  const pickBand = pick && bandPolygon(line, Math.max(1, line.width), pick.t0, pick.t1);
  return (
    <g>
      {showBand && (
        <polygon
          points={screenPoints(t, bandPolygon(line, line.width))}
          fill={line.color}
          fillOpacity={selected ? 0.16 : 0.08}
          stroke={line.color}
          strokeOpacity={0.6}
          strokeDasharray="3 3"
        />
      )}
      {pickBand && (
        <polygon
          points={screenPoints(t, pickBand)}
          fill={line.color}
          fillOpacity={0.45}
          stroke={line.color}
          strokeWidth={pick!.t0 === pick!.t1 ? 3 : 1}
        />
      )}
      <line x1={ax} y1={ay} x2={bx} y2={by} className="line-shadow" />
      <line x1={ax} y1={ay} x2={bx} y2={by} stroke={line.color} strokeWidth={selected ? 2 : 1.5} />
      {selected &&
        [
          [ax, ay],
          [bx, by],
        ].map(([x, y], i) => <rect key={i} x={x - 4} y={y - 4} width={8} height={8} className="line-handle" stroke={line.color} />)}
      {hoverT !== null && (() => {
        const [hx, hy] = t.toScreen(...pointAt(line, hoverT));
        return <circle cx={hx} cy={hy} r={4} fill="none" stroke={line.color} strokeWidth={2} className="line-hover" />;
      })()}
      <text x={(ax + bx) / 2} y={(ay + by) / 2 - 8} className="line-label" fill={line.color} textAnchor="middle">
        {line.name} · {label}
      </text>
    </g>
  );
}
