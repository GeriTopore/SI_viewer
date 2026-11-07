import { useRef } from "react";
import { getViewProfiles, useProfiles, useViewProfiles, type ProfileLine } from "../state/profiles";
import { isColour } from "../viewer/colormaps";
import { calibratedLength, sampleProfile } from "../viewer/profile";
import { formatValue } from "../format";
import { downloadCsv, safeFileName } from "../download";
import type { AxisCal, ViewLayer } from "./ImageView";
import { ProfilePlot, type DistanceRange, type ProfileSeries, type ProfileTable } from "./ProfilePlot";

// Profile of a grey image shown with maps.
const BASE_COLOR = "#a3a3a3";
// Lines after the first are told apart by dashes.
const DASHES: (number[] | undefined)[] = [undefined, [6, 4], [2, 3], [10, 3, 2, 3]];

interface Props {
  view: string;
  /** Image layers (base first); each visible one gets a profile per line. */
  layers: ViewLayer[];
  calX: AxisCal;
  calY: AxisCal;
  /** Name for exported files. */
  title: string;
  /** Picking a point or range of a line on the plot (spectrum images). */
  pickable?: boolean;
  /** Switches to the spectrum (where a picked spectrum is shown). */
  onShowSpectrum?: () => void;
}

/** Profiles of the visible layers along a view's lines, with the line list
 * (width, visibility), normalisation and CSV export. */
export function ProfilePanel({ view, layers, calX, calY, title, pickable, onShowSpectrum }: Props) {
  const { lines, selected, normalise, pick } = useViewProfiles(view);
  const { patch, updateLine, removeLine } = useProfiles.getState();
  const units = calX.units || "px";
  const shown = lines.filter((l) => l.visible);
  const drawn = layers.filter((l) => l.visible && l.frame);
  // The base image (e.g. HAADF) gets its own axis beside maps, unless normalised.
  const baseAxis = !normalise && drawn.length > 1 && drawn[0] === layers[0];

  // Recomputed only when a line, an image or a setting changes (not on hover).
  const deps = [
    lines, normalise, baseAxis, calX.scale, calY.scale,
    ...drawn.flatMap((l) => [l.frame, l.label, l.display?.colormap]),
  ];
  const memo = useRef<{ deps: unknown[]; tables: ProfileTable[] } | null>(null);
  if (!memo.current || memo.current.deps.length !== deps.length || memo.current.deps.some((d, i) => d !== deps[i])) {
    memo.current = { deps, tables: profileTables() };
  }
  const tables = memo.current.tables;

  function profileTables(): ProfileTable[] {
    return shown.map((line, li) => {
        const length = calibratedLength(line, calX, calY);
        const series: ProfileSeries[] = drawn.map((layer, i) => {
          let values = sampleProfile(layer.frame!, line);
          if (normalise) {
            const max = values.reduce((m, v) => (Number.isFinite(v) && Math.abs(v) > m ? Math.abs(v) : m), 0);
            if (max > 0) values = values.map((v) => v / max);
          }
          // Maps in their colour; a grey image in the line's colour when
          // alone, else neutral so it isn't mistaken for an element.
          const colormap = layer.display?.colormap;
          return {
            label: shown.length > 1 ? `${line.name} · ${layer.label}` : layer.label,
            color: colormap && isColour(colormap) ? colormap : drawn.length > 1 ? BASE_COLOR : line.color,
            dash: DASHES[li % DASHES.length],
            values,
            scale: baseAxis && i === 0 ? "y2" : "y",
          };
        });
        const n = series[0]?.values.length ?? 0;
        return { x: Array.from({ length: n }, (_, i) => (length * i) / Math.max(1, n - 1)), series };
    });
  }

  // Picks are kept as fractions of the line, so they follow it when it moves.
  const pickLine = pick && shown.find((l) => l.id === pick.line);
  const pickLength = pickLine ? calibratedLength(pickLine, calX, calY) : 0;
  const picked: DistanceRange | null = pickLine ? { d0: pick!.t0 * pickLength, d1: pick!.t1 * pickLength } : null;
  const onPick = ({ d0, d1 }: DistanceRange) => {
    // The selected line (or the first) is the one picked.
    const line = shown.find((l) => l.id === selected) ?? shown[0];
    if (!line) return;
    const length = calibratedLength(line, calX, calY);
    if (!length) return;
    const clamp = (d: number) => Math.min(1, Math.max(0, d / length));
    patch(view, { pick: { line: line.id, t0: clamp(Math.min(d0, d1)), t1: clamp(Math.max(d0, d1)) } });
  };

  const exportCsv = () =>
    downloadCsv(
      [
        ["line", `distance (${units})`, ...drawn.map((l) => l.label)],
        ...tables.flatMap((t, li) => Array.from(t.x, (d, i) => [shown[li].name, d, ...t.series.map((s) => s.values[i])])),
      ],
      `${safeFileName(title)}_profile${normalise ? "_normalised" : ""}.csv`,
    );

  return (
    <div className="profile-panel">
      <div className="profile-header">
        <div className="profile-lines">
          {lines.map((l) => (
            <LineChip
              key={l.id}
              line={l}
              selected={l.id === selected}
              units={units}
              onSelect={() => patch(view, { selected: l.id })}
              onChange={(p) => updateLine(view, l.id, p)}
              onRemove={() => removeLine(view, l.id)}
            />
          ))}
        </div>
        <span className="profile-spacer" />
        {pickable && picked && (
          <span className="profile-pick">
            Spectrum at {formatValue(picked.d0)}
            {picked.d1 !== picked.d0 && `–${formatValue(picked.d1)}`} {units}
            {onShowSpectrum && (
              <button className="link-button" onClick={onShowSpectrum} title="Show it on the Spectrum tab">
                Show
              </button>
            )}
            <button className="icon-button" onClick={() => patch(view, { pick: null })} title="Remove this spectrum">
              ×
            </button>
          </span>
        )}
        <label className="profile-check" title="Scale each profile to a maximum of 1">
          <input type="checkbox" checked={normalise} onChange={(e) => patch(view, { normalise: e.target.checked })} />
          Normalise
        </label>
        <button className="link-button" onClick={exportCsv} disabled={!tables.length} title="Save the profiles as CSV">
          CSV
        </button>
      </div>
      {!lines.length ? (
        <p className="muted small profile-empty">
          Drag on the image to draw a line; its profile appears here for every image layer shown.
          {pickable && " Click or drag on the profile to see the spectrum there."}
        </p>
      ) : drawn.length ? (
        <ProfilePlot
          tables={tables}
          xUnits={units}
          onPick={pickable ? onPick : undefined}
          picked={pickable ? picked : null}
          onHover={(d) => getViewProfiles(view).hover !== d && patch(view, { hover: d })}
        />
      ) : (
        <p className="muted small profile-empty">No image shown to profile.</p>
      )}
    </div>
  );
}

function LineChip({
  line, selected, units, onSelect, onChange, onRemove,
}: {
  line: ProfileLine;
  selected: boolean;
  units: string;
  onSelect: () => void;
  onChange: (p: Partial<ProfileLine>) => void;
  onRemove: () => void;
}) {
  return (
    <span className={`profile-line${selected ? " selected" : ""}`} onClick={onSelect}>
      <input
        type="checkbox"
        checked={line.visible}
        onChange={(e) => onChange({ visible: e.target.checked })}
        onClick={(e) => e.stopPropagation()}
        title="Show this line"
        style={{ accentColor: line.color }}
      />
      <span className="profile-line-name" style={{ color: line.color }}>
        {line.name}
      </span>
      <label className="profile-width" title={`Width integrated across the line (pixels; ${units} along it)`}>
        w
        <input
          type="number"
          min={1}
          max={200}
          step={1}
          value={line.width}
          onChange={(e) => {
            const w = Math.round(Number(e.target.value));
            if (w >= 1) onChange({ width: Math.min(200, w) });
          }}
        />
        px
      </label>
      <button
        className="icon-button"
        onClick={(e) => {
          e.stopPropagation();
          onRemove();
        }}
        title={`Delete ${line.name}`}
      >
        ×
      </button>
    </span>
  );
}
