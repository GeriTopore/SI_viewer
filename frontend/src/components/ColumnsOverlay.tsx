import { useEffect, useMemo } from "react";
import type { DatasetSummary } from "../api/client";
import { allDatasets, useDatasets } from "../state/datasets";
import {
  COLUMN_VALUES, SUBLATTICE_COLOURS, columnValues, ensureTable, inGroup, nearestColumn, sublatticesOf, useColumns, valuesFrame,
  type ColumnPoints,
} from "../state/columns";
import { DEFAULT_LAYER, useLayers, type LayerProps } from "../state/layers";
import { useDisplay } from "../state/display";
import { formatValue } from "../format";
import { colormapLut } from "../viewer/colormaps";
import type { DisplayParams } from "../viewer/renderer";
import { autoContrast } from "../views/useImageDisplay";
import { pixelSize } from "./AtomicSection";
import type { PanelLayer } from "./LayersPanel";
import type { Transform } from "./ImageView";
import type { Frame } from "../api/arrays";

/** The shape layers of an image's atomic columns: the guesses (found) and
 * the fitted positions, one layer for all or, once split, one per
 * sublattice (``group`` "0", "1" …; "u" for columns in none). Their
 * visibility, opacity, colour, size and colouring are layer props under
 * these keys. */
export type ColumnsKind = "found" | "fitted";
export const columnsLayerKey = (image: string, kind: ColumnsKind, group = "") =>
  `columns:${kind}${group ? `.${group}` : ""}:${image}`;
export const COLUMNS_COLOUR: Record<ColumnsKind, string> = { found: "#ffd60a", fitted: "#ff4d6d" };
const UNASSIGNED_COLOUR = "#9aa0a6";

/** A key from ``columnsLayerKey``, parsed. */
export function parseColumnsKey(key: string): { kind: ColumnsKind; group: string; image: string } | null {
  const m = /^columns:(found|fitted)(?:\.(\w+))?:(.+)$/.exec(key);
  return m ? { kind: m[1] as ColumnsKind, group: m[2] ?? "", image: m[3] } : null;
}

const columnsOf = (files: Parameters<typeof allDatasets>[0], image: string) =>
  allDatasets(files).find((d) => d.parent_id === image && d.atomic?.output === "columns");

/** The fitted columns' groups: all of them ("Fitted columns"), then one
 * per sublattice once split (the datasets under the Columns; and "No
 * sublattice" when some columns are in none). ``group`` picks its columns
 * (see ``inGroup``); ``id`` names its layer (a sublattice's dataset id);
 * ``dataset`` is the dataset it shows. Once split, all of them start
 * hidden (the sublattices show them). */
export interface FittedGroupInfo {
  group: string;
  id: string;
  label: string;
  colour: string;
  dataset?: string;
  hidden?: boolean;
}
export function fittedGroups(files: Parameters<typeof allDatasets>[0], c: DatasetSummary): FittedGroupInfo[] {
  const subs = sublatticesOf(files, c);
  const all: FittedGroupInfo = { group: "", id: "", label: "Fitted columns", colour: COLUMNS_COLOUR.fitted, dataset: c.id, hidden: subs.length > 0 };
  const groups: FittedGroupInfo[] = subs.map((d, k) => ({
    group: String(k), id: d.id, label: d.title, colour: SUBLATTICE_COLOURS[d.atomic?.index ?? k] ?? UNASSIGNED_COLOUR, dataset: d.id,
  }));
  const assigned = subs.reduce((a, d) => a + (d.atomic?.count ?? 0), 0);
  if (subs.length && (c.atomic?.count ?? 0) > assigned) groups.push({ group: "u", id: "u", label: "No sublattice", colour: UNASSIGNED_COLOUR });
  return [all, ...groups];
}

/** A group's layer props: as set, else the defaults (hidden if it starts so). */
export const groupProps = (p: LayerProps | undefined, g: FittedGroupInfo | undefined): LayerProps =>
  p ?? (g?.hidden ? { ...DEFAULT_LAYER, visible: false } : DEFAULT_LAYER);

type ColumnLayer = PanelLayer & { kind: ColumnsKind; image: string; group: string; dataset?: string };

/** The column layers of images ``images`` (ids) for the layer list: guesses
 * once found; the fitted columns (or each sublattice) once fitted. */
export function useColumnLayers(images: string[]): ColumnLayer[] {
  const byId = useColumns((s) => s.byId);
  const files = useDatasets((s) => s.files);
  const props = useLayers((s) => s.props);
  const displays = useDisplay((s) => s.display);
  const ids = images.join(",");
  return useMemo(() => {
    const out: ColumnLayer[] = [];
    const setProps = useLayers.getState().setProps;
    for (const image of ids ? ids.split(",") : []) {
      const add = (kind: ColumnsKind, g: FittedGroupInfo) => {
        const key = columnsLayerKey(image, kind, g.id);
        const p = groupProps(props[key], g);
        const by = kind === "fitted" ? p.colourBy : undefined;
        out.push({
          id: key, kind, image, group: g.id, dataset: g.dataset, label: g.label, visible: p.visible, opacity: p.opacity,
          colour: (by && displays[key]?.colormap) || (p.colour ?? g.colour),
          display: undefined, removable: kind === "found",
          size: { value: p.size ?? 1, onChange: (v: number) => setProps(key, { visible: p.visible, size: v }) },
          ...(kind === "fitted" && {
            choice: {
              value: by ?? "",
              title: "Colour the columns by a value (its limits: select this layer)",
              options: [{ value: "", label: "Plain" }, ...COLUMN_VALUES.map((c) => ({ value: c.id, label: c.label }))],
              onChange: (v: string) => setProps(key, { visible: p.visible, colourBy: v || undefined }),
            },
          }),
        });
      };
      if (byId[image]?.found?.length) add("found", { group: "", id: "", label: "Column guesses", colour: COLUMNS_COLOUR.found });
      const c = columnsOf(files, image);
      if (c) for (const g of fittedGroups(files, c)) add("fitted", g);
    }
    return out;
  }, [byId, files, props, displays, ids]);
}

/** The values a shape layer (``key``) is coloured by, for the contrast
 * toolbar and the Histogram tab: as a frame (of its group's columns), with
 * a label. */
export function useColumnsContrast(key: string | null): { frame: Frame; label: string } | null {
  const shape = key ? parseColumnsKey(key) : null;
  const columnsDs = useDatasets((s) => (shape?.kind === "fitted" ? columnsOf(s.files, shape.image) : undefined));
  const table = useColumns((s) => (columnsDs ? s.tables[columnsDs.id]?.points : undefined));
  const by = useLayers((s) => (key ? s.props[key]?.colourBy : undefined));
  const files = useDatasets((s) => s.files);
  const g = columnsDs ? fittedGroups(files, columnsDs).find((x) => x.id === (shape?.group ?? "")) : undefined;
  const group = g?.group;
  const name = g?.label;
  return useMemo(() => {
    if (!table || !by || group === undefined) return null;
    const label = COLUMN_VALUES.find((c) => c.id === by)?.label ?? by;
    return { frame: valuesFrame(table, by, group), label: `${name ?? "Fitted columns"} · ${label}` };
  }, [table, by, group, name]);
}

/** Dots (or rings) of radius ``r`` (data px) at points [y, x] (centres at
 * integers; pixel i spans [i, i + 1] here), as one SVG path. */
function circles(points: Iterable<[number, number]>, r: number): string {
  const parts: string[] = [];
  const d = (2 * r).toFixed(2);
  const rr = r.toFixed(2);
  for (const [y, x] of points) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    parts.push(`M${(x + 0.5 - r).toFixed(2)} ${(y + 0.5).toFixed(2)}a${rr} ${rr} 0 1 0 ${d} 0a${rr} ${rr} 0 1 0 -${d} 0`);
  }
  return parts.join("");
}

function* fitted(p: ColumnPoints, group: string, failed: boolean): Generator<[number, number]> {
  for (let i = 0; i < p.y.length; i++) if (!!p.failed[i] === failed && inGroup(p, group, i)) yield [p.y[i], p.x[i]];
}

/** Colour steps of a layer coloured by value (one path each). */
const STEPS = 64;

/** A group's fitted columns split by colour: value → step of the display's
 * colormap (with its limits and gamma). */
function colouredPaths(p: ColumnPoints, group: string, v: Float32Array, d: DisplayParams, r: number): { colour: string; d: string }[] {
  const lut = colormapLut(d.colormap);
  const buckets: [number, number][][] = Array.from({ length: STEPS }, () => []);
  const span = d.hi - d.lo || 1;
  for (let i = 0; i < v.length; i++) {
    if (p.failed[i] || !Number.isFinite(v[i]) || !inGroup(p, group, i)) continue;
    let t = Math.min(1, Math.max(0, (v[i] - d.lo) / span));
    if (d.gamma && d.gamma !== 1) t = t ** d.gamma;
    buckets[Math.min(STEPS - 1, Math.floor(t * STEPS))].push([p.y[i], p.x[i]]);
  }
  return buckets.flatMap((pts, k) => {
    if (!pts.length) return [];
    const j = Math.round(((k + 0.5) / STEPS) * 255) * 4;
    return [{ colour: `rgb(${lut[j]},${lut[j + 1]},${lut[j + 2]})`, d: circles(pts, r) }];
  });
}

/** One group of fitted columns (all, or a sublattice) as dots, coloured
 * plainly or by a value; failed fits as rings. */
function FittedGroup({ table, tableKey, image, info, r }: {
  table: ColumnPoints; tableKey: string; image: string; info: FittedGroupInfo; r: number;
}) {
  const { group, colour } = info;
  const key = columnsLayerKey(image, "fitted", info.id);
  const props = groupProps(useLayers((s) => s.props[key]), info);
  const display = useDisplay((s) => s.display[key]);
  const by = props.colourBy;
  const values = useMemo(() => (by ? columnValues(table, by) : null), [table, by]);
  // Limits from the group's values when first coloured by them (or the fit changed).
  useEffect(() => {
    if (by && props.visible) autoContrast(key, valuesFrame(table, by, group), `${by}:${tableKey}`, () => "viridis");
  }, [by, table, key, group, tableKey, props.visible]);
  const rr = r * (props.size ?? 1);
  const coloured = !!(values && display);
  const plain = useMemo(
    () => (props.visible && !coloured ? circles(fitted(table, group, false), rr * 0.8) : ""),
    [props.visible, coloured, table, group, rr],
  );
  const shaded = useMemo(
    () => (props.visible && coloured ? colouredPaths(table, group, values!, display!, rr * 0.8) : []),
    [props.visible, coloured, table, group, values, display?.lo, display?.hi, display?.gamma, display?.colormap, rr], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const failed = useMemo(() => (props.visible ? circles(fitted(table, group, true), rr * 1.3) : ""), [props.visible, table, group, rr]);
  if (!plain && !shaded.length && !failed) return null;
  const c = props.colour ?? colour;
  return (
    <g opacity={props.opacity}>
      {plain && <path className="columns-fitted" d={plain} fill={c} />}
      {shaded.map((p) => (
        <path key={p.colour} className="columns-fitted" d={p.d} fill={p.colour} />
      ))}
      {failed && <path className="columns-failed" d={failed} stroke={c} />}
    </g>
  );
}

/** The atomic columns of image ``image``, drawn over it as layers (see
 * ``useColumnLayers``): the guesses, and the fitted columns (or each
 * sublattice) as dots, failed fits as rings. The fitted ones can be
 * coloured by a value (their layer's ``colourBy``, with the display under
 * the layer's key); the one under the pointer shows its values. */
export function ColumnsOverlay({ t, image }: { t: Transform; image: string | undefined }) {
  const st = useColumns((s) => (image ? s.byId[image] : undefined));
  const columnsDs = useDatasets((s) => (image ? columnsOf(s.files, image) : undefined));
  const imageDs = useDatasets((s) => (image ? allDatasets(s.files).find((d) => d.id === image) : undefined));
  const props = useLayers((s) => s.props);
  const foundProps = (image && props[columnsLayerKey(image, "found")]) || DEFAULT_LAYER;
  const files = useDatasets((s) => s.files);
  const groups = useMemo(() => (columnsDs ? fittedGroups(files, columnsDs) : []), [files, columnsDs]);
  const shown = (g: FittedGroupInfo): LayerProps => groupProps(image ? props[columnsLayerKey(image, "fitted", g.id)] : undefined, g);
  const anyFitted = groups.some((g) => shown(g).visible);
  useEffect(() => {
    if (anyFitted && columnsDs) ensureTable(columnsDs);
  }, [anyFitted, columnsDs]);
  const tableEntry = useColumns((s) => (columnsDs ? s.tables[columnsDs.id] : undefined));
  const table = tableEntry?.points;
  const found = foundProps.visible ? st?.found : null;

  // A few screen pixels, less when the columns are close on screen.
  const spacing = (st?.settings.distance ?? 4) * t.zoom;
  const r = Math.min(3, Math.max(1, spacing * 0.22)) / t.zoom;
  const rFound = r * (foundProps.size ?? 1);
  const foundPath = useMemo(() => (found ? circles(found, rFound) : ""), [found, rFound]);

  // The fitted column under the pointer, in a shown group (within the least
  // distance between columns).
  // A column is in all of them and (once split) in one more: the topmost
  // shown one is the one hovered.
  const groupOf = (i: number) => [...groups].reverse().find((g) => inGroup(table!, g.group, i) && shown(g).visible);
  const hit = anyFitted && table && t.pointer
    ? nearestColumn(table, t.pointer.x, t.pointer.y, Math.max(st?.settings.distance ?? 4, 3 / t.zoom), (i) => !!groupOf(i))
    : -1;
  const hitGroup = hit >= 0 ? groupOf(hit) : undefined;
  // Its sublattice, shown or not.
  const hitSub = hit >= 0 ? groups.find((g) => g.group && inGroup(table!, g.group, hit)) : undefined;
  const hitSize = hitGroup ? (shown(hitGroup).size ?? 1) : 1;

  if (!foundPath && !(anyFitted && table && image)) return null;
  const [x0, y0] = t.toScreen(0, 0);
  const foundColour = foundProps.colour ?? COLUMNS_COLOUR.found;
  return (
    <>
      <g className="columns-overlay" transform={`translate(${x0} ${y0}) scale(${t.zoom})`}>
        {foundPath && <path className="columns-found" d={foundPath} fill={foundColour} opacity={foundProps.opacity} />}
        {table && image && tableEntry &&
          groups.map((g) => (
            <FittedGroup key={g.id} table={table} tableKey={tableEntry.key} image={image} info={g} r={r} />
          ))}
        {hit >= 0 && <path className="columns-hovered" d={circles([[table!.y[hit], table!.x[hit]]], Math.max(r * hitSize * 1.2, r * 1.9))} />}
      </g>
      {hit >= 0 && (
        <ColumnTip
          t={t}
          p={table!}
          i={hit}
          px={imageDs ? pixelSize(imageDs) : null}
          by={hitGroup ? shown(hitGroup).colourBy : undefined}
          sublattice={hitSub?.label}
        />
      )}
    </>
  );
}

/** The values of fitted column ``i``, beside it. */
function ColumnTip({ t, p, i, px, by, sublattice }: {
  t: Transform; p: ColumnPoints; i: number; px: { x: number; units: string } | null; by: string | undefined;
  sublattice?: string;
}) {
  const [sx, sy] = t.toScreen(p.x[i] + 0.5, p.y[i] + 0.5);
  const W = 190;
  const H = 96;
  const left = t.width !== undefined && sx + 14 + W > t.width ? sx - 14 - W : sx + 14;
  const top = t.height !== undefined && sy + 14 + H > t.height ? sy - 14 - H : sy + 14;
  const v = (id: string) => p.values[id]?.[i] ?? NaN;
  const len = (x: number) =>
    !px || px.units === "px" ? `${formatValue(x)} px` : `${formatValue(x * px.x)} ${px.units}`;
  const extra = by && !["intensity", "amplitude", "offset", "sigma_1", "sigma_2", "angle"].includes(by)
    ? COLUMN_VALUES.find((c) => c.id === by)
    : undefined;
  return (
    <foreignObject x={left} y={top} width={W} height={H} style={{ overflow: "visible" }}>
      <div className="columns-tip">
        <div>
          <span className="muted">Column {i}</span> x {formatValue(p.x[i])}, y {formatValue(p.y[i])}
        </div>
        {sublattice && <div className="muted">{sublattice}</div>}
        {p.failed[i] ? (
          <div>Fit failed</div>
        ) : (
          <>
            <div>
              I {formatValue(v("intensity"))} <span className="muted">· A</span> {formatValue(v("amplitude"))}{" "}
              <span className="muted">· offset</span> {formatValue(v("offset"))}
            </div>
            <div>
              σ {len(Math.abs(v("sigma_1")))} × {len(Math.abs(v("sigma_2")))}{" "}
              <span className="muted">·</span> {formatValue(columnValues(p, "angle")[i])}°
            </div>
            {extra && (
              <div>
                <span className="muted">{extra.label}</span> {formatValue(columnValues(p, extra.id)[i])}
              </div>
            )}
          </>
        )}
      </div>
    </foreignObject>
  );
}
