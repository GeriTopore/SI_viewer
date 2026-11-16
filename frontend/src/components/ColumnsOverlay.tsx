import { useEffect, useMemo } from "react";
import type { DatasetSummary } from "../api/client";
import { allDatasets, useDatasets } from "../state/datasets";
import {
  BOND_VALUES, COLUMN_VALUES, SUBLATTICE_COLOURS, bondFrame, columnValues, ensureBonds, ensureTable, inGroup, nearestColumn,
  sublatticesOf, useBondTables, useColumns, valuesFrame, type ColumnPoints,
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

/** Results of the column analysis (kemstem) are datasets ("items"): the
 * Columns, its sublattices, neighbourhoods and bonds. Selected in the
 * dataset pane, each is drawn over the image it came from: a panel of its
 * own in the grid (the image under it), or a layer over the image when
 * merged. Their visibility, opacity, colour, size and colouring are layer
 * props under the item's id (its display, when coloured by a value, too).
 * The guesses of the Atomic columns process are not a dataset: they are a
 * layer of their image (``foundLayerKey``). */
const ITEM_OUTPUTS = ["columns", "sublattice", "neighbourhood", "bonds"];
export const isColumnItem = (d: DatasetSummary | undefined) => !!d?.atomic && ITEM_OUTPUTS.includes(d.atomic.output);
export const foundLayerKey = (image: string) => `columns:found:${image}`;
export const isFoundKey = (key: string) => key.startsWith("columns:found:");
const COLOUR = { found: "#ffd60a", fitted: "#ff4d6d", bonds: "#f2f2f2", neighbourhood: "#c77dff" };

type Files = Parameters<typeof allDatasets>[0];

/** The Columns dataset an item belongs to, and the image under it. */
export function columnsOfItem(files: Files, item: DatasetSummary | undefined): DatasetSummary | undefined {
  const all = allDatasets(files);
  let up = item;
  while (up && up.atomic?.output !== "columns" && isColumnItem(up)) {
    const parent = up.parent_id;
    up = all.find((x) => x.id === parent);
  }
  return up?.atomic?.output === "columns" ? up : undefined;
}
export const imageOfItem = (files: Files, item: DatasetSummary | undefined) => columnsOfItem(files, item)?.parent_id ?? null;

/** What an item draws: columns as dots (``group``: which, see ``inGroup``)
 * or bonds as lines; its colour, label and the values it can be coloured
 * by; props it starts with. */
export interface ItemInfo {
  kind: "dots" | "bonds";
  columns: DatasetSummary;
  group: string;
  colour: string;
  label: string;
  values: { id: string; label: string }[];
  defaults?: Partial<LayerProps>;
}

/** Bonds start coloured by their length (in inferno). */
const BONDS_DEFAULTS: Partial<LayerProps> = { colourBy: "length" };

export function itemInfo(files: Files, item: DatasetSummary): ItemInfo | null {
  const columns = columnsOfItem(files, item);
  if (!columns) return null;
  const all = allDatasets(files);
  const subs = sublatticesOf(files, columns);
  const groupOf = (d: DatasetSummary | undefined) => {
    const k = d ? subs.findIndex((x) => x.id === d.id) : -1;
    return k >= 0 ? String(k) : "";
  };
  const short = (d: DatasetSummary | undefined) =>
    !d || d.atomic?.output === "columns" ? "all" : d.title.replace(/^Sublattice /, "");
  switch (item.atomic?.output) {
    case "columns":
      return { kind: "dots", columns, group: "", colour: COLOUR.fitted, label: item.title, values: COLUMN_VALUES };
    case "sublattice":
      return {
        kind: "dots", columns, group: groupOf(item), colour: SUBLATTICE_COLOURS[item.atomic.index ?? 0] ?? COLOUR.fitted,
        label: item.title, values: COLUMN_VALUES,
      };
    case "neighbourhood": {
      // Its origins.
      const origins = all.find((d) => d.id === (item.atomic?.origins ?? item.parent_id));
      return {
        kind: "dots", columns, group: groupOf(origins), colour: COLOUR.neighbourhood,
        label: item.title, values: COLUMN_VALUES,
      };
    }
    case "bonds": {
      const nb = all.find((d) => d.id === item.parent_id);
      const origins = all.find((d) => d.id === (nb?.atomic?.origins ?? nb?.parent_id));
      const to = all.find((d) => d.id === nb?.atomic?.neighbours);
      return {
        kind: "bonds", columns, group: "", colour: COLOUR.bonds, label: `${item.title} (${short(origins)} → ${short(to)})`,
        values: BOND_VALUES, defaults: BONDS_DEFAULTS,
      };
    }
  }
  return null;
}

/** An item's layer props: as set, over its defaults (a setting set to
 * nothing, e.g. "Plain", stays so). */
export const itemProps = (p: LayerProps | undefined, info: ItemInfo | null): LayerProps => ({
  ...DEFAULT_LAYER, ...info?.defaults, ...p,
});

/** An item's row in the layer list: show/hide, colour (or its colormap when
 * coloured by a value), opacity, size and colour-by. */
export function itemPanelLayer(
  item: DatasetSummary, info: ItemInfo, p: LayerProps | undefined, display: DisplayParams | undefined, removable: boolean,
): PanelLayer {
  const setProps = useLayers.getState().setProps;
  const q = itemProps(p, info);
  return {
    id: item.id, label: info.label, visible: q.visible, opacity: q.opacity,
    colour: (q.colourBy && display?.colormap) || (q.colour ?? info.colour),
    display: undefined, removable,
    size: { value: q.size ?? 1, onChange: (v: number) => setProps(item.id, { size: v }) },
    choice: {
      value: q.colourBy ?? "",
      title: `Colour the ${info.kind === "bonds" ? "bonds" : "columns"} by a value (its limits: select this layer)`,
      options: [{ value: "", label: "Plain" }, ...info.values.map((c) => ({ value: c.id, label: c.label }))],
      onChange: (v: string) => setProps(item.id, { colourBy: v || undefined }),
    },
  };
}

/** The guesses' row in the layer list of their image (once found). */
export function foundPanelLayer(image: string, p: LayerProps | undefined): PanelLayer {
  const setProps = useLayers.getState().setProps;
  const q = { ...DEFAULT_LAYER, ...p };
  const key = foundLayerKey(image);
  return {
    id: key, label: "Column guesses", visible: q.visible, opacity: q.opacity, colour: q.colour ?? COLOUR.found,
    display: undefined, removable: true,
    size: { value: q.size ?? 1, onChange: (v: number) => setProps(key, { size: v }) },
  };
}

/** The values an item is coloured by, for the contrast toolbar and the
 * Histogram tab: as a frame (of its columns or bonds), with a label. */
export function useItemContrast(item: DatasetSummary | null): { frame: Frame; label: string } | null {
  const files = useDatasets((s) => s.files);
  const info = useMemo(() => (item ? itemInfo(files, item) : null), [files, item]);
  const table = useColumns((s) => (info ? s.tables[info.columns.id]?.points : undefined));
  const bonds = useBondTables((s) => (item && info?.kind === "bonds" ? s.tables[item.id]?.points : undefined));
  const p = useLayers((s) => (item ? s.props[item.id] : undefined));
  const by = itemProps(p, info).colourBy;
  return useMemo(() => {
    if (!info || !by || !item) return null;
    const label = `${info.label} · ${info.values.find((v) => v.id === by)?.label ?? by}`;
    if (info.kind === "bonds") return bonds && (by === "length" || by === "angle") ? { frame: bondFrame(bonds, by), label } : null;
    return table ? { frame: valuesFrame(table, by, info.group), label } : null;
  }, [info, by, item, table, bonds]);
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

/** An item's columns as dots, coloured plainly or by a value; failed fits
 * as rings. */
function ItemDots({ table, tableKey, item, info, r }: {
  table: ColumnPoints; tableKey: string; item: DatasetSummary; info: ItemInfo; r: number;
}) {
  const key = item.id;
  const { group } = info;
  const props = itemProps(useLayers((s) => s.props[key]), info);
  const display = useDisplay((s) => s.display[key]);
  const by = props.colourBy;
  const values = useMemo(() => (by ? columnValues(table, by) : null), [table, by]);
  // Limits from its values when first coloured by them (or the fit changed).
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
  const c = props.colour ?? info.colour;
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

/** Bonds as lines from their origin column along their vector, plain or
 * coloured by length or angle. */
function ItemBonds({ table, item, info }: { table: ColumnPoints; item: DatasetSummary; info: ItemInfo }) {
  const key = item.id;
  const props = itemProps(useLayers((s) => s.props[key]), info);
  const display = useDisplay((s) => s.display[key]);
  const entry = useBondTables((s) => s.tables[item.id]);
  useEffect(() => {
    if (props.visible) ensureBonds(item);
  }, [props.visible, item]);
  const p = entry?.points;
  const by = props.colourBy === "length" || props.colourBy === "angle" ? props.colourBy : undefined;
  useEffect(() => {
    if (p && by && props.visible) autoContrast(key, bondFrame(p, by), `${by}:${entry?.key}`, () => "inferno");
  }, [p, by, key, entry?.key, props.visible]);
  const paths = useMemo(() => {
    if (!p || !props.visible) return [];
    const lut = by && display ? colormapLut(display.colormap) : null;
    const span = display ? display.hi - display.lo || 1 : 1;
    const buckets: string[][] = Array.from({ length: lut ? STEPS : 1 }, () => []);
    for (let i = 0; i < p.origin.length; i++) {
      const o = p.origin[i];
      if (o < 0 || !Number.isFinite(p.dx[i]) || !Number.isFinite(table.x[o])) continue;
      const x0 = table.x[o] + 0.5;
      const y0 = table.y[o] + 0.5;
      let b = 0;
      if (lut && display && by) {
        let t = Math.min(1, Math.max(0, (p[by][i] - display.lo) / span));
        if (display.gamma && display.gamma !== 1) t = t ** display.gamma;
        b = Math.min(STEPS - 1, Math.floor(t * STEPS));
      }
      buckets[b].push(`M${x0.toFixed(2)} ${y0.toFixed(2)}l${p.dx[i].toFixed(2)} ${p.dy[i].toFixed(2)}`);
    }
    return buckets.flatMap((parts, k) => {
      if (!parts.length) return [];
      const j = Math.round(((k + 0.5) / STEPS) * 255) * 4;
      return [{ colour: lut ? `rgb(${lut[j]},${lut[j + 1]},${lut[j + 2]})` : (props.colour ?? info.colour), d: parts.join("") }];
    });
  }, [p, table, props.visible, props.colour, by, display?.lo, display?.hi, display?.gamma, display?.colormap]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!paths.length) return null;
  return (
    <g opacity={props.opacity} style={{ strokeWidth: 1.5 * (props.size ?? 1) }}>
      {paths.map((q) => (
        <path key={q.colour} className="columns-bonds" d={q.d} stroke={q.colour} />
      ))}
    </g>
  );
}

/** Column items of image ``image`` drawn over it (``items``, in order:
 * later on top; bonds under dots), and its guesses with ``found``. The
 * column under the pointer (of a dots item shown) is ringed and its values
 * shown beside it. */
export function ColumnsOverlay({ t, image, items, found: withFound = false }: {
  t: Transform; image: string | undefined; items: DatasetSummary[]; found?: boolean;
}) {
  const files = useDatasets((s) => s.files);
  const st = useColumns((s) => (image ? s.byId[image] : undefined));
  const imageDs = useMemo(() => (image ? allDatasets(files).find((d) => d.id === image) : undefined), [files, image]);
  const props = useLayers((s) => s.props);
  const foundProps = (image && props[foundLayerKey(image)]) || DEFAULT_LAYER;
  const drawn = useMemo(
    () => items.flatMap((item) => {
      const info = itemInfo(files, item);
      return info && info.columns.parent_id === image ? [{ item, info }] : [];
    }),
    [items, files, image],
  );
  const shown = drawn.filter((x) => itemProps(props[x.item.id], x.info).visible);
  const columnsDs = drawn[0]?.info.columns;
  useEffect(() => {
    if (shown.length && columnsDs) ensureTable(columnsDs);
  }, [shown.length, columnsDs]);
  const tableEntry = useColumns((s) => (columnsDs ? s.tables[columnsDs.id] : undefined));
  const table = tableEntry?.points;
  const found = withFound && foundProps.visible ? st?.found : null;

  // A few screen pixels, less when the columns are close on screen.
  const spacing = (st?.settings.distance ?? 4) * t.zoom;
  const r = Math.min(3, Math.max(1, spacing * 0.22)) / t.zoom;
  const rFound = r * (foundProps.size ?? 1);
  const foundPath = useMemo(() => (found ? circles(found, rFound) : ""), [found, rFound]);

  // The column under the pointer, in the topmost dots item shown that has it.
  const dots = shown.filter((x) => x.info.kind === "dots");
  const owner = (i: number) => [...dots].reverse().find((x) => inGroup(table!, x.info.group, i));
  const hit = dots.length && table && t.pointer
    ? nearestColumn(table, t.pointer.x, t.pointer.y, Math.max(st?.settings.distance ?? 4, 3 / t.zoom), (i) => !!owner(i))
    : -1;
  const hitOwner = hit >= 0 ? owner(hit) : undefined;
  const hitSize = hitOwner ? (itemProps(props[hitOwner.item.id], hitOwner.info).size ?? 1) : 1;
  const sub = hit >= 0 && columnsDs ? sublatticesOf(files, columnsDs)[table!.sublattice[hit]] : undefined;

  if (!foundPath && !(shown.length && table)) return null;
  const [x0, y0] = t.toScreen(0, 0);
  const ordered = [...shown.filter((x) => x.info.kind === "bonds"), ...dots];
  return (
    <>
      <g className="columns-overlay" transform={`translate(${x0} ${y0}) scale(${t.zoom})`}>
        {foundPath && <path className="columns-found" d={foundPath} fill={foundProps.colour ?? COLOUR.found} opacity={foundProps.opacity} />}
        {table && tableEntry &&
          ordered.map(({ item, info }) =>
            info.kind === "bonds" ? (
              <ItemBonds key={item.id} table={table} item={item} info={info} />
            ) : (
              <ItemDots key={item.id} table={table} tableKey={tableEntry.key} item={item} info={info} r={r} />
            ),
          )}
        {hit >= 0 && <path className="columns-hovered" d={circles([[table!.y[hit], table!.x[hit]]], Math.max(r * hitSize * 1.2, r * 1.9))} />}
      </g>
      {hit >= 0 && (
        <ColumnTip
          t={t}
          p={table!}
          i={hit}
          px={imageDs ? pixelSize(imageDs) : null}
          by={hitOwner ? itemProps(props[hitOwner.item.id], hitOwner.info).colourBy : undefined}
          sublattice={sub?.title}
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
