import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { colourFrame, fetchArray, fetchFrame, fetchStackFrame, type Frame } from "../api/arrays";
import { api, dataVersion, type DatasetSummary, type MatchResult } from "../api/client";
import { ContextMenu, type MenuItem } from "../components/ContextMenu";
import { ImageView, type AxisCal, type Interaction, type Pt, type Transform } from "../components/ImageView";
import { AnnotationsToggle } from "../components/AnnotationsToggle";
import { ContrastControls } from "../components/ContrastControls";
import { FftBoxOverlay } from "../components/FftBoxOverlay";
import { LoupeToggle } from "../components/Loupe";
import { LineOverlay } from "../components/LineOverlay";
import { RoiOverlay } from "../components/RoiOverlay";
import { useCalibration } from "../state/calibration";
import { planeOf, useCentring } from "../state/centring";
import { chosenFor, isPolar, perAngstrom, useOrientation } from "../state/orientation";
import { MatchPanel } from "../components/MatchPanel";
import { useViewTemplate } from "../state/viewTemplate";
import { StructureScene } from "../components/StructureScene";
import { diskOptionsOf, useDisks, type Disk } from "../state/disks";
import { useLayers } from "../state/layers";
import { useProfiles } from "../state/profiles";
import { useProcessShown, useUi } from "../state/ui";
import { useFftBoxTool } from "./useFftBoxTool";
import { useLineTool } from "./useLineTool";
import { useNavigatorTools } from "./si/useNavigatorTools";
import { autoContrast } from "./useImageDisplay";
import type { ImageGeometry } from "./geometry";
import { ImageToolButtons, LayeredViewer, type ExtraLayers, type StackLayer } from "../components/LayeredViewer";
import { useLayersTab } from "../state/layersTab";
import { backgroundOptionsOf, usePatternBackground } from "../state/patternBackground";
import { bvmVersion, fitOf, latticeGrid, nearestPeak, peaksOnLattice, toleranceOf, useStrain, type StrainLattice } from "../state/strain";
import { formatValue } from "../format";
import { useShortcuts } from "../keyboard";
import { allDatasets, useDatasets } from "../state/datasets";
import { useDisplay } from "../state/display";
import {
  ROW_HEIGHTS, SCAN_BVM, SCAN_MAX, SCAN_MEAN, isScanSource, apertureR, cursorName, detectorForRequest, groupOf, imageRequest, responseLabel, sizeApertures, initialState, latticePoints, moveDetector, newCursor, useFourD,
  type Detector, type DetectorGroup, type FourDState, type Lattice, type OverlayLayer, type PatternSource, type Response, type ScanCursor,
} from "../state/fourd";
import { containsPoint, shapeToRegion, useSi, useSiState } from "../state/si";
import { imageGeometry } from "./geometry";
import { logFrame } from "./LineScanView";

const HANDLE_PX = 8;
// Bragg disks found on a pattern (their layer's default colour), and a dot's radius (screen px).
const DISK_COLOR = "#ffffff";
const DOT_PX = 2.5;
// The direct beam: where it was found, and where the descan plane puts it.
const BEAM_FOUND_COLOR = "#ff3b30";
const BEAM_PLANE_COLOR = "#00e5ff";
// Calibration: the annulus searched and the points found (yellow), the ellipse fitted (green).
const RING_COLOR = "#ffd60a";
const RING_FIT_COLOR = "#30d158";
const ZONE_COLOR = "#ff6bd6";
// The frames around the virtual image and the pattern in their group's / source's colour.
const FRAME_WIDTH = 2;
const FRAME_OPACITY = 0.6;
/** How close (screen px) a press must be to a cursor to grab it. */
const GRAB_PX = 8;
// Arrow keys and W/A/S/D move by one pixel, five with Shift.
const STEP = 1;
const SHIFT_STEP = 5;

export const SHAPES: { id: Detector["shape"]; label: string; title: string }[] = [
  { id: "annulus", label: "Annulus", title: "A ring: e.g. annular dark field" },
  { id: "rectangle", label: "Rectangle", title: "A rectangle" },
  { id: "point", label: "Point", title: "One pixel" },
];

/** A detector's kind as written in labels: discs (apertures) are a circle,
 * circles, an array, or circles with an array. */
export function shapeLabel(d: Detector): string {
  if (d.shape !== "apertures") return d.shape;
  const k = (d.apertures ?? []).length;
  return d.lattice ? (k ? "circles + array" : "array") : k > 1 ? "circles" : "circle";
}
const LATTICE_COLOR = "#8cff5a";
// The strain lattice on the Bragg vector map: its points (tolerance circles),
// the map's peaks on it and off it.
const STRAIN_COLOR = "#5ac8fa";
const PEAK_ON_COLOR = "#ffffff";
const PEAK_OFF_COLOR = "#ff9f0a";
/** Detectors' areas are filled in their group's colour, this opaque. */
const DETECTOR_FILL = 0.25;
const BAD_COLOR = "#ff3b30";
export const RESPONSES: { id: Response; label: string; title: string }[] = [
  { id: "sum", label: "Sum", title: "The total intensity inside the detector" },
  { id: "mean", label: "Mean", title: "The mean intensity inside the detector" },
  { id: "max", label: "Max", title: "The brightest pixel inside the detector" },
  { id: "com", label: "COM", title: "The centre of mass of the pixels inside the detector from the pattern's centre, as in the data (no descan or rotation), in colour: the direction round the colour wheel, the magnitude as brightness" },
  { id: "com_x", label: "COM x", title: "The centre of mass of the pixels inside the detector along the pattern's x, from its centre, as in the data" },
  { id: "com_y", label: "COM y", title: "The centre of mass of the pixels inside the detector along the pattern's y, from its centre, as in the data" },
  { id: "icom", label: "iCOM", title: "The centre of mass of the pixels inside the detector, integrated (the projected potential, up to scale, once the data is centred and in the scan's frame)" },
];

const asFrame = ({ data, shape }: { data: Float32Array; shape: number[] }): Frame => ({ data, height: shape[0], width: shape[1], bin: 1 });
/** A virtual image: a colour one ("com": direction and magnitude, [2, h, w]) drawn as a DPC object is. */
const asVirtualFrame = ({ data, shape }: { data: Float32Array; shape: number[] }): Frame =>
  shape.length === 3 ? colourFrame(data, shape[1], shape[2]) : asFrame({ data, shape });
const post = (url: string, body: unknown, signal?: AbortSignal) =>
  fetchArray(url, signal, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

/** The latest of a frame that changes quickly (a detector or cursor being
 * dragged): one request at a time, the newest wanted one next; the last frame
 * stays on screen meanwhile. */
function useLatest(key: string, run: () => Promise<Frame>, enabled = true) {
  const [shown, setShown] = useState<{ key: string; frame: Frame } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const busy = useRef(false);
  const wanted = useRef<{ key: string; run: () => Promise<Frame> } | null>(null);
  const alive = useRef(true);
  useEffect(() => () => { alive.current = false; }, []);
  const pump = useCallback(() => {
    const w = wanted.current;
    if (!w || busy.current) return;
    busy.current = true;
    w.run()
      .then((frame) => {
        if (!alive.current) return;
        setShown({ key: w.key, frame });
        setError(null);
      })
      .catch((e) => alive.current && setError((e as Error).message))
      .finally(() => {
        busy.current = false;
        if (wanted.current && wanted.current.key !== w.key) pump();
      });
  }, []);
  useEffect(() => {
    wanted.current = { key, run };
    if (enabled) pump();
  }, [key, enabled]); // eslint-disable-line react-hooks/exhaustive-deps
  return { frame: shown?.frame ?? null, current: shown?.key === key, shownKey: shown?.key, error };
}

/** Frames of several things at once (the grid's virtual images or patterns),
 * each under a key that changes when it must be fetched again: only the
 * changed ones are fetched (by ``fetchMany``, all in one go), one request at
 * a time, the rest kept on screen meanwhile. Off when ``enabled`` is false. */
function useFrameSet<T extends { id: string; key: string }>(items: T[], fetchMany: (todo: T[]) => Promise<Frame[]>, enabled: boolean) {
  const [frames, setFrames] = useState<Record<string, { key: string; frame: Frame }>>({});
  const [error, setError] = useState<string | null>(null);
  const shown = useRef(frames);
  const wanted = useRef(items);
  wanted.current = items;
  const fetcher = useRef(fetchMany);
  fetcher.current = fetchMany;
  const busy = useRef(false);
  const alive = useRef(true);
  useEffect(() => () => { alive.current = false; }, []);
  const pump = useCallback(() => {
    if (busy.current) return;
    const todo = wanted.current.filter((i) => shown.current[i.id]?.key !== i.key);
    if (!todo.length) return;
    busy.current = true;
    let ok = true;
    fetcher.current(todo)
      .then((got) => {
        if (!alive.current) return;
        const next = { ...shown.current };
        todo.forEach((it, k) => { next[it.id] = { key: it.key, frame: got[k] }; });
        shown.current = next;
        setFrames(next);
        setError(null);
      })
      .catch((e) => {
        ok = false;
        if (alive.current) setError((e as Error).message);
      })
      .finally(() => {
        busy.current = false;
        // What changed meanwhile (a failed request is tried again only on the next change).
        if (ok) pump();
      });
  }, []);
  const signature = enabled ? items.map((i) => `${i.id}=${i.key}`).join("|") : "";
  useEffect(() => {
    if (enabled) pump();
  }, [signature]); // eslint-disable-line react-hooks/exhaustive-deps
  return { frames, error };
}

/** The viewer's tools (line, FFT box, regions) on plain image views (the
 * list's rows), as ``LayeredViewer`` has them: with a tool on, its
 * interaction; else ``own``, a press also selecting the region under it.
 * Lines, boxes and regions are kept under ``key`` (shared with the other
 * layouts' views). ``overlay`` draws them. */
function useViewTools(key: string, w: number, h: number, own: Interaction) {
  const imageTool = useProfiles((s) => s.imageTool);
  const line = useLineTool(key);
  const box = useFftBoxTool(key, w, h);
  const regionTool = imageTool === "rect" || imageTool === "ellipse" || imageTool === "polygon" ? imageTool : null;
  const regions = useNavigatorTools(key, w, h, false, regionTool ?? "rect");
  const state = useSiState(key);
  const hidden = useUi((s) => !!s.annotationsHidden[key]);
  const interaction: Interaction =
    imageTool === "line" ? line
      : imageTool === "fft" ? box
        : regionTool ? regions.interaction
          : {
              ...own,
              onDown: (p, e, t) => {
                if (!hidden) regions.regionPress(p, t);
                return own.onDown?.(p, e, t) ?? false;
              },
            };
  const overlay = (t: Transform, calX: AxisCal, calY: AxisCal): ReactNode =>
    hidden ? null : (
      <>
        <FftBoxOverlay t={t} view={key} />
        <RoiOverlay t={t} state={state} cursorSize={1} draft={regions.draft} hover={regions.hover} />
        <LineOverlay t={t} view={key} calX={calX} calY={calY} />
      </>
    );
  return { interaction, overlay };
}

export function segmented<T extends string>(
  label: string, items: { id: T; label: string; title?: string; disabled?: boolean }[], value: T, set: (v: T) => void,
) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {items.map((i) => (
        <button
          key={i.id}
          className={value === i.id ? "active" : undefined}
          aria-pressed={value === i.id}
          title={i.title}
          disabled={i.disabled}
          onClick={() => set(i.id)}
        >
          {i.label}
        </button>
      ))}
    </div>
  );
}

type Handle = { id: string; x: number; y: number };

/** The detector's handles (pattern pixels): its centre, and what resizes it. */
function handles(d: Detector): Handle[] {
  switch (d.shape) {
    case "circle":
      return [{ id: "c", x: d.cx, y: d.cy }, { id: "r", x: d.cx + d.r, y: d.cy }];
    case "annulus": {
      const k = Math.SQRT1_2;
      return [{ id: "c", x: d.cx, y: d.cy }, { id: "r", x: d.cx + d.r, y: d.cy }, { id: "ri", x: d.cx + d.r_inner * k, y: d.cy - d.r_inner * k }];
    }
    case "rectangle":
      return [
        { id: "x0y0", x: d.x0, y: d.y0 }, { id: "x1y0", x: d.x1, y: d.y0 },
        { id: "x1y1", x: d.x1, y: d.y1 }, { id: "x0y1", x: d.x0, y: d.y1 },
      ];
    default:
      return [{ id: "c", x: Math.floor(d.cx) + 0.5, y: Math.floor(d.cy) + 0.5 }];
  }
}

/** The clicked aperture under ``p``, if any (its index). */
function apertureAt(d: Detector, p: Pt): number {
  const list = d.apertures ?? [];
  for (let i = list.length - 1; i >= 0; i--) if (Math.hypot(p.x - list[i].x, p.y - list[i].y) <= apertureR(d, list[i])) return i;
  return -1;
}

/** The g-vector array's markers: its origin and the tips of g1 and g2. */
function latticeMarkers(l: Lattice): Handle[] {
  return [
    { id: "o", x: l.ox, y: l.oy },
    { id: "g1", x: l.ox + l.g1x, y: l.oy + l.g1y },
    { id: "g2", x: l.ox + l.g2x, y: l.oy + l.g2y },
  ];
}

function inside(d: Detector, p: Pt): boolean {
  if (d.shape === "apertures") return apertureAt(d, p) >= 0;
  if (d.shape === "rectangle") {
    return p.x >= Math.min(d.x0, d.x1) && p.x <= Math.max(d.x0, d.x1) && p.y >= Math.min(d.y0, d.y1) && p.y <= Math.max(d.y0, d.y1);
  }
  if (d.shape === "point") return Math.floor(p.x) === Math.floor(d.cx) && Math.floor(p.y) === Math.floor(d.cy);
  return Math.hypot(p.x - d.cx, p.y - d.cy) <= d.r;
}

/** A button showing the current choice (a colour dot and its name) that
 * lists the choices under it. */
function ListPicker({ label, color, muted, title, items }: { label: string; color?: string; muted?: boolean; title: string; items: MenuItem[] }) {
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);
  return (
    <>
      <button
        className={`secondary fourd-source${muted ? " muted" : ""}`}
        title={title}
        aria-haspopup="menu"
        onMouseDown={(e) => menu && e.stopPropagation()}
        onClick={(e) => {
          if (menu) return setMenu(null);
          const r = e.currentTarget.getBoundingClientRect();
          setMenu({ x: r.left, y: r.bottom + 4 });
        }}
      >
        {color && <span className="fourd-swatch" style={{ background: color }} />}
        <span className="label">{label}</span>
        <span aria-hidden>▾</span>
      </button>
      {menu && <ContextMenu x={menu.x} y={menu.y} onClose={() => setMenu(null)} items={items} />}
    </>
  );
}

/** A detector's outline on the pattern (apertures: every disc, ``d`` as sent
 * to the backend), in ``color``, the area integrated over filled translucent. */
function outline(t: Transform, d: Detector, color: string, width: number): ReactNode {
  const z = t.zoom;
  const stroke = { fill: color, fillOpacity: DETECTOR_FILL, stroke: color, strokeWidth: width };
  const [cx, cy] = t.toScreen(d.cx, d.cy);
  switch (d.shape) {
    case "circle":
      return <circle cx={cx} cy={cy} r={d.r * z} {...stroke} />;
    case "annulus": {
      // The ring between the circles filled (even-odd), each circle outlined.
      const circle = (r: number) => `M ${cx - r} ${cy} a ${r} ${r} 0 1 0 ${2 * r} 0 a ${r} ${r} 0 1 0 ${-2 * r} 0 Z`;
      return (
        <>
          <path d={`${circle(d.r * z)} ${circle(d.r_inner * z)}`} fill={color} fillOpacity={DETECTOR_FILL} fillRule="evenodd" stroke="none" />
          <circle cx={cx} cy={cy} r={d.r * z} fill="none" stroke={color} strokeWidth={width} />
          <circle cx={cx} cy={cy} r={d.r_inner * z} fill="none" stroke={color} strokeWidth={width} strokeDasharray="4 3" />
        </>
      );
    }
    case "rectangle": {
      const [ax, ay] = t.toScreen(Math.min(d.x0, d.x1), Math.min(d.y0, d.y1));
      const [bx, by] = t.toScreen(Math.max(d.x0, d.x1), Math.max(d.y0, d.y1));
      return <rect x={ax} y={ay} width={bx - ax} height={by - ay} {...stroke} />;
    }
    case "apertures":
      return (d.apertures ?? []).map((a, i) => {
        const [x, y] = t.toScreen(a.x, a.y);
        return <circle key={i} cx={x} cy={y} r={apertureR(d, a) * z} {...stroke} />;
      });
    default: {
      const [ax, ay] = t.toScreen(Math.floor(d.cx), Math.floor(d.cy));
      const [bx, by] = t.toScreen(Math.floor(d.cx) + 1, Math.floor(d.cy) + 1);
      return <rect x={ax} y={ay} width={Math.max(3, bx - ax)} height={Math.max(3, by - ay)} {...stroke} />;
    }
  }
}

/** ``p`` is within ``tol`` (pattern pixels) of a detector's outline (``d``
 * as sent: apertures, any disc's edge). */
function onOutline(d: Detector, p: Pt, tol: number): boolean {
  const ring = (x: number, y: number, r: number) => Math.abs(Math.hypot(p.x - x, p.y - y) - r) <= tol;
  switch (d.shape) {
    case "circle":
      return ring(d.cx, d.cy, d.r);
    case "annulus":
      return ring(d.cx, d.cy, d.r) || ring(d.cx, d.cy, d.r_inner);
    case "apertures":
      return (d.apertures ?? []).some((a) => ring(a.x, a.y, apertureR(d, a)));
    case "rectangle": {
      const [x0, x1] = [Math.min(d.x0, d.x1), Math.max(d.x0, d.x1)];
      const [y0, y1] = [Math.min(d.y0, d.y1), Math.max(d.y0, d.y1)];
      const inX = p.x >= x0 - tol && p.x <= x1 + tol;
      const inY = p.y >= y0 - tol && p.y <= y1 + tol;
      return (inY && (Math.abs(p.x - x0) <= tol || Math.abs(p.x - x1) <= tol)) || (inX && (Math.abs(p.y - y0) <= tol || Math.abs(p.y - y1) <= tol));
    }
    default:
      return Math.hypot(p.x - (Math.floor(d.cx) + 0.5), p.y - (Math.floor(d.cy) + 0.5)) <= tol + 0.5;
  }
}

/** The detector as text, in the pattern's calibration. */
export function detectorText(d: Detector, scale: number, units: string): string {
  const n = (v: number) => formatValue(v * scale);
  switch (d.shape) {
    case "circle":
      return `r ${n(d.r)} ${units}`;
    case "annulus":
      return `${n(d.r_inner)}–${n(d.r)} ${units}`;
    case "rectangle":
      return `${n(Math.abs(d.x1 - d.x0))} × ${n(Math.abs(d.y1 - d.y0))} ${units}`;
    case "apertures": {
      const k = (d.apertures ?? []).length;
      return `${k > 1 ? `${k} placed · ` : ""}r ${n(d.r)} ${units}`;
    }
    default:
      return `pixel (${Math.floor(d.cx)}, ${Math.floor(d.cy)})`;
  }
}

/** A 4D-STEM dataset's view state (detector, cursor, region…), shared by the
 * view and the Virtual image process, with its scan and pattern geometry. */
export function useFourDState(ds: DatasetSummary) {
  const scanG = imageGeometry(ds, "navigation");
  const patternG = imageGeometry(ds, "signal");
  const init = useMemo(
    () => initialState([scanG.width, scanG.height], [patternG.width, patternG.height]),
    [ds.id, scanG.width, scanG.height, patternG.width, patternG.height], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const st: FourDState = useFourD((s) => s.byId[ds.id]) ?? init;
  const patchStore = useFourD((s) => s.patch);
  const patch = useCallback((p: Partial<FourDState>) => patchStore(ds.id, init, p), [patchStore, ds.id, init]);
  /** Group ``id`` changed (from the store's latest state, so quick changes add up). */
  const patchGroup = useCallback(
    (id: string, p: Partial<DetectorGroup>) => {
      const cur = useFourD.getState().byId[ds.id] ?? init;
      patchStore(ds.id, init, { groups: cur.groups.map((g) => (g.id === id ? { ...g, ...p } : g)) });
    },
    [patchStore, ds.id, init],
  );
  const group = groupOf(st);
  return { scanG, patternG, st, patch, group, det: group.detector, patchGroup };
}

/** A 4D-STEM dataset: its virtual image (left) through a virtual detector
 * drawn on the diffraction pattern (right), live. A cursor or a region on the
 * scan picks the pattern shown (or the mean or maximum over the whole scan).
 * Arrow keys move the cursor (scan focused) or the detector (pattern
 * focused); W/A/S/D move the detector; Shift moves 5 pixels. ``vectors``:
 * its Bragg vectors are selected (in the dataset pane): their disks shown. */
export function FourDView({ ds, vectors = false }: { ds: DatasetSummary; vectors?: boolean }) {
  const { scanG, patternG, st, patch, group, det } = useFourDState(ds);
  const version = dataVersion(ds);
  // How the marks over the pattern and the scan show (Layers tab).
  const overlays = useFourD((s) => s.overlays[ds.id]);
  const setOverlay = useFourD((s) => s.setOverlay);
  const ovOf = (key: string): OverlayLayer => overlays?.[key] ?? {};
  // A group's detector shows with the group (one eye, in the process and the Layers tab).
  const detShown = (g: DetectorGroup) => g.visible;
  const detOpacity = (g: DetectorGroup) => ovOf(`det:${g.id}`).opacity ?? 1;
  const cursorShown = (c: ScanCursor) => ovOf(`cursor:${c.id}`).visible ?? true;

  // ---------- the image on the scan: live through the detector, or a map made from the data ----------
  const files = useDatasets((s) => s.files);
  // Every image made from this data (at any depth: saved virtual images,
  // origin and beam-shift maps, DPC, iCOM, masks…) of the scan's shape, in
  // the dataset pane's order.
  const saved = useMemo(() => {
    const all = allDatasets(files);
    const byId = new Map(all.map((d) => [d.id, d]));
    const under = (d: DatasetSummary) => {
      let p = d.parent_id;
      for (let i = 0; p && i < 8; i++) {
        if (p === ds.id) return true;
        p = byId.get(p)?.parent_id ?? null;
      }
      return false;
    };
    const [w, h] = ds.navigation_shape;
    return all.filter((d) =>
      d.kind === "image" && !d.lazy && d.navigation_shape.length === 0 &&
      d.signal_shape[0] === w && d.signal_shape[1] === h && under(d));
  }, [files, ds.id, ds.navigation_shape]);
  const shownSaved = saved.find((d) => d.id === st.shown) ?? null;
  const shownColour = shownSaved?.dpc?.output === "colour";
  // A digital dark field is made again when the disks or the centring change.
  const ddfVersion = useMemo(() => {
    const all = allDatasets(files);
    const v = all.find((d) => d.origin === "disks" && d.parent_id === ds.id);
    const maps = all.filter((d) => d.origin === "origin-map" && d.origin_map?.source_id === ds.id);
    const centred = ds.calibration?.enabled?.origin ? maps.map((m) => `${m.id}@${dataVersion(m)}`).join() : "off";
    return `${v ? `${v.id}@${dataVersion(v)}` : "none"}|${centred}`;
  }, [files, ds]);
  const keyOf = (req: ReturnType<typeof imageRequest>) =>
    `${version}|${JSON.stringify(req)}${req.source === "vectors" ? `|${ddfVersion}` : ""}`;
  const hasVectors = !ddfVersion.startsWith("none|");
  const sent = imageRequest(group, patternG.width, patternG.height, hasVectors);
  const virtualKey = shownSaved ? `saved|${shownSaved.id}|${dataVersion(shownSaved)}` : keyOf(sent);
  const grid = st.layout === "grid";
  const list = st.layout === "list";
  // Merge: the visible groups' images as one colour composite, each in its colour.
  const merge = st.layout === "merge";
  // Grid, list and merge show every visible group's image (each in its colour).
  const multi = grid || list || merge;
  const virtual = useLatest(virtualKey, () =>
    shownSaved
      ? fetchStackFrame(shownSaved, {})
      : post(`/api/datasets/${ds.id}/4d/virtual`, sent).then(asVirtualFrame),
  !multi);
  const bad = useFourD((s) => s.bad[ds.id]) ?? null;

  // ---------- the pattern: a cursor's, a region's, or over the whole scan ----------
  const scanKey = `${ds.id}:virtual`;
  const scanRegions = useSiState(scanKey);
  const rois = scanRegions.rois.filter((r) => r.visible);
  const selectedRoi = useSi((s) => s.bySi[scanKey]?.selected ?? null);
  const { cursors } = st;
  const activeCursor = cursors.find((c) => c.id === st.source) ?? cursors.find((c) => c.id === st.lastCursor) ?? cursors[0];
  const sourceRoi = rois.find((r) => r.id === st.source) ?? null;
  // A source gone (its region deleted or hidden, the disks no longer found) falls back to the cursor.
  const source: PatternSource =
    sourceRoi || st.source === SCAN_MEAN || st.source === SCAN_MAX || (st.source === SCAN_BVM && hasVectors) ? st.source : activeCursor.id;
  // A region selected on the scan (clicked or just drawn) shows its pattern.
  useEffect(() => {
    if (selectedRoi && selectedRoi !== latest.current.source) patch({ source: selectedRoi });
  }, [selectedRoi]); // eslint-disable-line react-hooks/exhaustive-deps
  // While the Background process is shown (on this data, "Show on the
  // patterns"): every pattern shown with its background subtracted.
  const bgProcessShown = useProcessShown("background");
  const bgFocused = useDatasets((s) => s.focusId === ds.id);
  const bgPreview = usePatternBackground((s) => s.preview[ds.id] ?? true);
  const bgOpts = usePatternBackground((s) => backgroundOptionsOf(s, ds.id));
  const bgOn = bgProcessShown && bgFocused && bgPreview;
  // The radial median's centre follows the centring.
  const bgTag = bgOn ? `|bg|${JSON.stringify(bgOpts)}|${ddfVersion}` : "";
  const bgPattern = (spec: object) => post(`/api/datasets/${ds.id}/4d/background/preview`, { options: bgOpts, ...spec }).then(asFrame);
  // The Bragg vector map (pattern pixels): made again when the disks or the centring change; no background.
  const bvmKey = `${version}|${SCAN_BVM}|${ddfVersion}`;
  const bvmPattern = () => fetchArray(`/api/datasets/${ds.id}/4d/bvm-pattern`).then(asFrame);
  const patternKey = source === SCAN_BVM ? bvmKey : `${version}|${
    sourceRoi ? `roi|${JSON.stringify(sourceRoi.shape)}` : source === SCAN_MEAN || source === SCAN_MAX ? source : `cursor|${activeCursor.x},${activeCursor.y}`
  }${bgTag}`;
  const pattern = useLatest(patternKey, () => {
    if (source === SCAN_BVM) return bvmPattern();
    if (bgOn) {
      return bgPattern(sourceRoi ? { region: shapeToRegion(sourceRoi.shape) }
        : source === SCAN_MEAN || source === SCAN_MAX ? { scan: source === SCAN_MAX ? "max" : "mean" }
        : { x: activeCursor.x, y: activeCursor.y });
    }
    if (sourceRoi) return post(`/api/datasets/${ds.id}/4d/pattern`, { region: shapeToRegion(sourceRoi.shape), reduce: "mean" }).then(asFrame);
    if (source === SCAN_MEAN || source === SCAN_MAX) return fetchFrame(ds.id, { projection: source === SCAN_MAX ? "max-pattern" : "mean-pattern" });
    return fetchArray(`/api/datasets/${ds.id}/4d/pattern?x=${activeCursor.x}&y=${activeCursor.y}`).then(asFrame);
  }, !grid);
  const logKey = `${ds.id}:pattern`;
  const log = useDisplay((s) => s.imageLog[logKey]) ?? true;
  const setImageLog = useDisplay((s) => s.setImageLog);
  const drawnPattern = useMemo(() => pattern.frame && (log ? logFrame(pattern.frame) : pattern.frame), [pattern.frame, log]);

  // ---------- the grid: a virtual image per visible group, a pattern per cursor and region ----------
  const visibleGroups = st.groups.filter((g) => g.visible);
  const viItems = visibleGroups.map((g) => {
    const req = imageRequest(g, patternG.width, patternG.height, hasVectors);
    return { id: g.id, key: keyOf(req), req };
  });
  const viSet = useFrameSet(
    viItems,
    (todo) =>
      post(`/api/datasets/${ds.id}/4d/virtuals`, { items: todo.map((i) => i.req) }).then(({ data, shape, meta }) => {
        // Each image's channels one after the other (a colour one: 2).
        const [, h, w] = shape;
        const channels = (meta.channels as number[] | undefined) ?? todo.map(() => 1);
        let at = 0;
        return channels.map((c) => {
          const part = data.subarray(at * h * w, (at + c) * h * w);
          at += c;
          return c === 2 ? colourFrame(part, h, w) : { data: part, height: h, width: w, bin: 1 };
        });
      }),
    multi,
  );
  const scanSource = isScanSource(source) ? source : null;
  const dpItems = [
    ...cursors.map((c) => ({
      id: c.id, key: `${version}|cursor|${c.x},${c.y}${bgTag}`,
      run: () => (bgOn ? bgPattern({ x: c.x, y: c.y }) : fetchArray(`/api/datasets/${ds.id}/4d/pattern?x=${c.x}&y=${c.y}`).then(asFrame)),
    })),
    ...rois.map((r) => ({
      id: r.id, key: `${version}|roi|${JSON.stringify(r.shape)}${bgTag}`,
      run: () => (bgOn
        ? bgPattern({ region: shapeToRegion(r.shape) })
        : post(`/api/datasets/${ds.id}/4d/pattern`, { region: shapeToRegion(r.shape), reduce: "mean" }).then(asFrame)),
    })),
    ...(scanSource === SCAN_BVM
      ? [{ id: scanSource, key: bvmKey, run: bvmPattern }]
      : scanSource
      ? [{
          id: scanSource, key: `${version}|${scanSource}${bgTag}`,
          run: () => (bgOn
            ? bgPattern({ scan: scanSource === SCAN_MAX ? "max" : "mean" })
            : fetchFrame(ds.id, { projection: scanSource === SCAN_MAX ? "max-pattern" : "mean-pattern" })),
        }]
      : []),
  ];
  const dpSet = useFrameSet(dpItems, (todo) => Promise.all(todo.map((i) => i.run())), grid);
  const dpDrawn = useMemo(() => {
    const out: Record<string, Frame> = {};
    for (const [id, v] of Object.entries(dpSet.frames)) out[id] = log ? logFrame(v.frame) : v.frame;
    return out;
  }, [dpSet.frames, log]);
  // ---------- Bragg disks on the patterns shown ----------
  // While the Disk detection process is shown (the Process tab on this data's
  // Disk detection): found live with its settings ("Show on the patterns"),
  // else, once every pattern's are found, those at each cursor. Their layer
  // (Layers tab) shows or hides them, also outside the process; selecting the
  // Bragg vectors shows them.
  const diskProcessShown = useProcessShown("disks");
  const focused = useDatasets((s) => s.focusId === ds.id);
  // The Calibration and Strain processes are the Bragg vectors': shown while they (of this data) are focused.
  const braggFocused = useDatasets((s) => {
    const f = s.focusId ? allDatasets(s.files).find((x) => x.id === s.focusId) : undefined;
    return f?.origin === "disks" && f.parent_id === ds.id;
  });
  // ---------- the Strain process: its lattice on the Bragg vector map ----------
  const strainShown = useProcessShown("strain") && braggFocused && hasVectors;
  const strainLattice = useStrain((s) => s.lattice[ds.id]);
  const strainPeaks = useStrain((s) => s.peaks[ds.id]?.list);
  const strainTol = useStrain((s) => toleranceOf(s, ds.id));
  // Opening it shows the map in the pattern pane (where the lattice is placed).
  useEffect(() => {
    if (strainShown && latest.current.source !== SCAN_BVM) patch({ source: SCAN_BVM });
  }, [strainShown]); // eslint-disable-line react-hooks/exhaustive-deps
  // The indexing and the fit tried at the cursors, drawn on their patterns.
  const strainFit = useStrain((s) => fitOf(s, ds.id));
  const strainPreviewOn = useStrain((s) => s.preview[ds.id] ?? true) && strainShown && !!strainLattice;
  const strainCursors = cursors.map((c) => [c.x, c.y] as [number, number]);
  const strainPreviewKey = strainPreviewOn
    ? `${bvmVersion(files, ds)}|${ds.calibration ? JSON.stringify(ds.calibration.enabled ?? {}) : ""}|${JSON.stringify([strainLattice, strainTol, strainFit, strainCursors])}`
    : "";
  type StrainAt = { disks: [number, number, number, number, boolean][]; points?: [number, number, number, number][]; count: number; residual: number | null };
  const [strainPreview, setStrainPreview] = useState<{ key: string; at: Record<string, StrainAt> } | null>(null);
  useEffect(() => {
    if (!strainPreviewKey) return;
    let stale = false;
    const id = setTimeout(async () => {
      try {
        const res = await fetch(`/api/datasets/${ds.id}/4d/strain/preview`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ lattice: strainLattice, options: { tolerance: strainTol, ...strainFit }, positions: strainCursors }),
        });
        const body = await res.json();
        if (!stale && res.ok) setStrainPreview({ key: strainPreviewKey, at: body });
      } catch {
        /* the next change tries again */
      }
    }, 150);
    return () => {
      stale = true;
      clearTimeout(id);
    };
  }, [strainPreviewKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const diskOpen = diskProcessShown && focused;
  const diskPreview = useDisks((s) => s.preview[ds.id] ?? true) && diskOpen;
  const disksLayer = ovOf("disks");
  const disksVisible = disksLayer.visible ?? diskOpen;
  useEffect(() => {
    if (vectors) setOverlay(ds.id, "disks", { visible: true });
  }, [vectors, ds.id, setOverlay]);
  const diskOpts = useDisks((s) => diskOptionsOf(s, ds.id));
  const shownSourceIds = grid ? dpItems.map((i) => i.id) : [source];
  const specOf = (id: string): object | null => {
    const c = cursors.find((k) => k.id === id);
    if (c) return { x: c.x, y: c.y };
    const r = rois.find((k) => k.id === id);
    if (r) return { region: shapeToRegion(r.shape) };
    if (id === SCAN_MEAN || id === SCAN_MAX) return { scan: id === SCAN_MAX ? "max" : "mean" };
    return null;
  };
  // Disks are found on patterns, not on the Bragg vector map.
  const previewSpecs = shownSourceIds.flatMap((id) => {
    const spec = specOf(id);
    return spec ? [{ id, ...spec }] : [];
  });
  const previewKey = diskPreview ? `${version}|${JSON.stringify(diskOpts)}|${JSON.stringify(previewSpecs)}` : "";
  const [previewDisks, setPreviewDisks] = useState<{ key: string; r: number; disks: Record<string, Disk[]> } | null>(null);
  useEffect(() => {
    if (!previewKey) return;
    let stale = false;
    // After the settings rest a moment (typing, dragging a cursor).
    const id = setTimeout(async () => {
      try {
        const res = await fetch(`/api/datasets/${ds.id}/4d/disks/preview`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ options: diskOpts, sources: previewSpecs }),
        });
        const body = await res.json();
        if (!stale && res.ok) setPreviewDisks({ key: previewKey, r: body.template.r, disks: body.disks });
      } catch {
        /* the next change tries again */
      }
    }, 150);
    return () => {
      stale = true;
      clearTimeout(id);
    };
  }, [previewKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const vectorsDs = useMemo(
    () => allDatasets(files).find((d) => d.origin === "disks" && d.parent_id === ds.id) ?? null,
    [files, ds.id],
  );
  const vectorsKey = vectorsDs ? `${vectorsDs.id}@${dataVersion(vectorsDs)}` : "";
  const [storedDisks, setStoredDisks] = useState<Record<string, Disk[]>>({});
  const cursorKeys = cursors.map((c) => `${c.x},${c.y}`);
  useEffect(() => {
    if (!vectorsKey || diskPreview || !disksVisible) return;
    for (const xy of cursorKeys) {
      const k = `${vectorsKey}|${xy}`;
      if (storedDisks[k]) continue;
      const [x, y] = xy.split(",");
      fetch(`/api/datasets/${ds.id}/4d/disks?x=${x}&y=${y}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((body) => body && setStoredDisks((s) => ({ ...s, [k]: body.disks })))
        .catch(() => undefined);
    }
  }, [vectorsKey, diskPreview, disksVisible, cursorKeys.join("|")]); // eslint-disable-line react-hooks/exhaustive-deps
  /** The disks to draw on the pattern of source ``id`` (and their radius), if any. */
  const disksOf = (id: string): { disks: Disk[]; r: number } | null => {
    if (!disksVisible) return null;
    if (diskPreview) {
      const p = previewDisks;
      return p?.disks[id] ? { disks: p.disks[id], r: p.r } : null;
    }
    const c = cursors.find((k) => k.id === id);
    const found = c && vectorsDs?.disks ? storedDisks[`${vectorsKey}|${c.x},${c.y}`] : undefined;
    return found ? { disks: found, r: vectorsDs!.disks!.template.r } : null;
  };
  const disksOverlay = (t: Transform, id: string): ReactNode => {
    const found = disksOf(id);
    if (!found) return null;
    const colour = disksLayer.colour ?? DISK_COLOR;
    // A circle of the disk's size, or a dot at its position.
    const dot = disksLayer.style === "dot";
    const dotR = DOT_PX * (disksLayer.size ?? 1);
    return (
      <g className="fourd-overlay" opacity={disksLayer.opacity ?? 1}>
        {found.disks.map(([ky, kx], i) => {
          const [x, y] = t.toScreen(kx, ky);
          return dot
            ? <circle key={i} cx={x} cy={y} r={dotR} fill={colour} stroke="#000" strokeWidth={0.5} />
            : <circle key={i} cx={x} cy={y} r={Math.max(3, found.r * t.zoom)} fill="none" stroke={colour} strokeWidth={1.25} />;
        })}
      </g>
    );
  };

  // ---------- the direct beam, while the Centring process is shown ----------
  // At each cursor's pattern: where it was found, and where the descan puts it.
  const centringShown = useProcessShown("centring") && focused;
  const plane = useCentring((s) => planeOf(s, ds.id));
  const shiftDs = useMemo(
    () => allDatasets(files).filter((d) => d.origin === "beam-shift" && d.parent_id === ds.id),
    [files, ds.id],
  );
  const beamKey = shiftDs.length === 2 ? `${shiftDs.map((d) => `${d.id}@${dataVersion(d)}`).join()}|${plane.mode}|${plane.corners}` : "";
  const [beamAt, setBeamAt] = useState<Record<string, { found: [number, number]; used: [number, number] }>>({});
  useEffect(() => {
    if (!beamKey || !centringShown) return;
    for (const xy of cursorKeys) {
      const k = `${beamKey}|${xy}`;
      if (beamAt[k]) continue;
      const [x, y] = xy.split(",");
      fetch(`/api/datasets/${ds.id}/4d/beam?x=${x}&y=${y}&mode=${plane.mode}&corners=${plane.corners}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((b) => b && setBeamAt((s) => ({ ...s, [k]: { found: b.found, used: b.used } })))
        .catch(() => undefined);
    }
  }, [beamKey, centringShown, cursorKeys.join("|")]); // eslint-disable-line react-hooks/exhaustive-deps
  const beamOverlay = (t: Transform, id: string): ReactNode => {
    if (!centringShown) return null;
    const c = cursors.find((k) => k.id === id);
    const b = c && beamAt[`${beamKey}|${c.x},${c.y}`];
    if (!b) return null;
    const cross = ([x, y]: [number, number], color: string, k: string) => {
      const [sx, sy] = t.toScreen(x, y);
      return (
        <g key={k} stroke={color} strokeWidth={1.5}>
          <line x1={sx - 8} x2={sx + 8} y1={sy} y2={sy} />
          <line x1={sx} x2={sx} y1={sy - 8} y2={sy + 8} />
        </g>
      );
    };
    return (
      <g className="fourd-overlay">
        {cross(b.found, BEAM_FOUND_COLOR, "found")}
        {plane.mode !== "none" && cross(b.used, BEAM_PLANE_COLOR, "used")}
      </g>
    );
  };

  // ---------- the templates chosen (Orientation mapping): a layer on the pattern ----------
  // The template nearest the way its Structure is seen (the Structure tab
  // below, or its own view), turned as the crystal is, at the pattern's scale.
  const chosenTemplates = useOrientation((s) => s.chosen[ds.id] ?? null);
  const templatesDs = chosenTemplates ? allDatasets(files).find((x) => x.id === chosenTemplates && x.templates) : undefined;
  const templateStructure = templatesDs ? allDatasets(files).find((x) => x.id === templatesDs.parent_id && x.structure) : undefined;
  const templateLayer = ovOf("template");
  const templateVisible = !!templatesDs && (templateLayer.visible ?? true) && perAngstrom(ds) !== null;
  const viewPattern = useViewTemplate(templatesDs?.id ?? null, templateStructure?.id ?? null, templateVisible);
  const templateColour = templateLayer.colour ?? ZONE_COLOR;
  const templateScale = useOrientation((s) => s.scale[ds.id] ?? 1);
  const zoneOverlay = (t: Transform): ReactNode => {
    const z = templateVisible ? viewPattern : null;
    const per = perAngstrom(ds);
    const [ky, kx] = ds.axes.filter((a) => !a.navigate);
    if (!z || !per || !kx || !ky) return null;
    return (
      <g className="fourd-overlay" opacity={templateLayer.opacity ?? 1}>
        {z.x.map((x, i) => {
          // View y is up; the pattern's rows go down.
          // Scaled about the axes' 0 (Orientation mapping's Scale).
          const f = templateScale;
          const [sx, sy] = t.toScreen((f * x / per - kx.offset) / kx.scale, (-f * z.y[i] / per - ky.offset) / ky.scale);
          return (
            <circle key={i} cx={sx} cy={sy} r={2 + 5 * Math.sqrt(z.intensity[i])} fill="none" stroke={templateColour} strokeWidth={1.25}>
              <title>{z.hkl[i].join(" ")}</title>
            </circle>
          );
        })}
      </g>
    );
  };
  // ---------- matching at the cursors (pyxem's get_orientation) ----------
  // Each cursor's best template, as pyxem draws it (on the pattern's axes;
  // on a polar transform at angle, radius), as the cursors move.
  const polarData = isPolar(ds);
  const matchTemplates = useOrientation((s) => chosenFor(s, ds));
  const matchOn = useOrientation((s) => s.matching[ds.id] ?? false) && !!matchTemplates
    && allDatasets(files).some((x) => x.id === matchTemplates);
  const matchLayer = ovOf("match");
  const matchVisible = matchLayer.visible ?? true;
  const matchStyle = matchLayer.style === "cross" ? "cross" : "circle";
  const matchKey = matchOn ? `${version}|${matchTemplates}|${cursorKeys.join(";")}` : "";
  const [matches, setMatches] = useState<{ key: string; polar: string; at: Record<string, MatchResult> } | null>(null);
  const [matchError, setMatchError] = useState<string | null>(null);
  const matchBusy = useRef(false);
  // The newest wanted: its key, templates and positions (the cursors when wanted).
  const matchWanted = useRef<{ key: string; templates: string; positions: number[][] } | null>(null);
  useEffect(() => {
    matchWanted.current = matchKey ? { key: matchKey, templates: matchTemplates!, positions: cursors.map((c) => [c.x, c.y]) } : null;
    if (!matchKey || matches?.key === matchKey) return;
    const next = () => {
      const w = matchWanted.current;
      if (!w || matchBusy.current) return;
      matchBusy.current = true;
      api.matchAt(ds.id, w.templates, w.positions)
        .then((r) => {
          setMatches({ key: w.key, polar: r.polar, at: Object.fromEntries(r.matches.map((m) => [`${m.x},${m.y}`, m])) });
          setMatchError(null);
        }, (e: Error) => setMatchError(e.message))
        .finally(() => {
          matchBusy.current = false;
          if (matchWanted.current && matchWanted.current.key !== w.key) next();
        });
    };
    next();
  }, [matchKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const matchOf = (id: string) => {
    const c = cursors.find((k) => k.id === id);
    return matchOn && c ? matches?.at[`${c.x},${c.y}`] : undefined;
  };
  const matchOverlay = (t: Transform, id: string): ReactNode => {
    const m = matchVisible ? matchOf(id) : undefined;
    if (!m) return null;
    const c = cursors.find((k) => k.id === id);
    const colour = matchLayer.colour ?? c?.color ?? "#ff9f0a";
    const [ky, kx] = ds.axes.filter((a) => !a.navigate);
    const per = perAngstrom(ds);
    const points: [number, number][] = polarData
      // Polar transform: angle across (rad), radius down (1/Å).
      ? m.polar.theta.map((th, i) => [(th - kx.offset) / kx.scale, (m.polar.r[i] - ky.offset) / ky.scale])
      : per
        ? m.vectors.x.map((x, i) => [(x / per - kx.offset) / kx.scale, (m.vectors.y[i] / per - ky.offset) / ky.scale])
        : [];
    return (
      <g className="fourd-overlay" opacity={matchLayer.opacity ?? 1}>
        {points.map(([px, py], i) => {
          const [sx, sy] = t.toScreen(px, py);
          const r = 2.5 + 4 * Math.sqrt(m.vectors.intensity[i] ?? 0);
          const title = <title>{m.vectors.hkl[i]?.join(" ")}</title>;
          if (matchStyle === "cross") {
            return (
              <path key={i} d={`M${sx - r},${sy}H${sx + r}M${sx},${sy - r}V${sy + r}`}
                stroke={colour} strokeWidth={1.5} strokeLinecap="round">{title}</path>
            );
          }
          return (
            <circle key={i} cx={sx} cy={sy} r={r} fill="none" stroke={colour} strokeWidth={1.5}>{title}</circle>
          );
        })}
      </g>
    );
  };
  const matchRow = matchOn ? [{
    id: "fd:match",
    label: "Best match",
    visible: matchVisible,
    opacity: matchLayer.opacity ?? 1,
    display: undefined,
    removable: false,
    colour: matchLayer.colour ?? "#ff9f0a",
    properties: [{
      label: "Mark",
      control: (
        <select className="layer-choice" value={matchStyle} title="How each spot of the best template is marked"
          onChange={(e) => setOverlay(ds.id, "match", { style: e.target.value as "circle" | "cross" })}>
          <option value="circle">Circle</option>
          <option value="cross">Cross</option>
        </select>
      ),
    }],
  }] : [];
  const matchTabs = matchOn && !grid ? [{
    id: "match" as const,
    label: "Match",
    content: (
      <MatchPanel templatesId={matchTemplates!} cursors={cursors} activeId={activeCursor.id}
        at={matches?.key === matchKey ? matches.at : matches?.at ?? {}} error={matchError} />
    ),
  }] : [];

  const templateRow = templatesDs ? [{
    id: "fd:template",
    label: `${templateStructure?.title ?? templatesDs.templates!.name} · ${templatesDs.title.replace(/^Templates · /, "")}`
      + (viewPattern ? ` · [${viewPattern.uvw.map((x) => formatValue(x)).join(" ")}]` : ""),
    visible: templateLayer.visible ?? true,
    opacity: templateLayer.opacity ?? 1,
    display: undefined,
    removable: false,
    colour: templateColour,
  }] : [];
  const structureTabs = templateStructure && !grid ? [{
    id: "structure" as const,
    label: "Structure",
    content: (
      <div className="structure-tab">
        <StructureScene key={templateStructure.id} d={templateStructure}
          initial={useOrientation.getState().views[templateStructure.id]}
          onOrientation={(o) => useOrientation.getState().setView(templateStructure.id, o)} />
      </div>
    ),
  }] : [];


  // ---------- the ring, while the Calibration process is shown ----------
  // The annulus searched (dashed), the ellipse fitted and the points on it.
  const calibrationShown = useProcessShown("calibration") && (focused || braggFocused);
  const ringSearch = useCalibration((s) => s.search[ds.id]);
  const ringGuess = useCalibration((s) => s.guess[ds.id]);
  const fittedRing = useCalibration((s) => s.ring[ds.id] ?? null);
  const ringOverlay = (t: Transform): ReactNode => {
    if (!calibrationShown) return null;
    const z = t.zoom;
    const g = ringGuess ?? { cx: patternG.width / 2, cy: patternG.height / 2 };
    const [gx, gy] = t.toScreen(g.cx, g.cy);
    const search = ringSearch ?? { r_min: Math.round(patternG.width * 0.15), r_max: Math.round(patternG.width * 0.25) };
    return (
      <g className="fourd-overlay">
        {[search.r_min, search.r_max].map((r, i) => (
          <circle key={i} cx={gx} cy={gy} r={r * z} fill="none" stroke={RING_COLOR} strokeWidth={1} strokeDasharray="4 3" opacity={0.8} />
        ))}
        {fittedRing && (() => {
          const [ex, ey] = t.toScreen(fittedRing.cx, fittedRing.cy);
          return (
            <>
              {fittedRing.points.map(([x, y], i) => {
                const [px, py] = t.toScreen(x, y);
                return <circle key={i} cx={px} cy={py} r={1.5} fill={RING_COLOR} />;
              })}
              <ellipse cx={ex} cy={ey} rx={fittedRing.a * z} ry={fittedRing.b * z}
                transform={`rotate(${(fittedRing.angle * 180) / Math.PI} ${ex} ${ey})`}
                fill="none" stroke={RING_FIT_COLOR} strokeWidth={1.5} />
              <line x1={ex - 6} x2={ex + 6} y1={ey} y2={ey} stroke={RING_FIT_COLOR} strokeWidth={1.5} />
              <line x1={ex} x2={ex} y1={ey - 6} y2={ey + 6} stroke={RING_FIT_COLOR} strokeWidth={1.5} />
            </>
          );
        })()}
      </g>
    );
  };

  // A press that makes a grid panel the active one only does that (it
  // doesn't also move the cursor or the detector): set when the panel changes.
  const activatedAt = useRef(0);
  const swallow = () => performance.now() - activatedAt.current < 150;

  // ---------- moving things ----------
  const clampScan = (x: number, y: number) => ({
    x: Math.max(0, Math.min(scanG.width - 1, Math.floor(x))),
    y: Math.max(0, Math.min(scanG.height - 1, Math.floor(y))),
  });
  const latest = useRef(st);
  latest.current = st;
  // Moves land in ``latest`` at once, so quick key repeats add up before the next render.
  const setScan = (next: Partial<FourDState>) => {
    latest.current = { ...latest.current, ...next };
    patch(next);
  };
  /** Cursor ``id`` moved to (x, y), and its pattern shown. */
  const moveCursor = (id: string, x: number, y: number) =>
    setScan({
      cursors: latest.current.cursors.map((c) => (c.id === id ? { ...c, ...clampScan(x, y) } : c)),
      source: id,
      lastCursor: id,
    });
  const removeCursor = (id: string) => {
    const rest = latest.current.cursors.filter((c) => c.id !== id);
    if (!rest.length) return;
    const last = rest[rest.length - 1].id;
    setScan({
      cursors: rest,
      lastCursor: latest.current.lastCursor === id ? last : latest.current.lastCursor,
      source: latest.current.source === id ? last : latest.current.source,
    });
  };
  /** The cursor within a few screen pixels of ``p``, if any (the top one). */
  const cursorAt = (p: Pt, t: Transform) =>
    [...latest.current.cursors].reverse().find((c) => {
      const [x0, y0] = t.toScreen(c.x, c.y);
      const [x1, y1] = t.toScreen(c.x + 1, c.y + 1);
      const [px, py] = t.toScreen(p.x, p.y);
      return Math.hypot(Math.max(x0 - px, 0, px - x1), Math.max(y0 - py, 0, py - y1)) <= GRAB_PX;
    }) ?? null;
  const setDetector = (detector: Detector) => {
    const s = latest.current;
    const id = groupOf(s).id;
    setScan({ groups: s.groups.map((g) => (g.id === id ? { ...g, detector } : g)) });
  };

  const arrow = (e: { key: string; shiftKey: boolean }) => {
    const k = e.shiftKey ? SHIFT_STEP : STEP;
    const d = ({ ArrowLeft: [-k, 0], ArrowRight: [k, 0], ArrowUp: [0, -k], ArrowDown: [0, k] } as Record<string, number[]>)[e.key];
    return d ?? null;
  };
  const wasd = (k: number) => {
    const move = (dx: number, dy: number) => (e: KeyboardEvent) => {
      const s = e.shiftKey ? SHIFT_STEP : STEP;
      setDetector(moveDetector(groupOf(latest.current).detector, dx * s * k, dy * s * k));
    };
    return { w: move(0, -1), a: move(-1, 0), s: move(0, 1), d: move(1, 0) };
  };
  useShortcuts(wasd(1));

  // The scan: ⌘/Ctrl-press adds a cursor; a press on a cursor grabs it, on a
  // region shows its pattern; elsewhere the active cursor goes there. Drag to
  // move. Regions are drawn with the viewer's region tools.
  const scanDrag = useRef<string | null>(null);
  const scanInteraction: Interaction = {
    cursor: "crosshair",
    onDown(p, e, t) {
      if (swallow()) return true;
      const s = latest.current;
      if (e.metaKey || e.ctrlKey) {
        const c = newCursor(s.cursors, clampScan(p.x, p.y).x, clampScan(p.x, p.y).y);
        setScan({ cursors: [...s.cursors, c], source: c.id, lastCursor: c.id });
        scanDrag.current = c.id;
        return true;
      }
      const hit = cursorAt(p, t);
      if (hit) {
        setScan({ source: hit.id, lastCursor: hit.id });
        scanDrag.current = hit.id;
        return true;
      }
      const roi = [...rois].reverse().find((r) => containsPoint(r.shape, p.x, p.y));
      if (roi) {
        setScan({ source: roi.id });
        return false; // the image pans
      }
      const id = s.cursors.find((c) => c.id === s.source)?.id ?? s.lastCursor;
      moveCursor(id, p.x, p.y);
      scanDrag.current = id;
      return true;
    },
    onMove(p) {
      if (scanDrag.current) moveCursor(scanDrag.current, p.x, p.y);
    },
    onUp() {
      scanDrag.current = null;
    },
    onKeyDown(e) {
      const s = latest.current;
      if (e.key === "Delete" || e.key === "Backspace") {
        if (sourceRoi && s.source === sourceRoi.id) {
          useSi.getState().deleteRois(scanKey, [sourceRoi.id]);
          return true;
        }
        if (s.cursors.length > 1 && s.cursors.some((c) => c.id === s.source)) {
          removeCursor(s.source);
          return true;
        }
        return false;
      }
      const d = arrow(e);
      if (!d) return false;
      const c = s.cursors.find((k) => k.id === s.source) ?? s.cursors.find((k) => k.id === s.lastCursor) ?? s.cursors[0];
      moveCursor(c.id, c.x + d[0], c.y + d[1]);
      return true;
    },
  };
  // Right-click a cursor (with more than one): delete it.
  const scanMenu = (p: Pt): MenuItem[] => {
    const s = latest.current;
    if (s.cursors.length < 2) return [];
    const hit = [...s.cursors].reverse().find((c) => Math.floor(p.x) === c.x && Math.floor(p.y) === c.y);
    return hit ? [{ label: `Delete ${cursorName(hit)}`, danger: true, onSelect: () => removeCursor(hit.id) }] : [];
  };

  // The pattern: drag a handle to resize the detector, inside it to move it;
  // a press elsewhere moves it there.
  const patternDrag = useRef<{ mode: "move"; start: Pt; orig: Detector } | { mode: "handle"; id: string } | null>(null);
  const [patternCursor, setPatternCursor] = useState("crosshair");
  const handleAt = (p: Pt, t: Transform) => {
    const tol = HANDLE_PX / t.zoom;
    return handles(det).find((h) => h.id !== "c" && Math.hypot(p.x - h.x, p.y - h.y) <= tol) ?? null;
  };
  // Another visible group's detector under ``p`` (a press there selects that group).
  const otherAt = (p: Pt, t: Transform) =>
    [...latest.current.groups]
      .reverse()
      .find(
        (g) => g.visible && g.id !== groupOf(latest.current).id
          && onOutline(detectorForRequest(g.detector, patternG.width, patternG.height), p, HANDLE_PX / 2 / t.zoom),
      ) ?? null;
  const selectGroup = (id: string) => setScan({ selectedGroup: id, shown: "live" });
  const patternInteraction: Interaction = {
    cursor: patternCursor,
    onHover(p, t) {
      if (!p || !t) return setPatternCursor("crosshair");
      setPatternCursor(handleAt(p, t) || inside(det, p) ? "grab" : otherAt(p, t) ? "pointer" : "default");
    },
    onDown(p, _e, t) {
      if (swallow()) return true;
      const h = handleAt(p, t);
      if (h) {
        setPatternCursor("grabbing");
        patternDrag.current = { mode: "handle", id: h.id };
        return true;
      }
      const other = !inside(det, p) && otherAt(p, t);
      if (other) {
        selectGroup(other.id);
        return true;
      }
      // Only a drag from inside moves the detector: elsewhere the pattern
      // pans (and a double-click fits it).
      if (!inside(det, p)) return false;
      setPatternCursor("grabbing");
      patternDrag.current = { mode: "move", start: p, orig: det };
      return true;
    },
    onMove(p) {
      const g = patternDrag.current;
      if (!g) return;
      const d = groupOf(latest.current).detector;
      if (g.mode === "move") {
        setDetector(moveDetector(g.orig, p.x - g.start.x, p.y - g.start.y));
      } else if (g.id === "r") {
        const r = Math.hypot(p.x - d.cx, p.y - d.cy);
        setDetector({ ...d, r: d.shape === "annulus" ? Math.max(r, d.r_inner + 0.5) : Math.max(0.5, r) });
      } else if (g.id === "ri") {
        setDetector({ ...d, r_inner: Math.max(0, Math.min(Math.hypot(p.x - d.cx, p.y - d.cy), d.r - 0.5)) });
      } else {
        setDetector({ ...d, [g.id.slice(0, 2)]: p.x, [g.id.slice(2)]: p.y });
      }
    },
    onUp() {
      patternDrag.current = null;
      setPatternCursor("grab");
    },
    onKeyDown(e) {
      const d = arrow(e);
      if (!d) return false;
      setDetector(moveDetector(groupOf(latest.current).detector, d[0], d[1]));
      return true;
    },
  };

  // Apertures: a ⌘/Ctrl-press on the pattern places one (drag to move it), on one
  // moves it, on the handle beside the selected one sizes them all; the
  // array's markers (O, g1, g2) are dragged; right-click or Delete removes one.
  const [selectedAperture, setSelectedAperture] = useState<number | null>(null);
  const apDrag = useRef<
    { kind: "aperture"; i: number; start: Pt; orig: { x: number; y: number } } | { kind: "radius" } | { kind: "marker"; id: string } | null
  >(null);
  const aperturesOf = (d: Detector) => d.apertures ?? [];
  const sizedAperture = (d: Detector) => {
    const list = aperturesOf(d);
    return (selectedAperture !== null && list[selectedAperture]) || list[list.length - 1] || null;
  };
  const apHit = (p: Pt, t: Transform) => {
    const tol = HANDLE_PX / t.zoom;
    const d = det;
    const marker = d.lattice && latticeMarkers(d.lattice).find((h) => Math.hypot(p.x - h.x, p.y - h.y) <= tol);
    if (marker) return { kind: "marker" as const, id: marker.id };
    const sized = sizedAperture(d);
    if (sized && Math.hypot(p.x - (sized.x + apertureR(d, sized)), p.y - sized.y) <= tol) return { kind: "radius" as const };
    const i = apertureAt(d, p);
    return i >= 0 ? { kind: "aperture" as const, i } : null;
  };
  const setApertures = (list: { x: number; y: number }[]) => setDetector({ ...groupOf(latest.current).detector, apertures: list });
  const aperturesInteraction: Interaction = {
    cursor: patternCursor,
    onHover(p, t) {
      if (!p || !t) return setPatternCursor("default");
      setPatternCursor(apHit(p, t) ? "grab" : otherAt(p, t) ? "pointer" : "default");
    },
    onDown(p, e, t) {
      if (swallow()) return true;
      const h = apHit(p, t);
      const other = !h && otherAt(p, t);
      if (other) {
        selectGroup(other.id);
        return true;
      }
      // ⌘/Ctrl-click places an aperture; a plain press off the apertures pans
      // (and a double-click fits the pattern).
      const add = !h && (e.metaKey || e.ctrlKey);
      if (!h && !add) return false;
      setPatternCursor("grabbing");
      const list = aperturesOf(det);
      if (!h) {
        setApertures([...list, { x: p.x, y: p.y }]);
        setSelectedAperture(list.length);
        apDrag.current = { kind: "aperture", i: list.length, start: p, orig: { x: p.x, y: p.y } };
      } else if (h.kind === "aperture") {
        setSelectedAperture(h.i);
        apDrag.current = { kind: "aperture", i: h.i, start: p, orig: list[h.i] };
      } else {
        apDrag.current = h;
      }
      return true;
    },
    onMove(p) {
      const g = apDrag.current;
      if (!g) return;
      const d = groupOf(latest.current).detector;
      if (g.kind === "aperture") {
        setApertures(aperturesOf(d).map((a, i) => (i === g.i ? { x: g.orig.x + p.x - g.start.x, y: g.orig.y + p.y - g.start.y } : a)));
      } else if (g.kind === "radius") {
        const sized = sizedAperture(d);
        // The handle sizes them all (merged apertures lose their own sizes).
        if (sized) setDetector(sizeApertures(d, Math.max(0.5, Math.hypot(p.x - sized.x, p.y - sized.y))));
      } else if (d.lattice) {
        const l = d.lattice;
        const lattice = g.id === "o" ? { ...l, ox: p.x, oy: p.y } : g.id === "g1" ? { ...l, g1x: p.x - l.ox, g1y: p.y - l.oy } : { ...l, g2x: p.x - l.ox, g2y: p.y - l.oy };
        setDetector({ ...d, lattice });
      }
    },
    onUp() {
      apDrag.current = null;
      setPatternCursor("grab");
    },
    onContextMenu(p) {
      const i = apertureAt(det, p);
      if (i < 0) return false;
      setApertures(aperturesOf(det).filter((_, j) => j !== i));
      setSelectedAperture(null);
      return true;
    },
    onKeyDown(e) {
      if ((e.key === "Delete" || e.key === "Backspace") && selectedAperture !== null) {
        setApertures(aperturesOf(groupOf(latest.current).detector).filter((_, j) => j !== selectedAperture));
        setSelectedAperture(null);
        return true;
      }
      const d = arrow(e);
      if (!d) return false;
      setDetector(moveDetector(groupOf(latest.current).detector, d[0], d[1]));
      return true;
    },
  };

  // ---------- the strain lattice: O, g1 and g2 dragged on the Bragg vector map ----------
  // A tip let go near a peak snaps to it (Alt: no snapping).
  const strainDrag = useRef<string | null>(null);
  const strainMarkers = (l: StrainLattice): Handle[] => [
    { id: "o", x: l.ox, y: l.oy },
    { id: "g1", x: l.ox + l.g1x, y: l.oy + l.g1y },
    { id: "g2", x: l.ox + l.g2x, y: l.oy + l.g2y },
  ];
  const strainHit = (p: Pt, t: Transform) =>
    strainLattice ? strainMarkers(strainLattice).find((h) => Math.hypot(p.x - h.x, p.y - h.y) <= HANDLE_PX / t.zoom) ?? null : null;
  const strainInteraction: Interaction = {
    cursor: patternCursor,
    onHover(p, t) {
      if (!p || !t) return setPatternCursor("default");
      setPatternCursor(strainHit(p, t) ? "grab" : "default");
    },
    onDown(p, _e, t) {
      const h = strainHit(p, t);
      if (!h) return false;
      strainDrag.current = h.id;
      setPatternCursor("grabbing");
      return true;
    },
    onMove(p, e, t) {
      const id = strainDrag.current;
      const l = useStrain.getState().lattice[ds.id];
      if (!id || !l) return;
      const peaks = useStrain.getState().peaks[ds.id]?.list ?? [];
      const snap = e.altKey ? null : nearestPeak(peaks, p.x, p.y, Math.max(strainTol, HANDLE_PX / t.zoom));
      const x = snap ? snap[0] : p.x;
      const y = snap ? snap[1] : p.y;
      useStrain.getState().setLattice(ds.id, id === "o"
        ? { ...l, ox: x, oy: y }
        : id === "g1" ? { ...l, g1x: x - l.ox, g1y: y - l.oy } : { ...l, g2x: x - l.ox, g2y: y - l.oy });
    },
    onUp() {
      strainDrag.current = null;
      setPatternCursor("grab");
    },
  };
  const strainOverlay = (t: Transform): ReactNode => {
    const l = strainLattice;
    const peaks = strainPeaks ?? [];
    const points = l ? latticeGrid(l, patternG.width, patternG.height) : [];
    const on = peaksOnLattice(peaks, points, strainTol);
    const z = t.zoom;
    return (
      <g className="fourd-overlay">
        {points.map((q) => {
          const [x, y] = t.toScreen(q.x, q.y);
          return <circle key={`${q.h},${q.k}`} cx={x} cy={y} r={Math.max(3, strainTol * z)} fill="none" stroke={STRAIN_COLOR} strokeWidth={1} strokeDasharray="3 2" opacity={0.9} />;
        })}
        {peaks.map(([px, py], i) => {
          const [x, y] = t.toScreen(px, py);
          return <circle key={`p${i}`} cx={x} cy={y} r={2.5} fill={on.has(i) ? PEAK_ON_COLOR : PEAK_OFF_COLOR} stroke="#000" strokeWidth={0.5} />;
        })}
        {l && (() => {
          const [ox, oy] = t.toScreen(l.ox, l.oy);
          return strainMarkers(l).slice(1).map((h) => {
            const [x, y] = t.toScreen(h.x, h.y);
            return <line key={`v${h.id}`} x1={ox} y1={oy} x2={x} y2={y} stroke={STRAIN_COLOR} strokeWidth={2} />;
          });
        })()}
        {l && strainMarkers(l).map((h) => {
          const [x, y] = t.toScreen(h.x, h.y);
          return (
            <g key={h.id}>
              <circle cx={x} cy={y} r={5} fill={STRAIN_COLOR} stroke="#000" strokeWidth={0.75} />
              <text x={x + 7} y={y - 7} fill={STRAIN_COLOR} fontSize={11} fontWeight={600}>{h.id === "o" ? "O" : h.id}</text>
            </g>
          );
        })}
      </g>
    );
  };

  // At a cursor: its disks, indexed (white) or not (orange), and the lattice fitted to them (crosses).
  const strainAtOverlay = (t: Transform, at: StrainAt): ReactNode => (
    <g className="fourd-overlay">
      {(at.points ?? []).map(([px, py, h, k]) => {
        const [x, y] = t.toScreen(px, py);
        return (
          <g key={`f${h},${k}`} stroke={STRAIN_COLOR} strokeWidth={1.5}>
            <line x1={x - 5} x2={x + 5} y1={y} y2={y} />
            <line x1={x} x2={x} y1={y - 5} y2={y + 5} />
          </g>
        );
      })}
      {at.disks.map(([px, py, , , on], i) => {
        const [x, y] = t.toScreen(px, py);
        return <circle key={`d${i}`} cx={x} cy={y} r={Math.max(4, strainTol * t.zoom)} fill="none" stroke={on ? PEAK_ON_COLOR : PEAK_OFF_COLOR} strokeWidth={1.25} />;
      })}
    </g>
  );

  // ---------- overlays ----------
  /** A frame around an image in ``color`` (``w`` × ``h`` image pixels). */
  const frame = (t: Transform, w: number, h: number, color: string | null | undefined): ReactNode => {
    if (!color) return null;
    const [ax, ay] = t.toScreen(0, 0);
    const [bx, by] = t.toScreen(w, h);
    return <rect x={ax - 1} y={ay - 1} width={bx - ax + 2} height={by - ay + 2} fill="none" stroke={color} strokeWidth={FRAME_WIDTH} strokeOpacity={FRAME_OPACITY} />;
  };
  // ``key``: the grid panel's layer ("vi:<group>"); the live image is framed in its group's colour.
  const scanOverlay = (t: Transform, key?: string): ReactNode => (
    <g className="fourd-overlay">
      {frame(t, scanG.width, scanG.height, multi ? st.groups.find((g) => `vi:${g.id}` === key)?.color : shownSaved ? null : group.color)}
      {cursors.filter(cursorShown).map((c) => {
        const [ax, ay] = t.toScreen(c.x, c.y);
        const [bx, by] = t.toScreen(c.x + 1, c.y + 1);
        const [cx, cy] = [(ax + bx) / 2, (ay + by) / 2];
        const on = c.id === source;
        const w = on ? 2 : 1.25;
        return (
          <g key={c.id} opacity={(on ? 1 : 0.75) * (ovOf(`cursor:${c.id}`).opacity ?? 1)}>
            <rect x={ax} y={ay} width={bx - ax} height={by - ay} fill="none" stroke={c.color} strokeWidth={w} />
            <line x1={cx - 9} x2={cx - 4} y1={cy} y2={cy} stroke={c.color} strokeWidth={w} />
            <line x1={cx + 4} x2={cx + 9} y1={cy} y2={cy} stroke={c.color} strokeWidth={w} />
            <line x1={cx} x2={cx} y1={cy - 9} y2={cy - 4} stroke={c.color} strokeWidth={w} />
            <line x1={cx} x2={cx} y1={cy + 4} y2={cy + 9} stroke={c.color} strokeWidth={w} />
            {cursors.length > 1 && (
              <text x={cx + 7} y={cy - 7} fill={c.color} fontSize={11} fontWeight={600}>{c.id.slice(1)}</text>
            )}
          </g>
        );
      })}
    </g>
  );
  const badRings = (t: Transform): ReactNode =>
    bad && (
      <g className="fourd-overlay">
        {[...bad.hot, ...bad.dead].map(([y, x]) => {
          const [sx, sy] = t.toScreen(x + 0.5, y + 0.5);
          return <circle key={`${y},${x}`} cx={sx} cy={sy} r={Math.max(5, t.zoom)} fill="none" stroke={BAD_COLOR} strokeWidth={1.5} />;
        })}
      </g>
    );
  const aperturesOverlay = (t: Transform): ReactNode => {
    const d = det;
    const z = t.zoom;
    const placed = aperturesOf(d);
    const sized = sizedAperture(d);
    const generated = d.lattice ? latticePoints(d.lattice, patternG.width, patternG.height) : [];
    const marker = (h: Handle) => {
      const [x, y] = t.toScreen(h.x, h.y);
      return (
        <g key={h.id}>
          <circle cx={x} cy={y} r={5} fill={LATTICE_COLOR} stroke="#000" strokeWidth={0.75} />
          <text x={x + 7} y={y - 7} fill={LATTICE_COLOR} fontSize={11} fontWeight={600}>{h.id === "o" ? "O" : h.id}</text>
        </g>
      );
    };
    return (
      <g className="fourd-overlay">
        {generated.map((a, i) => {
          const [x, y] = t.toScreen(a.x, a.y);
          return <circle key={`g${i}`} cx={x} cy={y} r={d.r * z} fill={group.color} fillOpacity={DETECTOR_FILL} stroke={group.color} strokeWidth={1.25} strokeDasharray="4 3" />;
        })}
        {placed.map((a, i) => {
          const [x, y] = t.toScreen(a.x, a.y);
          return <circle key={`p${i}`} cx={x} cy={y} r={apertureR(d, a) * z} fill={group.color} fillOpacity={DETECTOR_FILL} stroke={group.color} strokeWidth={i === selectedAperture ? 2.5 : 1.5} />;
        })}
        {sized && (() => {
          const [hx, hy] = t.toScreen(sized.x + apertureR(d, sized), sized.y);
          return <rect x={hx - 3.5} y={hy - 3.5} width={7} height={7} fill={group.color} stroke="#000" strokeWidth={0.75} />;
        })()}
        {d.lattice && (() => {
          const [ox, oy] = t.toScreen(d.lattice.ox, d.lattice.oy);
          return latticeMarkers(d.lattice).slice(1).map((h) => {
            const [x, y] = t.toScreen(h.x, h.y);
            return <line key={`v${h.id}`} x1={ox} y1={oy} x2={x} y2={y} stroke={LATTICE_COLOR} strokeWidth={1.5} />;
          });
        })()}
        {d.lattice && latticeMarkers(d.lattice).map(marker)}
      </g>
    );
  };
  // ``key``: the grid panel's layer ("dp:<source>"); the pattern is framed in its cursor's or region's colour.
  const sourceBorder = (t: Transform, key?: string): ReactNode => (
    <g className="fourd-overlay">
      {frame(t, patternG.width, patternG.height, grid ? sources.find((s) => `dp:${s.id}` === key)?.color : sourceColor)}
    </g>
  );
  // The other visible groups: their outlines, thinner, in their colours.
  const othersOverlay = (t: Transform): ReactNode => (
    <g className="fourd-overlay" opacity={0.7}>
      {st.groups
        .filter((g) => g.id !== group.id && detShown(g))
        .map((g) => <g key={g.id} opacity={detOpacity(g)}>{outline(t, detectorForRequest(g.detector, patternG.width, patternG.height), g.color, 1)}</g>)}
    </g>
  );
  const patternOverlay = (t: Transform, key?: string): ReactNode => {
    const id = grid && key ? key.replace(/^dp:/, "") : source;
    // The strain lattice on the Bragg vector map, without the detectors (they'd hide it).
    if (strainShown && id === SCAN_BVM) return <>{sourceBorder(t, key)}{strainOverlay(t)}</>;
    // The last preview stays until the next arrives (no flicker while dragging).
    const at = strainPreviewOn && strainPreview
      ? (() => {
          const c = cursors.find((k) => k.id === id);
          return c ? strainPreview.at[`${c.x},${c.y}`] : undefined;
        })()
      : undefined;
    if (at) return <>{sourceBorder(t, key)}{strainAtOverlay(t, at)}</>;
    return (
      <>
        {sourceBorder(t, key)}
        {disksOverlay(t, id)}
        {beamOverlay(t, id)}
        {ringOverlay(t)}
        {zoneOverlay(t)}
        {matchOverlay(t, id)}
        {othersOverlay(t)}
        {detectorOverlay(t)}
      </>
    );
  };
  const detectorOverlay = (t: Transform): ReactNode => {
    const d = det;
    // Its layer hidden: not drawn (still moved by its keys and the process).
    if (!detShown(group)) return badRings(t);
    if (d.shape === "apertures") return <><g opacity={detOpacity(group)}>{aperturesOverlay(t)}</g>{badRings(t)}</>;
    return (
      <g className="fourd-overlay" opacity={detOpacity(group)}>
        {outline(t, d, group.color, 1.5)}
        {handles(d).filter((h) => h.id !== "c" || d.shape !== "point").map((h) => {
          const [hx, hy] = t.toScreen(h.x, h.y);
          return <rect key={h.id} x={hx - 3.5} y={hy - 3.5} width={7} height={7} fill={group.color} stroke="#000" strokeWidth={0.75} />;
        })}
        {badRings(t)}
      </g>
    );
  };

  // ---------- labels ----------
  const scanCal = scanG.calX;
  const patCal = patternG.calX;
  const at = (c: ScanCursor) =>
    `(${c.x}, ${c.y}) · ${formatValue(c.x * scanCal.scale)}, ${formatValue(c.y * scanCal.scale)} ${scanCal.units}`;
  const cursorText = `${cursors.length > 1 ? `${cursorName(activeCursor)} ` : ""}${at(activeCursor)}`;
  const patternLabel = sourceRoi
    ? `Mean pattern over ${sourceRoi.name}`
    : source === SCAN_BVM
      ? `Bragg vector map${ds.calibration?.enabled?.origin ? " (centred)" : ""}`
      : source === SCAN_MEAN || source === SCAN_MAX
      ? `${source === SCAN_MAX ? "Max" : "Mean"} pattern over the scan`
      : `Pattern at ${cursorText}`;
  const liveLabel = `${group.name} · ${sent.source === "vectors" ?`DDF${group.weight === "count" ? " count" : ""} · ` : ""}${shapeLabel(det)} · ${detectorText(det, patCal.scale, patCal.units)}${sent.response === "sum" ? "" : ` · ${responseLabel(sent.response)}`}`;
  const virtualLabel = shownSaved ? shownSaved.title : liveLabel;
  // The cursors and regions: a list to pick the pattern shown.
  const sources = [
    ...cursors.map((c) => ({ id: c.id, label: `${cursorName(c)} (${c.x}, ${c.y})`, color: c.color })),
    ...rois.map((r) => ({ id: r.id, label: r.name, color: r.color })),
  ];
  const picked = sources.find((s) => s.id === source) ?? null;
  // The DP's border: the colour of the cursor or region it comes from.
  const sourceColor = picked?.color ?? null;
  const shownInList = picked ?? sources.find((s) => s.id === activeCursor.id)!;
  const sourcePicker = (
    <ListPicker
      title="The cursor or region whose pattern is shown"
      label={shownInList.label}
      color={shownInList.color}
      muted={!picked}
      items={sources.map((s) => ({
        label: s.label,
        color: s.color,
        checked: s.id === source,
        onSelect: () => setScan({ source: s.id, ...(cursors.some((c) => c.id === s.id) && { lastCursor: s.id }) }),
      }))}
    />
  );
  // Over the whole scan: the mean or max pattern, or the Bragg vector map once the disks are found.
  const wholeScan = segmented("Whole scan", [
    { id: SCAN_MEAN, label: "Mean", title: "The mean pattern over the whole scan" },
    { id: SCAN_MAX, label: "Max", title: "The maximum of every pixel over the whole scan" },
    ...(hasVectors
      ? [{ id: SCAN_BVM, label: "BVM", title: "The Bragg vector map on the pattern's pixels (centred while the descan is corrected)" }]
      : []),
  ], source, (v) => setScan({ source: v }));
  // The image on the scan: a group's, live, or a map made from the data.
  const imagePicker = (
    <ListPicker
      title="The image on the scan: a detector group's, live, or a map made from the data"
      label={shownSaved ? shownSaved.title : group.name}
      color={shownSaved ? undefined : group.color}
      items={[
        ...st.groups.map((g) => ({
          label: g.name,
          color: g.color,
          checked: !shownSaved && g.id === group.id,
          onSelect: () => selectGroup(g.id),
        })),
        ...(saved.length ? [{ label: "Maps", disabled: true }] : []),
        ...saved.map((d) => ({ label: d.title, indent: true, checked: d.id === shownSaved?.id, onSelect: () => patch({ shown: d.id }) })),
      ]}
    />
  );

  // ---------- the grid's panels ----------
  // The grid's active panels follow the selected group and the pattern shown;
  // a panel clicked selects its group or source.
  const viRequest = useMemo(() => (grid || merge ? { key: `vi:${group.id}` } : null), [grid, merge, group.id]);
  const dpRequest = useMemo(() => (grid ? { key: `dp:${source}` } : null), [grid, source]);
  const onViActive = (key: string) => {
    const id = key.replace(/^vi:/, "");
    if (id === latest.current.selectedGroup || !latest.current.groups.some((g) => g.id === id)) return;
    activatedAt.current = performance.now();
    selectGroup(id);
  };
  const onDpActive = (key: string) => {
    const id = key.replace(/^dp:/, "");
    if (id === source || !dpItems.some((i) => i.id === id)) return;
    activatedAt.current = performance.now();
    setScan({ source: id, ...(cursors.some((c) => c.id === id) && { lastCursor: id }) });
  };
  // Contrast is redone for each new frame shown (its key), not when it is asked for.
  // A group imaged as the colour COM: its own contrast, on the colour wheel.
  const isColourCom = (id: string) => viItems.find((i) => i.id === id)?.req.response === "com";
  const viStacks: StackLayer[] = visibleGroups.map((g) => ({
    dataset: ds,
    key: `vi:${g.id}`,
    label: g.name,
    displayKey: `${ds.id}:vi:${g.id}${isColourCom(g.id) ? ":com" : ""}`,
    frame: viSet.frames[g.id]?.frame ?? null,
    version: viSet.frames[g.id]?.key,
    removable: false,
    geometry: scanG,
    ...(isColourCom(g.id) && { colormap: "hsv" }),
  }));
  // Merge: the same images as layers of one view, each in its group's colour
  // (colours add up), under contrast keys of their own.
  const mergeKey = (id: string) => `${ds.id}:merge:vi:${id}`;
  const mergeBlend = st.mergeBlend ?? "dominant";
  // Dominant: at each pixel only the group brightest there (its value within
  // its contrast limits; among the layers shown) keeps its value, the others
  // drop to their lower limit (black); drawn added, one colour per pixel.
  const mergeDisplays = useDisplay((s) => s.display);
  const layerProps = useLayers((s) => s.props);
  const dominant = useMemo(() => {
    if (!merge || mergeBlend !== "dominant") return null;
    const items = visibleGroups.map((g) => ({
      id: g.id,
      f: viSet.frames[g.id]?.frame,
      d: mergeDisplays[mergeKey(g.id)],
      shown: layerProps[`vi:${g.id}`]?.visible ?? true,
    }));
    // Until every layer has its contrast (set from its own frame), as is.
    if (!items.length || items.some((i) => !i.f || !i.d)) return null;
    const n = items[0].f!.data.length;
    if (items.some((i) => i.f!.data.length !== n)) return null;
    const out = items.map((i) => new Float32Array(i.f!.data));
    const floor = items.map((i) => Math.min(i.d!.lo, i.d!.hi));
    for (let p = 0; p < n; p++) {
      let best = -1;
      let bestV = -Infinity;
      for (let k = 0; k < items.length; k++) {
        const { f, d, shown } = items[k];
        if (!shown) continue;
        const v = (f!.data[p] - d!.lo) / (d!.hi - d!.lo || 1);
        if (v > bestV) {
          bestV = v;
          best = k;
        }
      }
      for (let k = 0; k < items.length; k++) if (k !== best) out[k][p] = floor[k];
    }
    return Object.fromEntries(items.map((i, k) => [i.id, { ...i.f!, data: out[k] }])) as Record<string, Frame>;
  }, [merge, mergeBlend, visibleGroups.map((g) => g.id).join(), viSet.frames, mergeDisplays, layerProps]); // eslint-disable-line react-hooks/exhaustive-deps
  const mergeStacks: StackLayer[] = viStacks.map((s, i) => ({
    ...s,
    // The same version: contrast stays set from the whole image.
    frame: dominant?.[visibleGroups[i].id] ?? s.frame,
    displayKey: mergeKey(visibleGroups[i].id),
    colormap: visibleGroups[i].color,
  }));
  const dpLabel = (id: string) =>
    sources.find((s) => s.id === id)?.label ??
    (id === SCAN_BVM ? "Bragg vector map" : id === SCAN_MAX ? "Max over the scan" : "Mean over the scan");
  const dpStacks: StackLayer[] = dpItems.map((it) => ({
    dataset: ds,
    key: `dp:${it.id}`,
    label: dpLabel(it.id),
    displayKey: `${ds.id}:dp:${it.id}${log ? ":log" : ""}`,
    frame: dpDrawn[it.id] ?? null,
    version: `${dpSet.frames[it.id]?.key}|${log}`,
    removable: false,
    geometry: patternG,
  }));
  const setLayout = (v: FourDState["layout"]) => {
    // The other layout's panels are another size: they open fitted.
    useDisplay.setState((s) => {
      const views = { ...s.views };
      for (const k of [`${ds.id}:virtual`, `${ds.id}:pattern`, `${ds.id}:pattern:log`, `${ds.id}:list:dp`, `${ds.id}:list:vi`]) delete views[k];
      return { views };
    });
    patch({ layout: v });
  };
  const layoutSwitch = segmented("Layout", [
    { id: "single", label: "Single", title: "One virtual image and one pattern" },
    { id: "grid", label: "Grid", title: "A virtual image per visible group, a pattern per cursor and region" },
    { id: "merge", label: "Merge", title: "The visible groups' virtual images as one colour composite, each in its group's colour" },
    { id: "list", label: "List", title: "A row per visible group: the pattern with its detector, and its virtual image (the wheel zooms; ⌘ + wheel or the scroll bar scrolls)" },
  ], st.layout, setLayout);
  // Right-click a grid panel: open it in Single (the press made its group,
  // or its cursor or region, the one shown).
  const openInSingle: MenuItem[] = grid ? [{ label: "Open in Single", onSelect: () => setLayout("single") }] : [];

  // ---------- the list: a row per visible group ----------
  const displays = useDisplay((s) => s.display);
  const setDisplay = useDisplay((s) => s.setDisplay);
  // Merge: each layer keeps its group's colour (a colour changed in the process follows).
  useEffect(() => {
    if (!merge) return;
    for (const g of visibleGroups) {
      const d = displays[mergeKey(g.id)];
      if (d && d.colormap !== g.color) setDisplay(mergeKey(g.id), { colormap: g.color });
    }
  });
  // Every image of the list has its own contrast (each row's pattern too).
  const listDpKey = (id: string) => `${ds.id}:list:dp:${id}${log ? ":log" : ""}`;
  const listViKey = (id: string) => `${ds.id}:list:vi:${id}${isColourCom(id) ? ":com" : ""}`;
  // Each image's contrast, redone for each new frame shown.
  useEffect(() => {
    if (!list) return;
    for (const g of visibleGroups) {
      if (drawnPattern) autoContrast(listDpKey(g.id), drawnPattern, `${pattern.shownKey}|${log}`);
      const v = viSet.frames[g.id];
      if (v) autoContrast(listViKey(g.id), v.frame, v.key, isColourCom(g.id) ? () => "hsv" : undefined);
    }
  });
  // The tools on the list's images: kept under the same keys as Single's panes.
  const viTools = useViewTools(scanKey, scanG.width, scanG.height, scanInteraction);
  const dpTools = useViewTools(
    `${ds.id}:pattern${log ? ":log" : ""}`, patternG.width, patternG.height,
    det.shape === "apertures" ? aperturesInteraction : patternInteraction,
  );
  const activateRow = (id: string) => {
    if (id === latest.current.selectedGroup) return;
    activatedAt.current = performance.now();
    selectGroup(id);
  };
  const cal = (g: ImageGeometry) => ({ calX: g.calX, calY: g.calY });

  // ---------- the Layers tab: the marks drawn, per pane ----------
  // The pattern: each group's detector, the Bragg disks; the scan: the cursors.
  const markKey = (id: string) => id.replace(/^fd:/, "");
  const disksStyle = disksLayer.style ?? "circle";
  const disksRow = {
    id: "fd:disks",
    label: vectorsDs ? vectorsDs.title : "Bragg disks",
    visible: disksVisible,
    opacity: disksLayer.opacity ?? 1,
    display: undefined,
    removable: false,
    colour: disksLayer.colour ?? DISK_COLOR,
    properties: [
      {
        label: "Mark",
        control: (
          <select
            className="layer-choice"
            value={disksStyle}
            title="A circle of the disk's size, or a dot at its position"
            onChange={(e) => setOverlay(ds.id, "disks", { style: e.target.value as "circle" | "dot" })}
          >
            <option value="circle">Circle</option>
            <option value="dot">Dot</option>
          </select>
        ),
      },
      ...(disksStyle === "dot"
        ? [{
            label: "Size",
            control: (
              <input
                type="range"
                className="layer-size"
                min={0.5}
                max={4}
                step={0.25}
                value={disksLayer.size ?? 1}
                title={`×${disksLayer.size ?? 1}`}
                onChange={(e) => setOverlay(ds.id, "disks", { size: Number(e.target.value) })}
              />
            ),
          }]
        : []),
    ],
  };
  const removeGroup = (id: string) => {
    const s = latest.current;
    const groups = s.groups.filter((g) => g.id !== id);
    if (groups.length) setScan({ groups, ...(s.selectedGroup === id && { selectedGroup: groups[groups.length - 1].id }) });
  };
  const dpExtra: ExtraLayers = {
    rows: [
      ...st.groups.map((g) => ({
        id: `fd:det:${g.id}`,
        label: `${g.name} · ${shapeLabel(g.detector)}`,
        visible: detShown(g),
        opacity: detOpacity(g),
        display: undefined,
        removable: st.groups.length > 1,
        colour: g.color,
      })),
      ...(vectorsDs || diskOpen ? [disksRow] : []),
      ...templateRow,
      ...matchRow,
    ],
    onActivate: (id) => id.startsWith("fd:det:") && selectGroup(id.slice("fd:det:".length)),
    onChange: (id, p) => {
      // A group's eye is the group's own (as in the Virtual image process); its opacity is the layer's.
      if (id.startsWith("fd:det:") && p.visible !== undefined) {
        const gid = id.slice("fd:det:".length);
        setScan({ groups: latest.current.groups.map((g) => (g.id === gid ? { ...g, visible: p.visible! } : g)) });
        const { visible: _, ...rest } = p;
        if (Object.keys(rest).length) setOverlay(ds.id, markKey(id), rest);
        return;
      }
      setOverlay(ds.id, markKey(id), p);
    },
    onColour: (id, colour) => {
      if (!id.startsWith("fd:det:")) return setOverlay(ds.id, markKey(id), { colour });
      const gid = id.slice("fd:det:".length);
      setScan({ groups: latest.current.groups.map((g) => (g.id === gid ? { ...g, color: colour } : g)) });
    },
    onRemove: (id) => id.startsWith("fd:det:") && removeGroup(id.slice("fd:det:".length)),
  };
  const viExtra: ExtraLayers = {
    rows: cursors.map((c) => ({
      id: `fd:cursor:${c.id}`,
      label: `${cursorName(c)} (${c.x}, ${c.y})`,
      visible: cursorShown(c),
      opacity: ovOf(`cursor:${c.id}`).opacity ?? 1,
      display: undefined,
      removable: cursors.length > 1,
      colour: c.color,
    })),
    onActivate: (id) => {
      const cid = id.slice("fd:cursor:".length);
      setScan({ source: cid, lastCursor: cid });
    },
    onChange: (id, p) => setOverlay(ds.id, markKey(id), p),
    onColour: (id, colour) => {
      const cid = id.slice("fd:cursor:".length);
      setScan({ cursors: latest.current.cursors.map((c) => (c.id === cid ? { ...c, color: colour } : c)) });
    },
    onRemove: (id) => removeCursor(id.slice("fd:cursor:".length)),
  };
  // The list has no layered viewers: its marks are published from here.
  const listOwner = `fourd-list:${ds.id}`;
  useEffect(() => {
    if (!list) return;
    const { publishSection } = useLayersTab.getState();
    const panelOf = (x: ExtraLayers) => ({
      layers: x.rows, active: `fd:det:${group.id}`, opacity: true,
      onActivate: (id: string) => x.onActivate?.(id), onChange: x.onChange, onColour: x.onColour, onRemove: (id: string) => x.onRemove?.(id),
    });
    publishSection("Diffraction pattern", listOwner, panelOf(dpExtra));
    publishSection("Virtual image", listOwner, panelOf(viExtra));
  });
  useEffect(() => {
    if (!list) return;
    return () => {
      useLayersTab.getState().clearSection("Diffraction pattern", listOwner);
      useLayersTab.getState().clearSection("Virtual image", listOwner);
    };
  }, [list, listOwner]);

  if (list) {
    return (
      <div className="view fourd-view">
        <div className="fourd-topbar">
          {layoutSwitch}
          <ImageToolButtons />
          {/* For every row: the patterns' and images' annotations, the magnifier. */}
          <AnnotationsToggle view={[scanKey, `${ds.id}:pattern${log ? ":log" : ""}`]} />
          <LoupeToggle which="viewer" />
        </div>
        <div className="fourd-list" data-wheel-scroll style={{ "--row-h": `${st.rowHeight}px` } as React.CSSProperties}>
          {visibleGroups.map((g, i) => {
            const v = viSet.frames[g.id]?.frame ?? null;
            const on = g.id === group.id;
            return (
              <section
                key={g.id}
                className={`fourd-row${on ? " selected" : ""}`}
                style={{ borderLeftColor: g.color }}
                onPointerDownCapture={() => activateRow(g.id)}
              >
                <header className="fourd-row-head">
                  <span className="fourd-swatch" style={{ background: g.color }} />
                  <span className="fourd-row-name">{g.name}</span>
                  <span className="muted small">{detectorText(g.detector, patCal.scale, patCal.units)}{g.response === "sum" ? "" : ` · ${responseLabel(g.response)}`}</span>
                </header>
                <div className="fourd-row-views">
                  <div className="fourd-cell">
                    <ContrastControls
                      frame={drawnPattern}
                      display={displays[listDpKey(g.id)]}
                      onChange={(p) => setDisplay(listDpKey(g.id), p)}
                    />
                    <ImageView
                      viewKey={`${ds.id}:list:dp`}
                      follow
                      wheelZoom="modifier"
                      loupeGroup={`${ds.id}:list:dp`}
                      layers={[{ key: "dp", label: patternLabel, frame: drawnPattern, display: displays[listDpKey(g.id)], opacity: 1, visible: true }]}
                      fullWidth={patternG.width}
                      fullHeight={patternG.height}
                      {...cal(patternG)}
                      loading={drawnPattern ? null : "Loading the pattern…"}
                      error={pattern.error}
                      interaction={dpTools.interaction}
                      renderOverlay={(t) => (
                        <>
                          {sourceBorder(t)}
                          {disksOverlay(t, source)}
                          {beamOverlay(t, source)}
                          {ringOverlay(t)}
                          {zoneOverlay(t)}
                          {matchOverlay(t, source)}
                          {on ? detectorOverlay(t) : detShown(g) && (
                            <g className="fourd-overlay" opacity={detOpacity(g)}>{outline(t, detectorForRequest(g.detector, patternG.width, patternG.height), g.color, 1.5)}</g>
                          )}
                          {dpTools.overlay(t, patternG.calX, patternG.calY)}
                        </>
                      )}
                      readout={false}
                      hoverChip
                      scaleBar={i === 0}
                      publish={false}
                    />
                  </div>
                  <div className="fourd-cell">
                    <ContrastControls
                      frame={v}
                      display={displays[listViKey(g.id)]}
                      onChange={(p) => setDisplay(listViKey(g.id), p)}
                    />
                    <ImageView
                      viewKey={`${ds.id}:list:vi`}
                      follow
                      wheelZoom="modifier"
                      loupeGroup={`${ds.id}:list:vi`}
                      layers={[{ key: "vi", label: g.name, frame: v, display: displays[listViKey(g.id)], opacity: 1, visible: true }]}
                      fullWidth={scanG.width}
                      fullHeight={scanG.height}
                      {...cal(scanG)}
                      loading={v ? null : "Computing the virtual image…"}
                      error={viSet.error}
                      interaction={viTools.interaction}
                      fftView={scanKey}
                      renderOverlay={(t) => (
                        <>
                          {scanOverlay(t, `vi:${g.id}`)}
                          {viTools.overlay(t, scanG.calX, scanG.calY)}
                        </>
                      )}
                      readout={false}
                      hoverChip
                      scaleBar={i === 0}
                      publish={on}
                    />
                  </div>
                </div>
              </section>
            );
          })}
          {!visibleGroups.length && <div className="image-message">Show a detector group (Virtual imaging process) to see its row</div>}
        </div>
        <div className="stack-bar fourd-bar fourd-list-bar">
          {sourcePicker}
          {wholeScan}
          {segmented("Scale", [{ id: "linear", label: "Linear" }, { id: "log", label: "Log", title: "log10 of the intensity" }], log ? "log" : "linear", (v) => setImageLog(logKey, v === "log"))}
          <label className="field" title="Row height">
            Rows
            <input
              type="range"
              min={ROW_HEIGHTS.min}
              max={ROW_HEIGHTS.max}
              step={20}
              value={st.rowHeight}
              onChange={(e) => patch({ rowHeight: Number(e.target.value) })}
            />
          </label>
          <span className="muted small fourd-readout" title="⌘/Ctrl-click adds a cursor; right-click or Delete removes one">
            {cursorText}
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="view fourd-view">
      {/* The layout, and the tools for both panes (patterns and virtual images). */}
      <div className="fourd-topbar">
        {layoutSwitch}
        <ImageToolButtons />
      </div>
      <div className="fourd-panes">
      <div className="fourd-pane">
        <LayeredViewer
          viewKey={`${ds.id}:virtual`}
          toolButtons={false}
          base={grid || merge ? null : {
            // A colour COM image keeps its own contrast, on the colour wheel.
            key: shownSaved ? `${shownSaved.id}${shownColour ? ":com" : ""}` : `${ds.id}:virtual${sent.response === "com" ? ":com" : ""}`,
            label: virtualLabel,
            frame: virtual.frame,
            version: virtual.shownKey,
            source: shownSaved ?? ds,
            ...((shownSaved ? shownColour : sent.response === "com") && { colormap: () => "hsv" }),
          }}
          images={[]}
          stacks={grid ? viStacks : merge ? mergeStacks : undefined}
          gridOnly={grid}
          blend={merge && mergeBlend === "max" ? "max" : "add"}
          requestActive={viRequest}
          onActiveChange={grid || merge ? onViActive : undefined}
          empty="Show a detector group (Virtual imaging process) to see its image"
          geometry={scanG}
          loading={grid || merge || virtual.frame ? null : "Computing the virtual image…"}
          error={grid || merge ? viSet.error : virtual.error}
          overlays={false}
          interaction={scanInteraction}
          imageMenu={(p) => [...openInSingle, ...scanMenu(p)]}
          renderOverlay={scanOverlay}
          layersSection="Virtual image"
          extraLayers={viExtra}
          footer={
            <div className="stack-bar fourd-bar">
              {!grid && !merge && imagePicker}
              {merge && segmented("Combine", [
                { id: "dominant", label: "Dominant", title: "At each pixel, only the group brightest there (for its contrast): one colour per pixel" },
                { id: "max", label: "Max", title: "The brighter of the groups' colours, channel by channel" },
                { id: "add", label: "Add", title: "The groups' colours added up" },
              ], mergeBlend, (v) => patch({ mergeBlend: v }))}
              <span className="muted small fourd-readout" title="⌘/Ctrl-click adds a cursor; right-click or Delete removes one">
                {cursorText}
              </span>
            </div>
          }
        >
          {!grid && !merge && <span className="frame-chip">{virtualLabel}</span>}
          {merge && (
            <span className="frame-chip">
              {visibleGroups.map((g, i) => (
                <span key={g.id} style={{ color: g.color }}>{i ? " + " : ""}{g.name}</span>
              ))}
            </span>
          )}
        </LayeredViewer>
      </div>
      <div className="fourd-pane fourd-pattern">
        <LayeredViewer
          viewKey={`${ds.id}:pattern${log ? ":log" : ""}`}
          base={grid ? null : {
            key: `${ds.id}:pattern${log ? ":log" : ""}`,
            label: patternLabel,
            frame: drawnPattern,
            version: `${pattern.shownKey}|${log}`,
          }}
          images={[]}
          stacks={grid ? dpStacks : undefined}
          gridOnly={grid}
          requestActive={dpRequest}
          onActiveChange={grid ? onDpActive : undefined}
          geometry={patternG}
          loading={grid || pattern.frame ? null : "Loading the pattern…"}
          error={grid ? dpSet.error : pattern.error}
          overlays={false}
          toolButtons={false}
          interaction={strainShown && source === SCAN_BVM ? strainInteraction : det.shape === "apertures" ? aperturesInteraction : patternInteraction}
          renderOverlay={patternOverlay}
          imageMenu={(p) => (det.shape === "apertures" && apertureAt(det, p) >= 0 ? [] : openInSingle)}
          layersSection="Diffraction pattern"
          extraLayers={dpExtra}
          bottomTabs={[...matchTabs, ...structureTabs]}
          footer={
            <div className="fourd-footer">
              <div className="stack-bar fourd-bar">
                {!grid && sourcePicker}
                {wholeScan}
                {segmented("Scale", [{ id: "linear", label: "Linear" }, { id: "log", label: "Log", title: "log10 of the intensity" }], log ? "log" : "linear", (v) => setImageLog(logKey, v === "log"))}
              </div>
            </div>
          }
        >
          {!grid && <span className="frame-chip">{patternLabel}</span>}
        </LayeredViewer>
      </div>
      </div>
    </div>
  );
}

