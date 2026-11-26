import { create } from "zustand";

/** A virtual detector on the pattern, in pattern pixels (pixel i spans
 * [i, i + 1)). The backend's ``processing/fourd.py`` ``Detector``. */
export interface Detector {
  shape: "circle" | "annulus" | "rectangle" | "point" | "apertures";
  cx: number;
  cy: number;
  r: number;
  r_inner: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  /** Apertures placed by click (shape "apertures"; radius ``r``). */
  apertures?: { x: number; y: number }[];
  /** A g-vector array of apertures, added to the clicked ones. */
  lattice?: Lattice | null;
}

/** Apertures at O + (i + s1) g1 + (j + s2) g2, i in −n1…n1 and j in −n2…n2
 * (pattern pixels; g1 and g2 as vectors from O). */
export interface Lattice {
  ox: number;
  oy: number;
  g1x: number;
  g1y: number;
  g2x: number;
  g2y: number;
  n1: number;
  n2: number;
  s1: number;
  s2: number;
  /** Include the aperture at O itself (when the shifts put one there). */
  origin: boolean;
}

/** The apertures of a g-vector array that fall on a ``w`` × ``h`` pattern. */
export function latticePoints(l: Lattice, w: number, h: number): { x: number; y: number }[] {
  const out: { x: number; y: number }[] = [];
  for (let i = -l.n1; i <= l.n1; i++) {
    for (let j = -l.n2; j <= l.n2; j++) {
      const a = i + l.s1;
      const b = j + l.s2;
      if (!l.origin && Math.abs(a) < 1e-9 && Math.abs(b) < 1e-9) continue;
      const x = l.ox + a * l.g1x + b * l.g2x;
      const y = l.oy + a * l.g1y + b * l.g2y;
      if (x >= 0 && x <= w && y >= 0 && y <= h) out.push({ x, y });
    }
  }
  return out;
}

/** The detector as the backend takes it: the array's apertures added to the clicked ones. */
export function detectorForRequest(d: Detector, w: number, h: number): Detector {
  if (d.shape !== "apertures") return d;
  return { ...d, apertures: [...(d.apertures ?? []), ...(d.lattice ? latticePoints(d.lattice, w, h) : [])] };
}

/** A g-vector array centred on (cx, cy) with short g vectors along x and y. */
export function defaultLattice(cx: number, cy: number, w: number): Lattice {
  const g = Math.max(4, Math.round(w / 6));
  return { ox: cx, oy: cy, g1x: g, g1y: 0, g2x: 0, g2y: g, n1: 1, n2: 1, s1: 0, s2: 0, origin: true };
}

export type Response = "sum" | "mean" | "max";

/** A cursor on the scan: the scan pixel it is on. */
export interface ScanCursor {
  id: string;
  x: number;
  y: number;
  color: string;
}

export const CURSOR_COLORS = ["#ff9f0a", "#ff453a", "#30d158", "#0a84ff", "#ffffff", "#bf5af2"];

/** The cursor's name, e.g. "Cursor 2" (from its id "c2"). */
export const cursorName = (c: ScanCursor) => `Cursor ${c.id.slice(1)}`;

/** The pattern shown: a cursor's or a region's (their ids; a region's is
 * the mean over it), or the mean or maximum over the whole scan. */
export type PatternSource = string;
export const SCAN_MEAN = "scan-mean";
export const SCAN_MAX = "scan-max";

/** A detector group: one virtual detector (any shape) with its response,
 * name and colour; each gives a virtual image. */
export interface DetectorGroup {
  id: string;
  name: string;
  color: string;
  visible: boolean;
  detector: Detector;
  response: Response;
}

export const GROUP_COLORS = ["#ff6bd6", "#5ac8fa", "#ffd60a", "#8cff5a", "#ff9f0a", "#b18cff"];

/** The selected group (the first if it is gone). */
export const groupOf = (s: FourDState): DetectorGroup => s.groups.find((g) => g.id === s.selectedGroup) ?? s.groups[0];

/** A new group, numbered and coloured after the others; named "Group n" unless ``name``. */
export function newGroup(groups: DetectorGroup[], detector: Detector, name?: string, response: Response = "sum"): DetectorGroup {
  const n = groups.reduce((m, g) => Math.max(m, Number(g.id.slice(1))), 0) + 1;
  const used = new Set(groups.map((g) => g.color));
  const color = GROUP_COLORS.find((c) => !used.has(c)) ?? GROUP_COLORS[(n - 1) % GROUP_COLORS.length];
  return { id: `g${n}`, name: name ?? `Group ${n}`, color, visible: true, detector, response };
}

/** The 4D view of one dataset. */
export interface FourDState {
  groups: DetectorGroup[];
  /** The group edited (handles on the pattern, controls in the process) and shown live. */
  selectedGroup: string;
  cursors: ScanCursor[];
  /** The cursor a press on the scan moves when the pattern shown isn't a cursor's. */
  lastCursor: string;
  source: PatternSource;
  /** The image on the scan: "live" (through the detector) or a saved virtual image's id. */
  shown: string;
  /** One virtual image and one pattern, a grid of each (a virtual image per
   * visible group, a pattern per cursor and region), or a scrolling list of
   * rows (per visible group: the pattern with its detector, its image). */
  layout: "single" | "grid" | "list" | "merge";
  /** The list's row height (px). */
  rowHeight: number;
  /** Merge: the groups' colours added, the brighter kept per channel, or at
   * each pixel only the group that is brightest there (for its contrast). */
  mergeBlend: "add" | "max" | "dominant";
}

export const ROW_HEIGHTS = { min: 160, max: 640, initial: 280 };

/** A new cursor at (x, y), numbered and coloured after the others. */
export function newCursor(cursors: ScanCursor[], x: number, y: number): ScanCursor {
  const n = cursors.reduce((m, c) => Math.max(m, Number(c.id.slice(1))), 0) + 1;
  const used = new Set(cursors.map((c) => c.color));
  const color = CURSOR_COLORS.find((c) => !used.has(c)) ?? CURSOR_COLORS[(n - 1) % CURSOR_COLORS.length];
  return { id: `c${n}`, x, y, color };
}

/** A detector of ``shape`` sized for a ``w`` × ``h`` pattern, centred. */
export function defaultDetector(w: number, h: number, shape: Detector["shape"] = "circle", from?: Detector): Detector {
  const cx = from?.cx ?? w / 2;
  const cy = from?.cy ?? h / 2;
  const r = from && from.shape !== "rectangle" && from.shape !== "point" ? from.r : Math.max(1, Math.min(w, h) / 8);
  const half = from?.shape === "rectangle" ? Math.abs(from.x1 - from.x0) / 2 : r;
  return {
    shape, cx, cy, r: shape === "annulus" ? Math.max(r, 2) : shape === "apertures" ? Math.max(1, Math.min(r, Math.min(w, h) / 16)) : r,
    r_inner: shape === "annulus" ? Math.max(r, 2) / 2 : 0,
    x0: cx - half, y0: cy - half, x1: cx + half, y1: cy + half,
    // Switching shapes keeps the apertures and array placed, for coming back to them.
    apertures: from?.apertures ?? [],
    lattice: from?.lattice ?? null,
  };
}

export function initialState(scan: [number, number], pattern: [number, number]): FourDState {
  const [sw, sh] = scan;
  const [pw, ph] = pattern;
  return {
    groups: [newGroup([], defaultDetector(pw, ph))],
    selectedGroup: "g1",
    cursors: [newCursor([], Math.floor(sw / 2), Math.floor(sh / 2))],
    lastCursor: "c1",
    source: "c1",
    shown: "live",
    layout: "single",
    rowHeight: ROW_HEIGHTS.initial,
    mergeBlend: "dominant",
  };
}

/** Hot and dead detector pixels found in a dataset's mean pattern ([y, x]). */
export interface BadPixels {
  hot: [number, number][];
  dead: [number, number][];
  border: number;
}

interface Store {
  byId: Record<string, FourDState>;
  patch: (id: string, init: FourDState, patch: Partial<FourDState>) => void;
  /** Pixels found by the Bad pixels process, ringed on the pattern. */
  bad: Record<string, BadPixels | null>;
  setBad: (id: string, bad: BadPixels | null) => void;
}

/** Per dataset, kept while the app is open (switching datasets keeps each one's detector and cursor). */
export const useFourD = create<Store>((set) => ({
  byId: {},
  patch: (id, init, patch) => set((s) => ({ byId: { ...s.byId, [id]: { ...(s.byId[id] ?? init), ...patch } } })),
  bad: {},
  setBad: (id, bad) => set((s) => ({ bad: { ...s.bad, [id]: bad } })),
}));

/** The detector moved by (dx, dy) pattern pixels (apertures and the array with it). */
export function moveDetector(d: Detector, dx: number, dy: number): Detector {
  return {
    ...d, cx: d.cx + dx, cy: d.cy + dy, x0: d.x0 + dx, x1: d.x1 + dx, y0: d.y0 + dy, y1: d.y1 + dy,
    ...(d.shape === "apertures" && {
      apertures: (d.apertures ?? []).map((a) => ({ x: a.x + dx, y: a.y + dy })),
      lattice: d.lattice ? { ...d.lattice, ox: d.lattice.ox + dx, oy: d.lattice.oy + dy } : d.lattice,
    }),
  };
}
