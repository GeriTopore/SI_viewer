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
  /** Apertures placed by click (shape "apertures"; radius ``r``, or their
   * own when kept from a group merged in). */
  apertures?: Aperture[];
  /** A g-vector array of apertures, added to the clicked ones. */
  lattice?: Lattice | null;
}

export interface Aperture {
  x: number;
  y: number;
  /** Its own radius (from a group merged in); unset: the detector's. */
  r?: number;
}

/** An aperture's radius: its own, else its detector's. */
export const apertureR = (d: Detector, a: Aperture) => a.r ?? d.r;

/** Every aperture of ``d`` of radius ``r`` (their own sizes dropped). */
export const sizeApertures = (d: Detector, r: number): Detector => ({
  ...d, r, apertures: (d.apertures ?? []).map(({ x, y }) => ({ x, y })),
});

/** Can these detectors be merged (apertures and circles: as apertures)? */
export const mergeable = (d: Detector) => d.shape === "apertures" || d.shape === "circle";

/** ``target`` with ``source`` merged in: one apertures detector, the
 * target's radius kept, the source's apertures (its array's too, and a
 * circle as one aperture) keeping theirs. Null if either can't merge. */
export function mergeDetectors(target: Detector, source: Detector, w: number, h: number): Detector | null {
  if (!mergeable(target) || !mergeable(source)) return null;
  const asPoints = (d: Detector): Aperture[] =>
    d.shape === "circle"
      ? [{ x: d.cx, y: d.cy, r: d.r }]
      : (detectorForRequest(d, w, h).apertures ?? []).map((a) => ({ ...a, r: apertureR(d, a) }));
  const base: Detector = target.shape === "circle" ? { ...target, shape: "apertures", apertures: [], lattice: target.lattice ?? null } : target;
  const own = target.shape === "circle" ? [{ x: target.cx, y: target.cy }] : (target.apertures ?? []);
  // The source's apertures of the target's size need no radius of their own.
  const added = asPoints(source).map((a) => (a.r === base.r ? { x: a.x, y: a.y } : a));
  return { ...base, apertures: [...own, ...added] };
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

export type Response = "sum" | "mean" | "max" | "com" | "com_x" | "com_y" | "icom";

/** The centre of mass of the pixels inside the detector, from the pattern's
 * centre as in the data: in colour ("com": direction and magnitude), its x
 * and y, and that field integrated (iCOM). */
export const isComResponse = (r: Response) => r === "com" || r === "com_x" || r === "com_y" || r === "icom";

/** A response as written in labels, e.g. "COM x". */
export const responseLabel = (r: Response) =>
  ({ sum: "sum", mean: "mean", max: "max", com: "COM", com_x: "COM x", com_y: "COM y", icom: "iCOM" })[r];

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
 * the mean over it), the mean or maximum over the whole scan, or the Bragg
 * vector map (on the pattern's pixels; once the disks are found). */
export type PatternSource = string;
export const SCAN_MEAN = "scan-mean";
export const SCAN_MAX = "scan-max";
export const SCAN_BVM = "scan-bvm";
/** Sources over the whole scan (not a cursor or region). */
export const isScanSource = (s: string) => s === SCAN_MEAN || s === SCAN_MAX || s === SCAN_BVM;

/** A detector group: one virtual detector (any shape) with its response,
 * name and colour; each gives a virtual image. */
export interface DetectorGroup {
  id: string;
  name: string;
  color: string;
  visible: boolean;
  detector: Detector;
  response: Response;
  /** Its image from the patterns' pixels inside the detector (default), or
   * from the Bragg vectors: a digital dark field (the disks inside it; an
   * aperture's radius is then the match tolerance). */
  source?: ImageSource;
  /** Digital dark field: the disks' intensity (default), or their number. */
  weight?: "intensity" | "count";
  /** Made from the disks found (apertures on them): which are kept. */
  disks?: DisksOptions;
}

/** Apertures on the disks found at a cursor: the brightest ``brightest``
 * (0: all), the direct beam's too if ``beam``. */
export interface DisksOptions {
  brightest: number;
  beam: boolean;
}

export type ImageSource = "pixels" | "vectors";

/** What the backend takes to make a group's image (``api/fourd.py``
 * ``VirtualBody``); from its pixels unless its data has Bragg vectors
 * (``vectors``). */
export function imageRequest(g: DetectorGroup, w: number, h: number, vectors: boolean) {
  const source = (g.source === "vectors" && vectors ? "vectors" : "pixels") as ImageSource;
  // A centre of mass is of the pixels: a digital dark field sums instead.
  const response: Response = source === "vectors" && isComResponse(g.response) ? "sum" : g.response;
  return {
    detector: detectorForRequest(g.detector, w, h),
    response,
    source,
    weight: g.weight ?? "intensity",
  };
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

/** A circle: one aperture at the pattern's centre (⌘/Ctrl-click adds more). */
export function circleDetector(w: number, h: number): Detector {
  const r = Math.max(1, Math.min(w, h) / 8);
  return { ...defaultDetector(w, h, "apertures"), r, apertures: [{ x: w / 2, y: h / 2 }] };
}

export function initialState(scan: [number, number], pattern: [number, number]): FourDState {
  const [sw, sh] = scan;
  const [pw, ph] = pattern;
  return {
    groups: [newGroup([], circleDetector(pw, ph))],
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

/** How a mark drawn over the pattern or the scan shows (Layers tab): a
 * group's detector, the Bragg disks, a cursor. Unset: its default. */
export interface OverlayLayer {
  visible?: boolean;
  opacity?: number;
  colour?: string;
  /** Bragg disks: a circle of the disk's size, or a dot at its position. */
  style?: "circle" | "dot";
  /** A dot's size (× its default). */
  size?: number;
}

interface Store {
  byId: Record<string, FourDState>;
  patch: (id: string, init: FourDState, patch: Partial<FourDState>) => void;
  /** Pixels found by the Bad pixels process, ringed on the pattern. */
  bad: Record<string, BadPixels | null>;
  setBad: (id: string, bad: BadPixels | null) => void;
  /** Per dataset, per mark ("det:<group>", "disks", "cursor:<id>"). */
  overlays: Record<string, Record<string, OverlayLayer>>;
  setOverlay: (id: string, key: string, patch: OverlayLayer) => void;
}

/** Per dataset, kept while the app is open (switching datasets keeps each one's detector and cursor). */
export const useFourD = create<Store>((set) => ({
  byId: {},
  patch: (id, init, patch) => set((s) => ({ byId: { ...s.byId, [id]: { ...(s.byId[id] ?? init), ...patch } } })),
  bad: {},
  setBad: (id, bad) => set((s) => ({ bad: { ...s.bad, [id]: bad } })),
  overlays: {},
  setOverlay: (id, key, patch) =>
    set((s) => ({
      overlays: { ...s.overlays, [id]: { ...s.overlays[id], [key]: { ...s.overlays[id]?.[key], ...patch } } },
    })),
}));

/** The detector moved by (dx, dy) pattern pixels (apertures and the array with it). */
export function moveDetector(d: Detector, dx: number, dy: number): Detector {
  return {
    ...d, cx: d.cx + dx, cy: d.cy + dy, x0: d.x0 + dx, x1: d.x1 + dx, y0: d.y0 + dy, y1: d.y1 + dy,
    ...(d.shape === "apertures" && {
      apertures: (d.apertures ?? []).map((a) => ({ ...a, x: a.x + dx, y: a.y + dy })),
      lattice: d.lattice ? { ...d.lattice, ox: d.lattice.ox + dx, oy: d.lattice.oy + dy } : d.lattice,
    }),
  };
}
