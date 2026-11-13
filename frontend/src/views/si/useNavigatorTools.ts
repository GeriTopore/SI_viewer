import { useEffect, useRef, useState } from "react";
import type { Interaction, Pt, Transform } from "../../components/ImageView";
import {
  containsPoint, getSiState, shapeBounds, translateShape, useSi, type Shape,
} from "../../state/si";

type Corner = { x: "x0" | "x1"; y: "y0" | "y1" };

type Drag =
  | { mode: "cursor"; which: 1 | 2; start: Pt; orig: { x: number; y: number } }
  | { mode: "draw"; start: Pt; last: Pt }
  | { mode: "move"; id: string; orig: Shape; start: Pt }
  | { mode: "resize"; id: string; orig: Shape & { kind: "rect" | "ellipse" }; corner: Corner };

const HANDLE_PX = 7;
/** How close (screen px) a press must be to the cursor marker to grab it. */
const GRAB_PX = 8;

/** Rectangle/ellipse with x0 < x1 and y0 < y1. */
function normalise(shape: Shape): Shape {
  if (shape.kind === "polygon") return shape;
  const [x0, y0, x1, y1] = shapeBounds(shape);
  return { ...shape, x0, y0, x1, y1 };
}

/** The rectangle/ellipse drawn by a drag from ``start`` to ``p``: Shift
 * keeps it square (a circle), Alt/Option centres it on the pixel pressed. */
function drawnBox(
  kind: "rect" | "ellipse", start: Pt, p: Pt, square: boolean, centred: boolean, width: number, height: number,
): Shape {
  const clamp = (v: number, hi: number) => Math.min(hi, Math.max(0, v));
  if (centred) {
    const [cx, cy] = [clamp(Math.floor(start.x), width - 1), clamp(Math.floor(start.y), height - 1)];
    // Whole pixels on each side of the pressed one: odd sizes, centred on
    // it, as far as the image goes.
    const [mx, my] = [Math.min(cx, width - 1 - cx), Math.min(cy, height - 1 - cy)];
    let nx = Math.min(mx, Math.max(0, Math.round(Math.abs(p.x - (cx + 0.5)) - 0.5)));
    let ny = Math.min(my, Math.max(0, Math.round(Math.abs(p.y - (cy + 0.5)) - 0.5)));
    if (square) nx = ny = Math.min(Math.max(nx, ny), mx, my);
    return { kind, x0: cx - nx, y0: cy - ny, x1: cx + 1 + nx, y1: cy + 1 + ny };
  }
  // The edges stop at the image's.
  const [x0, y0] = [clamp(Math.round(start.x), width), clamp(Math.round(start.y), height)];
  let [dx, dy] = [clamp(Math.round(p.x), width) - x0, clamp(Math.round(p.y), height) - y0];
  if (square) {
    const room = (d: number, v0: number, n: number) => (d < 0 ? v0 : n - v0);
    const side = Math.min(Math.max(Math.abs(dx), Math.abs(dy)), room(dx, x0, width), room(dy, y0, height));
    dx = (dx < 0 ? -1 : 1) * side;
    dy = (dy < 0 ? -1 : 1) * side;
  }
  return { kind, x0, y0, x1: x0 + dx, y1: y0 + dy };
}

/** Removes repeated vertices (a double-click adds the last one twice). */
function dedupe(points: [number, number][]) {
  return points.filter((p, i) => i === 0 || p[0] !== points[i - 1][0] || p[1] !== points[i - 1][1]);
}

/**
 * Pointer and keyboard handling for the SI navigator:
 * - cursor tool: the spectrum cursor moves only by dragging its marker (a
 *   press within a few px of it; snapped to pixels) — a click or drag
 *   elsewhere leaves it alone (the image pans); arrows nudge it (Shift: ×10;
 *   with ``frameArrows``, ←/→ are the frame keys and ⌥←/⌥→ nudge it)
 * - rect/ellipse: drag draws a region (Shift: square/circle; Alt: centred
 *   on the pixel pressed); drag inside a region moves it
 * - polygon: click adds vertices; click the first vertex, double-click or
 *   Enter closes it; Esc cancels
 * - the selected region's corner handles resize it with any tool;
 *   Delete removes it
 */
export function useNavigatorTools(
  si: string, width: number, height: number, frameArrows = false, toolOverride?: "rect" | "ellipse" | "polygon",
) {
  // Spectrum cursors: on a spectrum image's navigator, not on plain images' regions.
  const cursors = !toolOverride;
  const storeTool = useSi((s) => s.tool);
  const tool = toolOverride ?? storeTool;
  const cursorSize = useSi((s) => s.cursorSize);
  const { patch, addRoi, updateRoi, deleteRois } = useSi.getState();
  const [draft, setDraft] = useState<Shape | null>(null);
  const [hover, setHover] = useState<Pt | null>(null);
  const drag = useRef<Drag | null>(null);
  // The pointer is over the cursor marker / dragging it (sets the mouse cursor).
  const [near, setNear] = useState(false);
  const [grabbing, setGrabbing] = useState(false);

  // A region clicked or drawn is shown (and edited) in the Info tab, until
  // it is deselected or a dataset is clicked.
  const focusRegion = () => useSi.getState().setRegionView({ key: si, width, height });
  const selected = useSi((s) => s.bySi[si]?.selected ?? null);
  useEffect(() => {
    const clear = () => {
      if (useSi.getState().regionView?.key === si) useSi.getState().setRegionView(null);
    };
    if (!selected) clear();
  }, [si, selected]);
  useEffect(
    () => () => {
      if (useSi.getState().regionView?.key === si) useSi.getState().setRegionView(null);
    },
    [si],
  );

  const snap = Math.round; // region edges sit on pixel edges
  // Regions stay on the image: points kept within its edges.
  const inX = (v: number) => Math.min(width, Math.max(0, v));
  const inY = (v: number) => Math.min(height, Math.max(0, v));
  const toCursor = (p: Pt) => ({
    x: Math.min(width - 1, Math.max(0, Math.floor(p.x))),
    y: Math.min(height - 1, Math.max(0, Math.floor(p.y))),
  });
  const setCursor = (p: Pt, which: 1 | 2 = 1) => {
    const c = toCursor(p);
    const cur = which === 1 ? getSiState(si).cursor : getSiState(si).cursor2;
    if (!cur || cur.x !== c.x || cur.y !== c.y)
      patch(si, which === 1 ? { cursor: c, showCursor: true } : { cursor2: c, showCursor2: true });
  };

  /** The pointer is within GRAB_PX screen pixels of a cursor's marker. */
  const onMarker = (p: Pt, t: Transform, which: 1 | 2 = 1) => {
    const s = getSiState(si);
    const c = which === 1 ? s.cursor : s.cursor2;
    if (!c || !(which === 1 ? s.showCursor : s.showCursor2)) return false;
    const half = Math.floor((cursorSize - 1) / 2);
    const [x0, y0] = t.toScreen(c.x - half, c.y - half);
    const [x1, y1] = t.toScreen(c.x - half + cursorSize, c.y - half + cursorSize);
    const [px, py] = t.toScreen(p.x, p.y);
    return Math.hypot(Math.max(x0 - px, 0, px - x1), Math.max(y0 - py, 0, py - y1)) <= GRAB_PX;
  };

  const cornerAt = (shape: Shape, p: Pt, t: Transform): Corner | null => {
    if (shape.kind === "polygon") return null;
    const n = normalise(shape) as Shape & { kind: "rect" | "ellipse" };
    const tol = HANDLE_PX / t.zoom;
    for (const x of ["x0", "x1"] as const) {
      for (const y of ["y0", "y1"] as const) {
        if (Math.abs(p.x - n[x]) <= tol && Math.abs(p.y - n[y]) <= tol) return { x, y };
      }
    }
    return null;
  };

  /** A press on the image: a region there is selected and shown in the Info
   * tab; elsewhere (not on the selected one's handles) the region is
   * deselected and the Info tab shows the dataset again. Returns the region. */
  const regionPress = (p: Pt, t: Transform) => {
    const s = getSiState(si);
    const sel = s.rois.find((r) => r.id === s.selected && r.visible);
    if (sel && cornerAt(sel.shape, p, t)) return sel;
    const hit = [...s.rois].reverse().find((r) => r.visible && containsPoint(r.shape, p.x, p.y));
    if (hit) {
      if (s.selected !== hit.id) patch(si, { selected: hit.id });
      focusRegion();
    } else {
      if (s.selected) patch(si, { selected: null });
      if (useSi.getState().regionView?.key === si) useSi.getState().setRegionView(null);
    }
    return hit ?? null;
  };

  const finishPolygon = () => {
    if (draft?.kind !== "polygon") return;
    const points = dedupe(draft.points);
    if (points.length >= 3) {
      addRoi(si, { kind: "polygon", points });
      focusRegion();
    }
    setDraft(null);
  };

  const redraw = (start: Pt, p: Pt, square: boolean, centred: boolean) =>
    setDraft((cur) => (cur && cur.kind !== "polygon" ? drawnBox(cur.kind, start, p, square, centred, width, height) : cur));
  // Shift or Alt pressed or let go mid-drag: the drawn shape follows at once.
  const drawing = draft !== null && draft.kind !== "polygon";
  useEffect(() => {
    if (!drawing) return;
    const onKey = (e: KeyboardEvent) => {
      const d = drag.current;
      if (d?.mode !== "draw" || (e.key !== "Shift" && e.key !== "Alt")) return;
      redraw(d.start, d.last, e.shiftKey, e.altKey);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKey);
    };
  }, [drawing]);

  const interaction: Interaction = {
    cursor:
      tool === "cursor" ? (grabbing ? "grabbing" : near ? "move" : undefined) : "cell",

    onDown(p, e, t) {
      if (tool === "line" || tool === "fft") return false; // lines, FFT boxes: the viewer's tools
      // ⌘-click (Ctrl-click): the second cursor there, dragged on; with the
      // cursor tool its marker is also dragged as is.
      if (cursors && (e.metaKey || e.ctrlKey) && !draft) {
        setCursor(p, 2);
        drag.current = { mode: "cursor", which: 2, start: p, orig: toCursor(p) };
        setGrabbing(true);
        return true;
      }
      if (cursors && tool === "cursor" && onMarker(p, t, 2)) {
        drag.current = { mode: "cursor", which: 2, start: p, orig: getSiState(si).cursor2! };
        setGrabbing(true);
        return true;
      }
      if (tool !== "polygon" || !draft) regionPress(p, t);
      const s = getSiState(si);
      const selected = s.rois.find((r) => r.id === s.selected && r.visible);
      if (selected && tool !== "polygon") {
        const corner = cornerAt(selected.shape, p, t);
        if (corner) {
          const orig = normalise(selected.shape) as Shape & { kind: "rect" | "ellipse" };
          drag.current = { mode: "resize", id: selected.id, orig, corner };
          return true;
        }
      }
      if (tool === "cursor") {
        // No cursor shown: put it where the image was clicked (and drag on).
        if (!s.cursor || !s.showCursor) {
          setCursor(p);
          drag.current = { mode: "cursor", which: 1, start: p, orig: toCursor(p) };
          setGrabbing(true);
          return true;
        }
        const cur = s.cursor;
        if (!onMarker(p, t)) return false; // elsewhere the image pans
        drag.current = { mode: "cursor", which: 1, start: p, orig: cur };
        setGrabbing(true);
        return true;
      }
      if (tool === "polygon") {
        const pts = draft?.kind === "polygon" ? draft.points : [];
        if (pts.length >= 3) {
          const [fx, fy] = t.toScreen(...pts[0]);
          const [px, py] = t.toScreen(p.x, p.y);
          if (Math.hypot(fx - px, fy - py) < 8) {
            finishPolygon();
            return true;
          }
        }
        setDraft({ kind: "polygon", points: [...pts, [inX(p.x), inY(p.y)]] });
        return true;
      }
      // Region tools: dragging inside an existing region moves it.
      const hit = [...s.rois].reverse().find((r) => r.visible && containsPoint(r.shape, p.x, p.y));
      if (hit) {
        drag.current = { mode: "move", id: hit.id, orig: hit.shape, start: p };
        return true;
      }
      const x = inX(snap(p.x));
      const y = inY(snap(p.y));
      setDraft({ kind: tool, x0: x, y0: y, x1: x, y1: y });
      drag.current = { mode: "draw", start: p, last: p };
      return true;
    },

    onMove(p, e) {
      const d = drag.current;
      if (!d) return;
      if (d.mode === "cursor") {
        // Move by whole pixels from where the marker was grabbed.
        setCursor({
          x: d.orig.x + Math.floor(p.x) - Math.floor(d.start.x) + 0.5,
          y: d.orig.y + Math.floor(p.y) - Math.floor(d.start.y) + 0.5,
        }, d.which);
      } else if (d.mode === "draw") {
        d.last = p;
        redraw(d.start, p, e.shiftKey, e.altKey);
      } else if (d.mode === "move") {
        // Moved as far as the image's edges.
        const [bx0, by0, bx1, by1] = shapeBounds(d.orig);
        const dx = Math.min(width - bx1, Math.max(-bx0, Math.round(p.x - d.start.x)));
        const dy = Math.min(height - by1, Math.max(-by0, Math.round(p.y - d.start.y)));
        updateRoi(si, d.id, { shape: translateShape(d.orig, dx, dy) });
      } else {
        updateRoi(si, d.id, { shape: { ...d.orig, [d.corner.x]: inX(snap(p.x)), [d.corner.y]: inY(snap(p.y)) } });
      }
    },

    onUp() {
      const d = drag.current;
      drag.current = null;
      if (d?.mode === "cursor") setGrabbing(false);
      else if (d?.mode === "draw" && draft && draft.kind !== "polygon") {
        const n = normalise(draft) as Shape & { kind: "rect" | "ellipse" };
        if (n.x1 - n.x0 >= 1 && n.y1 - n.y0 >= 1) {
          addRoi(si, n);
          focusRegion();
        }
        setDraft(null);
      } else if (d?.mode === "resize") {
        const roi = getSiState(si).rois.find((r) => r.id === d.id);
        if (roi) updateRoi(si, d.id, { shape: normalise(roi.shape) });
      }
    },

    onHover(p, t) {
      if (tool === "polygon") setHover(p);
      setNear(tool === "cursor" && !!p && !!t && (onMarker(p, t) || (cursors && onMarker(p, t, 2))));
    },

    onDoubleClick() {
      if (tool !== "polygon") return false;
      finishPolygon();
      return true;
    },

    onKeyDown(e) {
      const s = getSiState(si);
      const step = e.shiftKey ? 10 : 1;
      const arrows: Record<string, [number, number]> = {
        ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step],
      };
      // In-situ series: ←/→ step the frames, so the cursor moves sideways with ⌥←/⌥→.
      if (frameArrows && (e.key === "ArrowLeft" || e.key === "ArrowRight") && !e.altKey) return false;
      if (e.key in arrows && s.cursor) {
        const [dx, dy] = arrows[e.key];
        setCursor({ x: s.cursor.x + dx + 0.5, y: s.cursor.y + dy + 0.5 });
        return true;
      }
      if ((e.key === "Delete" || e.key === "Backspace") && s.selected) {
        deleteRois(si, [s.selected]);
        return true;
      }
      if (e.key === "Escape") {
        if (draft) setDraft(null);
        else patch(si, { selected: null });
        return true;
      }
      if (e.key === "Enter" && draft?.kind === "polygon") {
        finishPolygon();
        return true;
      }
      return false;
    },
  };

  return { interaction, draft, hover, regionPress };
}
