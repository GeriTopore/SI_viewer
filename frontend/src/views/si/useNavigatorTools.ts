import { useRef, useState } from "react";
import type { Interaction, Pt, Transform } from "../../components/ImageView";
import {
  containsPoint, getSiState, shapeBounds, translateShape, useSi, type Shape,
} from "../../state/si";

type Corner = { x: "x0" | "x1"; y: "y0" | "y1" };

type Drag =
  | { mode: "cursor"; start: Pt; orig: { x: number; y: number } }
  | { mode: "draw" }
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

/** Removes repeated vertices (a double-click adds the last one twice). */
function dedupe(points: [number, number][]) {
  return points.filter((p, i) => i === 0 || p[0] !== points[i - 1][0] || p[1] !== points[i - 1][1]);
}

/**
 * Pointer and keyboard handling for the SI navigator:
 * - cursor tool: the spectrum cursor moves only by dragging its marker (a
 *   press within a few px of it; snapped to pixels) — a click or drag
 *   elsewhere leaves it alone (the image pans); arrows nudge it (Shift: ×10)
 * - rect/ellipse: drag draws a region; drag inside a region moves it
 * - polygon: click adds vertices; click the first vertex, double-click or
 *   Enter closes it; Esc cancels
 * - the selected region's corner handles resize it with any tool;
 *   Delete removes it
 */
export function useNavigatorTools(si: string, width: number, height: number) {
  const tool = useSi((s) => s.tool);
  const cursorSize = useSi((s) => s.cursorSize);
  const { patch, addRoi, updateRoi, removeRoi } = useSi.getState();
  const [draft, setDraft] = useState<Shape | null>(null);
  const [hover, setHover] = useState<Pt | null>(null);
  const drag = useRef<Drag | null>(null);
  // The pointer is over the cursor marker / dragging it (sets the mouse cursor).
  const [near, setNear] = useState(false);
  const [grabbing, setGrabbing] = useState(false);

  const snap = Math.round; // region edges sit on pixel edges
  const toCursor = (p: Pt) => ({
    x: Math.min(width - 1, Math.max(0, Math.floor(p.x))),
    y: Math.min(height - 1, Math.max(0, Math.floor(p.y))),
  });
  const setCursor = (p: Pt) => {
    const c = toCursor(p);
    const cur = getSiState(si).cursor;
    if (!cur || cur.x !== c.x || cur.y !== c.y) patch(si, { cursor: c, showCursor: true });
  };

  /** The pointer is within GRAB_PX screen pixels of the cursor marker. */
  const onMarker = (p: Pt, t: Transform) => {
    const c = getSiState(si).cursor;
    if (!c || !getSiState(si).showCursor) return false;
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

  const finishPolygon = () => {
    if (draft?.kind !== "polygon") return;
    const points = dedupe(draft.points);
    if (points.length >= 3) addRoi(si, { kind: "polygon", points });
    setDraft(null);
  };

  const interaction: Interaction = {
    cursor:
      tool === "pan" ? "grab" : tool === "cursor" ? (grabbing ? "grabbing" : near ? "move" : undefined) : "cell",

    onDown(p, _e, t) {
      if (tool === "pan" || tool === "line") return false; // lines: the viewer's line tool
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
        const cur = s.cursor;
        if (!cur || !onMarker(p, t)) return false; // elsewhere the image pans
        drag.current = { mode: "cursor", start: p, orig: cur };
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
        setDraft({ kind: "polygon", points: [...pts, [p.x, p.y]] });
        return true;
      }
      // Region tools: dragging inside an existing region moves it.
      const hit = [...s.rois].reverse().find((r) => r.visible && containsPoint(r.shape, p.x, p.y));
      if (hit) {
        patch(si, { selected: hit.id });
        drag.current = { mode: "move", id: hit.id, orig: hit.shape, start: p };
        return true;
      }
      const x = snap(p.x);
      const y = snap(p.y);
      setDraft({ kind: tool, x0: x, y0: y, x1: x, y1: y });
      drag.current = { mode: "draw" };
      return true;
    },

    onMove(p) {
      const d = drag.current;
      if (!d) return;
      if (d.mode === "cursor") {
        // Move by whole pixels from where the marker was grabbed.
        setCursor({
          x: d.orig.x + Math.floor(p.x) - Math.floor(d.start.x) + 0.5,
          y: d.orig.y + Math.floor(p.y) - Math.floor(d.start.y) + 0.5,
        });
      } else if (d.mode === "draw") {
        setDraft((cur) => (cur && cur.kind !== "polygon" ? { ...cur, x1: snap(p.x), y1: snap(p.y) } : cur));
      } else if (d.mode === "move") {
        updateRoi(si, d.id, {
          shape: translateShape(d.orig, Math.round(p.x - d.start.x), Math.round(p.y - d.start.y)),
        });
      } else {
        updateRoi(si, d.id, { shape: { ...d.orig, [d.corner.x]: snap(p.x), [d.corner.y]: snap(p.y) } });
      }
    },

    onUp() {
      const d = drag.current;
      drag.current = null;
      if (d?.mode === "cursor") setGrabbing(false);
      else if (d?.mode === "draw" && draft && draft.kind !== "polygon") {
        const n = normalise(draft) as Shape & { kind: "rect" | "ellipse" };
        if (n.x1 - n.x0 >= 1 && n.y1 - n.y0 >= 1) addRoi(si, n);
        setDraft(null);
      } else if (d?.mode === "resize") {
        const roi = getSiState(si).rois.find((r) => r.id === d.id);
        if (roi) updateRoi(si, d.id, { shape: normalise(roi.shape) });
      }
    },

    onHover(p, t) {
      if (tool === "polygon") setHover(p);
      setNear(tool === "cursor" && !!p && !!t && onMarker(p, t));
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
      if (e.key in arrows && s.cursor) {
        const [dx, dy] = arrows[e.key];
        setCursor({ x: s.cursor.x + dx + 0.5, y: s.cursor.y + dy + 0.5 });
        return true;
      }
      if ((e.key === "Delete" || e.key === "Backspace") && s.selected) {
        removeRoi(si, s.selected);
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

  return { interaction, draft, hover };
}
