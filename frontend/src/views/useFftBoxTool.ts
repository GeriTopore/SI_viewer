import { useRef, useState } from "react";
import { GRAB, GRABBING, type Interaction, type Pt, type Transform } from "../components/ImageView";
import { getViewFftBoxes, useFftBoxes, type BoxGeometry, type FftBox } from "../state/fftBoxes";

type Corner = 0 | 1 | 2 | 3; // top-left, top-right, bottom-right, bottom-left

type Drag =
  | { mode: "draw"; id: string; start: Pt; source: string | null }
  | { mode: "corner"; id: string; anchor: Pt }
  | { mode: "move"; id: string; orig: BoxGeometry; start: Pt };

const HANDLE_PX = 7;
/** Smallest box (the FFT panel needs at least 16 pixels). */
export const MIN_BOX = 16;

const corners = (b: BoxGeometry): Pt[] => [
  { x: b.x, y: b.y },
  { x: b.x + b.size, y: b.y },
  { x: b.x + b.size, y: b.y + b.size },
  { x: b.x, y: b.y + b.size },
];

/** The square with one corner at ``anchor`` reaching towards ``p`` (side: the
 * larger of the two distances), in whole pixels and inside the image. */
function square(anchor: Pt, p: Pt, width: number, height: number): BoxGeometry {
  const sx = p.x >= anchor.x ? 1 : -1;
  const sy = p.y >= anchor.y ? 1 : -1;
  const ax = Math.round(anchor.x);
  const ay = Math.round(anchor.y);
  // Room to the image edge in each drag direction.
  const room = Math.min(sx > 0 ? width - ax : ax, sy > 0 ? height - ay : ay);
  const size = Math.max(0, Math.min(Math.round(Math.max(Math.abs(p.x - ax), Math.abs(p.y - ay))), room));
  return { x: sx > 0 ? ax : ax - size, y: sy > 0 ? ay : ay - size, size };
}

const inside = (b: FftBox, p: Pt) => p.x >= b.x && p.x <= b.x + b.size && p.y >= b.y && p.y <= b.y + b.size;

/**
 * FFT box tool: drag to draw a square (its FFT is shown in the FFT panel),
 * drag a corner to resize it, drag inside to move it; Delete removes the
 * selected box, Esc deselects. ``view`` is the image view's key; boxes stay
 * inside its ``width`` × ``height`` pixels.
 */
export function useFftBoxTool(view: string, width: number, height: number): Interaction {
  const { patch, addBox, updateBox, removeBox } = useFftBoxes.getState();
  const drag = useRef<Drag | null>(null);
  const [cursor, setCursor] = useState("crosshair");

  const hit = (p: Pt, t: Transform): { box: FftBox; corner: Corner | null } | null => {
    const s = getViewFftBoxes(view);
    const tol = HANDLE_PX / t.zoom;
    // The selected box's corners win, then any corner, then insides (topmost first).
    const ordered = [...s.boxes].reverse().sort((a, b) => Number(b.id === s.selected) - Number(a.id === s.selected));
    for (const box of ordered) {
      const c = corners(box).findIndex((q) => Math.hypot(p.x - q.x, p.y - q.y) <= tol);
      if (c >= 0) return { box, corner: c as Corner };
    }
    const box = ordered.find((b) => inside(b, p));
    return box ? { box, corner: null } : null;
  };

  return {
    cursor,

    onHover(p, t) {
      setCursor(p && t && hit(p, t) ? GRAB : "crosshair");
    },

    onDown(p, _e, t) {
      const h = hit(p, t);
      if (h) {
        setCursor(GRABBING);
        // Touching a box shows its FFT.
        patch(view, { selected: h.box.id, source: h.box.id });
        if (h.corner !== null) {
          drag.current = { mode: "corner", id: h.box.id, anchor: corners(h.box)[(h.corner + 2) % 4] };
        } else {
          drag.current = { mode: "move", id: h.box.id, orig: { x: h.box.x, y: h.box.y, size: h.box.size }, start: p };
        }
        return true;
      }
      const { source } = getViewFftBoxes(view);
      const id = addBox(view, square(p, p, width, height));
      drag.current = { mode: "draw", id, start: p, source };
      return true;
    },

    onMove(p) {
      const d = drag.current;
      if (!d) return;
      if (d.mode === "draw") updateBox(view, d.id, square(d.start, p, width, height));
      else if (d.mode === "corner") {
        const g = square(d.anchor, p, width, height);
        if (g.size >= MIN_BOX) updateBox(view, d.id, g);
      } else {
        const o = d.orig;
        const x = Math.min(Math.max(Math.round(o.x + p.x - d.start.x), 0), width - o.size);
        const y = Math.min(Math.max(Math.round(o.y + p.y - d.start.y), 0), height - o.size);
        updateBox(view, d.id, { x, y });
      }
    },

    onUp(p, _e, t) {
      const d = drag.current;
      drag.current = null;
      setCursor(hit(p, t) ? GRAB : "crosshair");
      if (d?.mode !== "draw") return;
      // A click (or a box too small to transform) draws nothing and
      // deselects; the FFT keeps its region.
      const b = getViewFftBoxes(view).boxes.find((x) => x.id === d.id);
      if (b && b.size < MIN_BOX) {
        removeBox(view, d.id);
        patch(view, { selected: null, source: d.source });
      }
    },

    onKeyDown(e) {
      const s = getViewFftBoxes(view);
      if ((e.key === "Delete" || e.key === "Backspace") && s.selected) {
        useFftBoxes.getState().deleteBox(view, s.selected);
        return true;
      }
      if (e.key === "Escape" && s.selected) {
        patch(view, { selected: null });
        return true;
      }
      return false;
    },
  };
}
