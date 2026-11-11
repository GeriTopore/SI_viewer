import { useRef } from "react";
import type { Interaction, Pt } from "../components/ImageView";
import { getViewProfiles, useProfiles, type LineGeometry, type ProfileLine } from "../state/profiles";
import { useUi } from "../state/ui";
import { askDeleteLine } from "../components/LineDeleteDialog";
import { distanceToLine } from "../viewer/profile";

type Drag =
  | { mode: "draw"; id: string; start: Pt }
  | { mode: "end"; id: string; end: 0 | 1 }
  | { mode: "move"; id: string; orig: LineGeometry; start: Pt };

const HANDLE_PX = 7;
// Shift-drag snaps the angle to multiples of this.
const SNAP_RAD = Math.PI / 12;

/** Endpoint at ``p`` relative to ``from``, its angle snapped when Shift is held. */
function endpoint(from: Pt, p: Pt, snap: boolean): Pt {
  if (!snap) return p;
  const len = Math.hypot(p.x - from.x, p.y - from.y);
  const a = Math.round(Math.atan2(p.y - from.y, p.x - from.x) / SNAP_RAD) * SNAP_RAD;
  return { x: from.x + len * Math.cos(a), y: from.y + len * Math.sin(a) };
}

/**
 * Line tool for profiles: drag to draw a line (Shift snaps the angle to
 * 15°), drag an endpoint or the line itself to change it, Delete removes the
 * selected line, Esc deselects. Drawing or changing a line shows the Line
 * profile tab under the image. ``view`` is the image view's key.
 */
/** The visible line of ``view`` under ``p`` (the top one), within a few
 * screen pixels or its width. */
export function lineAt(view: string, p: { x: number; y: number }, zoom: number): ProfileLine | undefined {
  const tol = HANDLE_PX / zoom;
  const visible = getViewProfiles(view).lines.filter((l) => l.visible);
  return [...visible].reverse().find((l) => distanceToLine(l, p.x, p.y) <= Math.max(tol, l.width / 2));
}

export function useLineTool(view: string): Interaction {
  const { patch, addLine, updateLine, removeLine } = useProfiles.getState();
  const drag = useRef<Drag | null>(null);

  return {
    cursor: "crosshair",

    onDown(p, _e, t) {
      // Every drag below draws or changes a line: show its profile.
      useUi.getState().setBottomTab("profile");
      const s = getViewProfiles(view);
      const tol = HANDLE_PX / t.zoom;
      const visible = s.lines.filter((l) => l.visible);
      // Endpoints first (the selected line's win), then line bodies.
      const ordered = [...visible].sort((a, b) => Number(b.id === s.selected) - Number(a.id === s.selected));
      for (const l of ordered) {
        for (const end of [0, 1] as const) {
          const [x, y] = end === 0 ? [l.x0, l.y0] : [l.x1, l.y1];
          if (Math.hypot(p.x - x, p.y - y) <= tol) {
            patch(view, { selected: l.id });
            drag.current = { mode: "end", id: l.id, end };
            return true;
          }
        }
      }
      const hit = [...visible].reverse().find((l) => distanceToLine(l, p.x, p.y) <= Math.max(tol, l.width / 2));
      if (hit) {
        patch(view, { selected: hit.id });
        drag.current = { mode: "move", id: hit.id, orig: { x0: hit.x0, y0: hit.y0, x1: hit.x1, y1: hit.y1 }, start: p };
        return true;
      }
      const id = addLine(view, { x0: p.x, y0: p.y, x1: p.x, y1: p.y });
      drag.current = { mode: "draw", id, start: p };
      return true;
    },

    onMove(p, e) {
      const d = drag.current;
      if (!d) return;
      if (d.mode === "draw") {
        const q = endpoint(d.start, p, e.shiftKey);
        updateLine(view, d.id, { x1: q.x, y1: q.y });
      } else if (d.mode === "end") {
        const l = getViewProfiles(view).lines.find((x) => x.id === d.id);
        if (!l) return;
        const fixed = d.end === 0 ? { x: l.x1, y: l.y1 } : { x: l.x0, y: l.y0 };
        const q = endpoint(fixed, p, e.shiftKey);
        updateLine(view, d.id, d.end === 0 ? { x0: q.x, y0: q.y } : { x1: q.x, y1: q.y });
      } else {
        const dx = p.x - d.start.x;
        const dy = p.y - d.start.y;
        const o = d.orig;
        updateLine(view, d.id, { x0: o.x0 + dx, y0: o.y0 + dy, x1: o.x1 + dx, y1: o.y1 + dy });
      }
    },

    onUp() {
      const d = drag.current;
      drag.current = null;
      if (d?.mode !== "draw") return;
      // A click without a drag draws nothing (and deselects).
      const l = getViewProfiles(view).lines.find((x) => x.id === d.id);
      if (l && Math.hypot(l.x1 - l.x0, l.y1 - l.y0) < 1) {
        removeLine(view, d.id);
        patch(view, { selected: null });
      }
    },

    onKeyDown(e) {
      const s = getViewProfiles(view);
      if ((e.key === "Delete" || e.key === "Backspace") && s.selected) {
        askDeleteLine(view, s.selected);
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
