import { useEffect, useRef } from "react";

/** True when a key event comes from a text field or menu (global shortcuts
 * should then leave it alone). */
export const isTyping = (e: Event) =>
  e.target instanceof Element && !!e.target.closest("input, textarea, select");

/** True while a dialog is open (it owns the keyboard). */
const dialogOpen = () => !!document.querySelector(".modal-backdrop");

/**
 * Single-key shortcuts (no ⌘/Ctrl/Alt), e.g. ``{ f: fit, "1": oneToOne }``.
 * Letters match either case. Ignored while typing in a field or with a
 * dialog open. The latest handlers are always used. A handler that returns
 * ``false`` leaves the key to other listeners.
 */
export function useShortcuts(keys: Record<string, (e: KeyboardEvent) => unknown>, enabled = true) {
  const live = useRef(keys);
  live.current = keys;
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || isTyping(e) || dialogOpen()) return;
      const fn = live.current[e.key.length === 1 ? e.key.toLowerCase() : e.key] ?? live.current[e.key];
      if (!fn || fn(e) === false) return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [enabled]);
}

/** Shift + / Shift - (the "+" and "_" keys on most layouts, or = and -
 * with Shift): ``fn(1)`` or ``fn(-1)`` for ``useShortcuts``. ``fn`` returns
 * false when it has nothing to change. */
export function stepKeys(fn: (step: 1 | -1) => boolean): Record<string, (e: KeyboardEvent) => unknown> {
  const on = (step: 1 | -1) => (e: KeyboardEvent) => e.shiftKey && fn(step);
  return { "+": on(1), "=": on(1), _: on(-1), "-": on(-1) };
}

/** Shortcuts listed in the help overlay (? key). */
export const SHORTCUTS: { group: string; keys: [string, string][] }[] = [
  {
    group: "General",
    keys: [
      ["⌘/Ctrl O", "Open files"],
      ["⌘/Ctrl Z", "Undo the last deletion (maps, results, regions, lines, FFT boxes)"],
      ["↑ / ↓", "Previous / next dataset"],
      ["Shift + click", "Dataset: show just that one"],
      ["[ / ]", "Fold or unfold the datasets / right-hand pane"],
      ["?", "Show these shortcuts"],
      ["Shift F", "Focus the spectrum pane (the right-hand pane beside it, the navigator in a flap); over the datasets pane, the highlighted dataset's processing as a tree; again or Esc to leave"],
      ["Esc", "Close a dialog or menu; deselect"],
    ],
  },
  {
    group: "Image",
    keys: [
      ["F", "Fit the image under the mouse (else the main one) to its window, or double-click"],
      ["1", "One screen pixel per data pixel (same image)"],
      ["M", "Magnifier on or off beside the pointer (over the FFT: the FFT's own)"],
      ["G", "Several images: merged or a grid of panels"],
      ["⌘/Ctrl S", "Save the image as shown as PNG"],
      ["Space + drag", "Pan with any tool (or middle-drag)"],
      ["← / →", "Previous / next frame of a stack or in-situ series (⇧: ten; in-situ: ⌥← / ⌥→ nudge the cursor)"],
      ["Home / End", "First / last frame of a stack or in-situ series"],
    ],
  },
  {
    group: "Tools",
    keys: [
      ["C", "Spectrum cursor (arrow keys nudge it, Shift ×10)"],
      ["R / E / P", "Rectangle / ellipse / polygon region"],
      ["⌘ click", "Spectrum image: a second cursor there (drag to move it)"],
      ["Shift / ⌥ drag", "Rectangle, ellipse: square / centred on the press (both: both)"],
      ["L", "Line profile"],
      ["Shift + / Shift −", "Selected line: wider / narrower; selected region: bigger / smaller"],
      ["Delete", "Remove the selected region, line or energy window"],
      ["Shift + drag", "Line: snap the angle to 15°"],
    ],
  },
  {
    group: "Below the image",
    keys: [
      ["T", "Switch between the Spectrum and Line profile tabs"],
      ["Shift + drag", "Spectrum: new energy window (map); right-click a window to delete it"],
    ],
  },
];
