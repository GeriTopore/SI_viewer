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
 * dialog open. The latest handlers are always used.
 */
export function useShortcuts(keys: Record<string, (e: KeyboardEvent) => void>, enabled = true) {
  const live = useRef(keys);
  live.current = keys;
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || isTyping(e) || dialogOpen()) return;
      const fn = live.current[e.key.length === 1 ? e.key.toLowerCase() : e.key] ?? live.current[e.key];
      if (!fn) return;
      e.preventDefault();
      fn(e);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [enabled]);
}

/** Shortcuts listed in the help overlay (? key). */
export const SHORTCUTS: { group: string; keys: [string, string][] }[] = [
  {
    group: "General",
    keys: [
      ["⌘/Ctrl O", "Open files"],
      ["↑ / ↓", "Previous / next dataset"],
      ["[ / ]", "Fold or unfold the datasets / right-hand pane"],
      ["?", "Show these shortcuts"],
      ["Esc", "Close a dialog or menu; deselect"],
    ],
  },
  {
    group: "Image",
    keys: [
      ["F", "Fit the image to the window (or double-click)"],
      ["1", "One screen pixel per data pixel"],
      ["⌘/Ctrl S", "Save the image as shown as PNG"],
      ["Space + drag", "Pan with any tool (or middle-drag)"],
      ["← / →", "Previous / next frame of a stack"],
      ["Home / End", "First / last frame of a stack"],
    ],
  },
  {
    group: "Tools",
    keys: [
      ["C", "Spectrum cursor (arrow keys nudge it, Shift ×10)"],
      ["R / E / P", "Rectangle / ellipse / polygon region"],
      ["L", "Line profile"],
      ["H", "Pan"],
      ["Delete", "Remove the selected region or line"],
      ["Shift + drag", "Line: snap the angle to 15°"],
    ],
  },
  {
    group: "Below the image",
    keys: [["T", "Switch between the Spectrum and Line profile tabs"]],
  },
];
