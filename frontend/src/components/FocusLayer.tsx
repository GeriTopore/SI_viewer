import { useEffect } from "react";
import { isTyping } from "../keyboard";
import { useUi } from "../state/ui";
import { FLAP_MIN_W } from "./LayeredViewer";
import { ProcessTree } from "./ProcessTree";
import { Splitter } from "./Splitter";

/** Narrowest the right-hand pane and the pane in focus get (px). */
const RIGHT_MIN = 240;
const PANE_MIN = 320;
/** Gap and inset as in the styles (--focus-gap, --focus-inset-x: 2.5vw). */
const GAP = 10;
const insetX = () => window.innerWidth * 0.025;

/** Menus and dialogs that Esc closes before it leaves focus mode. */
const OWNS_ESC = ".modal-backdrop, .context-menu, .technique-popover, .examples-popover, .periodic-table, .cmap-menu";

/** Focus mode (M20): Shift F lifts the bottom pane's active half over the
 * blurred, dimmed interface, with the right-hand pane beside it and, on the
 * left, the navigator in a flap. The panes are the live ones, placed by CSS
 * (classes on <html>), not copies. Shift F again, Esc or a click on the
 * backdrop leaves. */
export function FocusLayer() {
  const focus = useUi((s) => s.focus);
  const nav = useUi((s) => s.focusNav);
  const rightCollapsed = useUi((s) => s.rightCollapsed);
  const rightW = useUi((s) => s.focusRightW);
  const { setFocus, setFocusNav, setFocusWidth } = useUi.getState();

  // Before other shortcuts (F alone fits the image): capture phase. Over the
  // datasets pane, Shift F shows the processing tree; elsewhere it lifts the
  // bottom pane.
  useEffect(() => {
    let over: EventTarget | null = null;
    const onMove = (e: PointerEvent) => {
      over = e.target;
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || isTyping(e)) return;
      const on = useUi.getState().focus;
      if (e.shiftKey && (e.code === "KeyF" || e.key === "F")) {
        if (!on && document.querySelector(".modal-backdrop")) return;
        const datasets = over instanceof Element && !!over.closest(".sidebar");
        if (!on && !datasets && !document.querySelector(".viewer-half")) return; // no pane to focus
        setFocus(on ? null : datasets ? "datasets" : "bottom");
      } else if (e.key === "Escape" && on && !document.querySelector(OWNS_ESC)) {
        setFocus(null);
      } else return;
      e.preventDefault();
      e.stopPropagation();
    };
    window.addEventListener("keydown", onKey, true);
    window.addEventListener("pointermove", onMove, true);
    return () => {
      window.removeEventListener("keydown", onKey, true);
      window.removeEventListener("pointermove", onMove, true);
    };
  }, [setFocus]);

  useEffect(() => {
    const root = document.documentElement.classList;
    root.toggle("focus-mode", !!focus);
    root.toggle("focus-nav", focus === "bottom" && nav);
    root.toggle("focus-right", focus === "bottom" && !rightCollapsed);
    return () => root.remove("focus-mode", "focus-nav", "focus-right");
  }, [focus, nav, rightCollapsed]);

  // The right-hand pane's width, as dragged (the default otherwise).
  useEffect(() => {
    const style = document.documentElement.style;
    if (focus === "bottom" && rightW) style.setProperty("--focus-right-w", `${rightW}px`);
    else style.removeProperty("--focus-right-w");
  }, [focus, rightW]);

  /** Left edge of the pane in focus (px), from the navigator's flap. */
  const paneLeft = () => {
    const nav = document.querySelector(".viewer-top.focus-nav-lifted")?.getBoundingClientRect();
    return nav ? nav.right + 2 * GAP + 26 : insetX() + 26;
  };
  const navLeft = () => document.querySelector(".viewer-top.focus-nav-lifted")?.getBoundingClientRect().left ?? insetX();
  const rightEdge = () => window.innerWidth - insetX();

  if (!focus) return null;
  if (focus === "datasets") {
    return (
      <>
        <div className="focus-backdrop" onClick={() => setFocus(null)} aria-hidden />
        <ProcessTree />
      </>
    );
  }
  return (
    <>
      <div className="focus-backdrop" onClick={() => setFocus(null)} aria-hidden />
      {!rightCollapsed && (
        <Splitter
          direction="col"
          className="focus-resizer right"
          label="Resize the pane and the right-hand pane"
          onDrag={(x) => {
            const max = rightEdge() - paneLeft() - PANE_MIN - GAP;
            setFocusWidth("right", Math.max(RIGHT_MIN, Math.min(max, rightEdge() - x - GAP / 2)));
          }}
          onReset={() => setFocusWidth("right", null)}
        />
      )}
      {nav && (
        <Splitter
          direction="col"
          className="focus-resizer left"
          label="Resize the navigator and the pane"
          onDrag={(x) => {
            const right = document.querySelector(".inspector-panel")?.getBoundingClientRect();
            const limit = (rightCollapsed || !right ? rightEdge() : right.left - GAP) - PANE_MIN - 26 - 2 * GAP - navLeft();
            setFocusWidth("nav", Math.max(FLAP_MIN_W, Math.min(limit, x - navLeft() - GAP / 2)));
          }}
          onReset={() => setFocusWidth("nav", null)}
        />
      )}
      <button
        className="focus-flap-handle"
        onClick={() => setFocusNav(!nav)}
        title={nav ? "Hide the navigator" : "Show the navigator"}
        aria-label={nav ? "Hide the navigator" : "Show the navigator"}
        aria-expanded={nav}
      >
        <span className="focus-flap-arrow" aria-hidden>{nav ? "▸" : "◂"}</span>
        <span className="focus-flap-label">Navigator</span>
      </button>
    </>
  );
}
