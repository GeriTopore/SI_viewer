import { useUi } from "../state/ui";

const Chevron = ({ dir }: { dir: "left" | "right" }) => (
  <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
    <path
      d={dir === "left" ? "M8.5 3 4.5 7l4 4" : "M5.5 3l4 4-4 4"}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

/** Button in a side pane's header that folds the pane towards its edge. */
export function CollapseButton({ side, label }: { side: "left" | "right"; label: string }) {
  const setCollapsed = useUi((s) => s.setCollapsed);
  return (
    <button
      className="icon-button pane-collapse"
      onClick={() => setCollapsed(side, true)}
      title={`Hide the ${label} pane`}
      aria-label={`Hide the ${label} pane`}
    >
      <Chevron dir={side} />
    </button>
  );
}

/** What remains of a collapsed side pane: a thin strip that reopens it. */
export function CollapsedStrip({ side, label }: { side: "left" | "right"; label: string }) {
  const setCollapsed = useUi((s) => s.setCollapsed);
  return (
    <button
      className="pane-strip"
      onClick={() => setCollapsed(side, false)}
      title={`Show the ${label} pane`}
      aria-label={`Show the ${label} pane`}
    >
      <Chevron dir={side === "left" ? "right" : "left"} />
      <span className="pane-strip-label">{label}</span>
    </button>
  );
}
