import { useCallback, useRef, useState } from "react";
import type { DatasetSummary } from "../api/client";
import { isLineScan } from "../format";
import { setElements } from "../views/si/xray";
import { MapOptionsControls } from "./MapOptionsControls";
import { PeriodicTable } from "./PeriodicTable";

interface Props {
  ds: DatasetSummary;
  /** Element whose lines to preview (hovered in the table). */
  onPreview: (symbol: string | null) => void;
}

const PANEL_WIDTH = 560;

/** "Elements" button opening a periodic table that edits ``ds``'s elements. */
export function ElementsPicker({ ds, onPreview }: Props) {
  const [open, setOpen] = useState(false);
  const buttonRef = useRef<HTMLButtonElement>(null);
  // Fixed position under the button, so panels with overflow:hidden don't clip it.
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const elements = ds.edges?.elements ?? ds.eds?.elements ?? [];
  const close = useCallback(() => setOpen(false), []);
  if (!ds.eds && !ds.edges) return null;
  const axis = ds.axes.find((a) => !a.navigate);
  // EELS: only elements with an edge inside the energy range can be chosen.
  const edgeRange: [number, number] | undefined =
    ds.edges && axis ? [axis.offset, axis.offset + axis.scale * (axis.size - 1)] : undefined;

  const toggle = (symbol: string) =>
    setElements(ds, elements.includes(symbol) ? elements.filter((e) => e !== symbol) : [...elements, symbol]);

  return (
    <div className="elements-picker">
      <button
        className={`secondary small-button elements-button${open ? " active" : ""}`}
        ref={buttonRef}
        onClick={() => {
          const r = buttonRef.current!.getBoundingClientRect();
          setPos({ top: r.bottom + 6, left: Math.max(8, Math.min(r.left, window.innerWidth - PANEL_WIDTH - 8)) });
          setOpen(!open);
        }}
        aria-expanded={open}
        aria-label="Elements"
        title={ds.edges ? "Choose elements to mark and fit their edges" : "Choose elements to mark their X-ray lines"}
      >
        <PeriodicTableIcon />
        {elements.length > 0 && <span className="elements-count">{elements.length}</span>} ▾
      </button>
      {open && (
        <PeriodicTable
          style={{ position: "fixed", top: pos.top, left: pos.left, width: PANEL_WIDTH }}
          selected={elements}
          onToggle={toggle}
          onClear={() => setElements(ds, [])}
          onHover={onPreview}
          onClose={close}
          edgeRange={edgeRange}
          footer={ds.eds && ds.kind === "spectrum_image" && !isLineScan(ds) ? <MapOptionsControls ds={ds} /> : undefined}
        />
      )}
    </div>
  );
}

/** The periodic table's silhouette (H and He on top, the s and p blocks,
 * the full rows; the transition metals' gap shortened) with the f block
 * below, outlined in the accent colour. */
function PeriodicTableIcon() {
  const style = { stroke: "var(--accent)", fill: "var(--accent)", fillOpacity: 0.25, strokeWidth: 1.2 };
  return (
    <svg className="elements-icon" width="19" height="11" viewBox="-0.5 -0.5 15 8.6" aria-hidden>
      <path
        d="M0 0H1.4V1.5H3V3.2H8V1.5H12.6V0H14V5.2H0Z"
        vectorEffect="non-scaling-stroke"
        strokeLinejoin="round"
        {...style}
      />
      <rect x="2.5" y="6.2" width="10" height="1.4" vectorEffect="non-scaling-stroke" {...style} />
    </svg>
  );
}
