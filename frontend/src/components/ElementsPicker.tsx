import { useCallback, useRef, useState } from "react";
import type { DatasetSummary } from "../api/client";
import { setElements } from "../views/si/xray";
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
  const elements = ds.eds?.elements ?? [];
  const close = useCallback(() => setOpen(false), []);
  if (!ds.eds) return null;

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
        title="Choose elements to mark their X-ray lines"
      >
        Elements{elements.length > 0 && ` (${elements.length})`} ▾
      </button>
      {open && (
        <PeriodicTable
          style={{ position: "fixed", top: pos.top, left: pos.left, width: PANEL_WIDTH }}
          selected={elements}
          onToggle={toggle}
          onClear={() => setElements(ds, [])}
          onHover={onPreview}
          onClose={close}
        />
      )}
    </div>
  );
}
