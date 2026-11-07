import { useRef } from "react";

interface Props {
  /** "col": a vertical bar dragged sideways (widths); "row": a horizontal
   * bar dragged up and down (heights). */
  direction: "col" | "row";
  /** Called while dragging with the pointer's clientX (col) or clientY (row). */
  onDrag: (pos: number) => void;
  /** Double-click: back to the default size. */
  onReset?: () => void;
  className?: string;
  label: string;
}

/** Drag handle between two panes. */
export function Splitter({ direction, onDrag, onReset, className, label }: Props) {
  const dragging = useRef(false);
  const pos = (e: React.PointerEvent) => (direction === "col" ? e.clientX : e.clientY);
  return (
    <div
      className={`splitter ${direction}${className ? ` ${className}` : ""}`}
      role="separator"
      aria-orientation={direction === "col" ? "vertical" : "horizontal"}
      aria-label={label}
      title={`Drag to resize${onReset ? "; double-click to reset" : ""}`}
      onPointerDown={(e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        e.currentTarget.setPointerCapture(e.pointerId);
        dragging.current = true;
        document.body.classList.add(`resizing-${direction}`);
      }}
      onPointerMove={(e) => dragging.current && onDrag(pos(e))}
      onPointerUp={(e) => {
        dragging.current = false;
        e.currentTarget.releasePointerCapture(e.pointerId);
        document.body.classList.remove(`resizing-${direction}`);
      }}
      onDoubleClick={onReset}
    />
  );
}
