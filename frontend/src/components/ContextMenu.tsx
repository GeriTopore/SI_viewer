import { useEffect, useLayoutEffect, useRef, useState } from "react";

export interface MenuItem {
  label: string;
  onSelect?: () => void;
  danger?: boolean;
  /** Greyed out and not clickable (a heading or a placeholder). */
  disabled?: boolean;
  /** An entry under a heading item. */
  indent?: boolean;
  title?: string;
  /** A colour dot before the label (e.g. a cursor's or region's colour). */
  color?: string;
  /** The entry chosen now (a list of choices). */
  checked?: boolean;
}

interface Props {
  x: number;
  y: number;
  items: MenuItem[];
  onClose: () => void;
}

/** Right-click menu at (x, y), kept inside the window. */
export function ContextMenu({ x, y, items, onClose }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ left: x, top: y });

  useLayoutEffect(() => {
    const r = ref.current!.getBoundingClientRect();
    setPos({
      left: Math.min(x, window.innerWidth - r.width - 4),
      top: Math.min(y, window.innerHeight - r.height - 4),
    });
  }, [x, y, items.length]);

  useEffect(() => {
    const close = (e: Event) => {
      if (e.type === "mousedown" && ref.current?.contains(e.target as Node)) return;
      onClose();
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("mousedown", close);
    window.addEventListener("blur", close);
    window.addEventListener("resize", close);
    document.addEventListener("scroll", close, true);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", close);
      window.removeEventListener("blur", close);
      window.removeEventListener("resize", close);
      document.removeEventListener("scroll", close, true);
      window.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  return (
    <div className="context-menu" ref={ref} role="menu" style={pos} onContextMenu={(e) => e.preventDefault()}>
      {items.map((item, i) => (
        <button
          key={i}
          role="menuitem"
          disabled={item.disabled}
          title={item.title}
          aria-checked={item.checked}
          className={[item.danger && "danger", item.indent && "indent", item.checked && "checked"].filter(Boolean).join(" ") || undefined}
          onClick={() => {
            onClose();
            item.onSelect?.();
          }}
        >
          {item.color && <span className="menu-swatch" style={{ background: item.color }} />}
          {item.label}
        </button>
      ))}
    </div>
  );
}
