import { useEffect, useRef, type ReactNode } from "react";

/** ``children`` (a name, a swatch) that open a colour picker when clicked. */
export function ColourPick({ color, label, onChange, children }: {
  color: string;
  label: string;
  onChange: (color: string) => void;
  children: ReactNode;
}) {
  const input = useRef<HTMLInputElement>(null);
  // Closing the picker gives the keys back to the shortcuts (e.g. L for the line tool).
  useEffect(() => {
    const el = input.current!;
    const blur = () => el.blur();
    el.addEventListener("change", blur);
    return () => el.removeEventListener("change", blur);
  }, []);
  return (
    <label className="colour-pick" title={`${label} colour — click to change`}>
      {children}
      <input
        ref={input}
        type="color"
        value={color}
        onChange={(e) => onChange(e.target.value)}
        aria-label={`${label} colour`}
      />
    </label>
  );
}
