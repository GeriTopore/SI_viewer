import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { useDisplay } from "../state/display";
import { swatchBackground } from "../viewer/colormaps";
import { edgeLabel, lineLabel, useElementTable, type ElementInfo } from "../views/si/xray";

// Standard 18-column layout; "" is a gap, "*"/"**" mark the f-block rows.
const ROWS = [
  "H . . . . . . . . . . . . . . . . He",
  "Li Be . . . . . . . . . . B C N O F Ne",
  "Na Mg . . . . . . . . . . Al Si P S Cl Ar",
  "K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr",
  "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe",
  "Cs Ba La Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn",
  "Fr Ra Ac Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og",
  "",
  ". . . Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu .",
  ". . . Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr .",
].map((r) => (r ? r.split(" ").map((c) => (c === "." ? "" : c)) : []));

interface Props {
  selected: string[];
  onToggle: (symbol: string) => void;
  onClear: () => void;
  onHover: (symbol: string | null) => void;
  onClose: () => void;
  style?: CSSProperties;
  /** Shown under the table (e.g. map options for a spectrum image). */
  footer?: ReactNode;
  /** EELS: the spectrum's energy range (eV). Elements are chosen by their
   * edges inside it, not by their X-ray lines. */
  edgeRange?: [number, number];
}

/** Popover periodic table for choosing elements; hover previews their lines. */
export function PeriodicTable({ selected, onToggle, onClear, onHover, onClose, style, footer, edgeRange }: Props) {
  const edgesOf = (el: ElementInfo) =>
    edgeRange ? el.edges.filter((e) => e.onset >= edgeRange[0] && e.onset <= edgeRange[1]) : [];
  const table = useElementTable();
  const colours = useDisplay((s) => s.elementColours);
  const ref = useRef<HTMLDivElement>(null);
  const [info, setInfo] = useState<ElementInfo | null>(null);
  const bySymbol = new Map(table.map((e) => [e.symbol, e]));
  const chosen = new Set(selected);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node) && !(e.target as HTMLElement).closest(".elements-button")) onClose();
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("mousedown", onClick);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("mousedown", onClick);
      onHover(null);
    };
  }, [onClose, onHover]);

  return (
    <div className="periodic-table" ref={ref} role="dialog" aria-label="Elements" style={style}>
      <div className="pt-grid" onMouseLeave={() => (onHover(null), setInfo(null))}>
        {ROWS.map((row, r) =>
          row.length === 0 ? (
            <div key={r} className="pt-gap" />
          ) : (
            row.map((sym, c) => {
              const el = sym ? bySymbol.get(sym) : undefined;
              if (!sym) return <div key={`${r}-${c}`} />;
              const on = chosen.has(sym);
              return (
                <button
                  key={sym}
                  className={`pt-cell${on ? " on" : ""}`}
                  disabled={!el || !(edgeRange ? edgesOf(el).length || on : el.lines.length)}
                  style={on ? { background: swatchBackground(colours[sym] ?? "#4c8dff") } : undefined}
                  onClick={() => onToggle(sym)}
                  onMouseEnter={() => {
                    if (!el) return;
                    onHover(sym);
                    setInfo(el);
                  }}
                  title={el ? `${el.Z} ${el.name}` : sym}
                >
                  {sym}
                </button>
              );
            })
          ),
        )}
      </div>
      <div className="pt-footer">
        <span className="pt-info">
          {info ? (
            <>
              <b>
                {info.Z} {info.symbol}
              </b>{" "}
              <span className="muted">{info.name}</span>{" "}
              {edgeRange
                ? `${edgesOf(info)
                    .map((e) => `${edgeLabel(e.edge)} ${Math.round(e.onset)}`)
                    .join(" · ")} eV`
                : `${info.lines
                    .filter((l) => l.weight >= 0.5)
                    .map((l) => `${lineLabel(l.line)} ${l.energy.toFixed(3)}`)
                    .join(" · ")} keV`}
            </>
          ) : (
            <span className="muted">
              {selected.length ? `${selected.join(", ")}` : edgeRange ? "Click elements to mark their edges" : "Click elements to mark their X-ray lines"}
            </span>
          )}
        </span>
        <button className="link-button" onClick={onClear} disabled={!selected.length}>
          Clear
        </button>
      </div>
      {footer}
    </div>
  );
}


