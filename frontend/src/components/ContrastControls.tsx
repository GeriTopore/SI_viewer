import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { Frame } from "../api/arrays";
import { COLORMAPS, OVERLAY_COLOURS, colormapGradient, isColour } from "../viewer/colormaps";
import { autoRange, fullRange, histogram, percentile, sortedSample } from "../viewer/contrast";
import type { DisplayParams } from "../viewer/renderer";
import { formatValue } from "../format";
import { useUi } from "../state/ui";

interface Props {
  frame: Frame | null;
  display: DisplayParams | undefined;
  onChange: (patch: Partial<DisplayParams>) => void;
}

const BINS = 96;
// A limit line is grabbed within this many pixels.
const GRAB_PX = 5;

type Drag = { kind: "lo" | "hi" } | { kind: "move"; at: number; lo: number; hi: number };

/** ±the larger of |lo| and |hi|. */
const symmetricOf = (lo: number, hi: number): [number, number] => {
  const m = Math.max(Math.abs(lo), Math.abs(hi));
  return [-m, m];
};
const SOLID = "__solid";

/** Colormap, histogram with draggable min/max lines (dragging between them
 * moves the window; right-click to type the limits or lock them symmetric),
 * auto contrast and gamma. */
export function ContrastControls({ frame, display, onChange }: Props) {
  const sample = useMemo(() => (frame ? sortedSample(frame.data) : null), [frame]);
  const theme = useUi((s) => s.theme); // histogram colours come from CSS
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragging = useRef<Drag | null>(null);
  const [hoverZone, setHoverZone] = useState<"line" | "inside" | null>(null);
  const [limitsAt, setLimitsAt] = useState<{ x: number; y: number } | null>(null);
  // Colour restored when switching back to "solid colour".
  const lastColour = useRef(display && isColour(display.colormap) ? display.colormap : OVERLAY_COLOURS[0]);
  if (display && isColour(display.colormap)) lastColour.current = display.colormap;

  // Histogram range: the bulk of the data, widened to include the current limits.
  const bulk = useMemo(
    (): [number, number] | null =>
      sample ? [percentile(sample, 0.05), percentile(sample, 99.95)] : null,
    [sample],
  );
  let range: [number, number] | null = null;
  if (bulk && display) {
    const a = Math.min(bulk[0], display.lo);
    range = [a, Math.max(bulk[1], display.hi, a + 1e-12)];
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !sample || !range || !display) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(dpr, dpr);
    const css = getComputedStyle(canvas);
    const counts = histogram(sample, range[0], range[1], BINS);
    const peak = Math.sqrt(Math.max(...counts)) || 1;
    const x = (v: number) => ((v - range[0]) / (range[1] - range[0])) * w;

    ctx.fillStyle = css.getPropertyValue("--hist-sel");
    ctx.fillRect(x(display.lo), 0, x(display.hi) - x(display.lo), h);
    ctx.fillStyle = css.getPropertyValue("--hist-bar");
    const bw = w / BINS;
    counts.forEach((c, i) => {
      const bh = (Math.sqrt(c) / peak) * (h - 2);
      ctx.fillRect(i * bw, h - bh, Math.max(1, bw - 0.5), bh);
    });
    ctx.fillStyle = css.getPropertyValue("--accent");
    for (const v of [display.lo, display.hi]) ctx.fillRect(Math.round(x(v)) - 1, 0, 2, h);
  }, [sample, range?.[0], range?.[1], display, theme]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!display) return <div className="contrast-controls" />;

  const valueAt = (clientX: number) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const f = (clientX - rect.left) / rect.width;
    return range ? range[0] + f * (range[1] - range[0]) : 0;
  };
  // What is under the pointer: a limit line (the nearer), between them, or neither.
  const zoneAt = (clientX: number): "lo" | "hi" | "inside" | null => {
    if (!range) return null;
    const rect = canvasRef.current!.getBoundingClientRect();
    const px = (v: number) => rect.left + ((v - range[0]) / (range[1] - range[0])) * rect.width;
    const dLo = Math.abs(clientX - px(display.lo));
    const dHi = Math.abs(clientX - px(display.hi));
    // A narrow window keeps its middle for moving it.
    const grab = Math.min(GRAB_PX, (px(display.hi) - px(display.lo)) / 4);
    if (Math.min(dLo, dHi) <= Math.max(grab, 1.5)) return dLo <= dHi ? "lo" : "hi";
    return clientX > px(display.lo) && clientX < px(display.hi) ? "inside" : null;
  };
  // Set the limits, kept symmetric when locked.
  const setLimits = (lo: number, hi: number) => {
    if (display.symmetric) [lo, hi] = symmetricOf(lo, hi);
    onChange({ lo, hi });
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    const zone = zoneAt(e.clientX);
    // Locked symmetric, the window can't move off 0: dragging inside does nothing.
    if (!zone || (zone === "inside" && display.symmetric)) return;
    dragging.current =
      zone === "inside" ? { kind: "move", at: valueAt(e.clientX), lo: display.lo, hi: display.hi } : { kind: zone };
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = dragging.current;
    if (!d) {
      const zone = zoneAt(e.clientX);
      setHoverZone(zone === "lo" || zone === "hi" ? "line" : zone === "inside" && !display.symmetric ? "inside" : null);
      return;
    }
    const v = valueAt(e.clientX);
    if (d.kind === "move") {
      const shift = v - d.at;
      onChange({ lo: d.lo + shift, hi: d.hi + shift });
    } else if (display.symmetric) {
      const m = Math.abs(v);
      onChange({ lo: -m, hi: m });
    } else if (d.kind === "lo") onChange({ lo: Math.min(v, display.hi) });
    else onChange({ hi: Math.max(v, display.lo) });
  };

  const gammaSlider = Math.log2(display.gamma) / 2; // gamma 0.25–4 ↔ -1…1

  return (
    <div className="contrast-controls">
      <div className="colormap-picker">
        <ColormapMenu
          current={display.colormap}
          solidColour={lastColour.current}
          onPick={(name) => onChange({ colormap: name === SOLID ? lastColour.current : name })}
        />
        {isColour(display.colormap) && (
          <input
            type="color"
            className="colour-input"
            value={display.colormap}
            onChange={(e) => {
              lastColour.current = e.target.value;
              onChange({ colormap: e.target.value });
            }}
            title="Colour at maximum intensity (black at minimum)"
          />
        )}
      </div>

      <div className="histogram">
        <canvas
          ref={canvasRef}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={() => (dragging.current = null)}
          onPointerLeave={() => !dragging.current && setHoverZone(null)}
          onContextMenu={(e) => {
            e.preventDefault();
            setLimitsAt({ x: e.clientX, y: e.clientY });
          }}
          style={{ cursor: dragging.current?.kind === "move" ? "grabbing" : hoverZone === "line" ? "ew-resize" : hoverZone === "inside" ? "grab" : undefined }}
          title="Drag a limit, or the range between; right-click to type limits"
        />
        <div className="histogram-labels">
          <span>{formatValue(display.lo)}</span>
          {display.symmetric && <span className="histogram-symmetric" title="Locked symmetric about 0">±</span>}
          <span>{formatValue(display.hi)}</span>
        </div>
      </div>
      {limitsAt && (
        <LimitsPopover
          at={limitsAt}
          display={display}
          onChange={onChange}
          onClose={() => setLimitsAt(null)}
        />
      )}

      <div className="contrast-buttons">
        <button
          disabled={!sample}
          onClick={() => sample && setLimits(...autoRange(sample))}
          title="Auto: clip 0.5 % at each end"
          aria-label="Auto contrast"
        >
          <RangeIcon inset />
        </button>
        <button
          disabled={!sample}
          onClick={() => sample && setLimits(...fullRange(sample))}
          title="Min–max: the full data range"
          aria-label="Full data range"
        >
          <RangeIcon />
        </button>
      </div>

      <GammaSlider gamma={display.gamma} value={gammaSlider} onChange={(gamma) => onChange({ gamma })} />
    </div>
  );
}

/** A histogram with the range marked: inside its tails (auto) or at its ends (min–max). */
function RangeIcon({ inset = false }: { inset?: boolean }) {
  const [a, b] = inset ? [5, 13] : [1.5, 16.5];
  return (
    <svg width="18" height="10" viewBox="0 0 18 10" aria-hidden fill="none" stroke="currentColor" strokeLinecap="round">
      <path d="M1 9C4 9 6 2 9 2S14 9 17 9" strokeWidth="1.2" />
      <path d={`M${a} 0.8V9.2M${b} 0.8V9.2`} strokeWidth="1.6" />
    </svg>
  );
}

/** Gamma on a short slider (0.25–4; double-click resets to 1). A click on
 * the handle without dragging opens a bubble to type any value. */
function GammaSlider({ gamma, value, onChange }: { gamma: number; value: number; onChange: (gamma: number) => void }) {
  const [bubble, setBubble] = useState(false);
  const atDown = useRef<number | null>(null);
  const wrap = useRef<HTMLLabelElement>(null);
  useEffect(() => {
    if (!bubble) return;
    const onDown = (e: MouseEvent) => !wrap.current?.contains(e.target as Node) && setBubble(false);
    window.addEventListener("mousedown", onDown);
    return () => window.removeEventListener("mousedown", onDown);
  }, [bubble]);
  return (
    <label ref={wrap} className="gamma" title="Gamma: click to type, double-click to reset">
      γ
      <span className="gamma-track">
        <input
          type="range"
          min={-1}
          max={1}
          step={0.01}
          value={value}
          aria-label="Gamma"
          onPointerDown={() => (atDown.current = gamma)}
          onPointerUp={() => {
            if (atDown.current === gamma) setBubble(true);
            atDown.current = null;
          }}
          onChange={(e) => onChange(2 ** (2 * Number(e.target.value)))}
          onDoubleClick={() => {
            setBubble(false);
            onChange(1);
          }}
        />
        {bubble && (
          <span className="gamma-bubble" style={{ left: `${((value + 1) / 2) * 100}%` }}>
            <GammaInput gamma={gamma} onChange={onChange} onDone={() => setBubble(false)} />
          </span>
        )}
      </span>
    </label>
  );
}

/** The colormap's icon; a click lists the colormaps (and "solid colour") in a
 * popover under it. */
function ColormapMenu({
  current, solidColour, onPick,
}: { current: string; solidColour: string; onPick: (name: string) => void }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const names = [...COLORMAPS, SOLID];
  const currentName = isColour(current) ? SOLID : current;
  const label = (name: string) => (name === SOLID ? "solid colour" : name);

  const show = () => {
    setActive(Math.max(0, names.indexOf(currentName)));
    setPos(null);
    setOpen(true);
  };
  const close = (refocus = false) => {
    setOpen(false);
    if (refocus) buttonRef.current?.focus();
  };
  const choose = (name: string) => {
    onPick(name);
    close(true);
  };

  // Under the icon, kept inside the viewport (flipped above when it won't fit).
  useLayoutEffect(() => {
    if (!open || !buttonRef.current || !listRef.current) return;
    const b = buttonRef.current.getBoundingClientRect();
    const w = listRef.current.offsetWidth;
    const h = listRef.current.offsetHeight;
    const m = 6;
    const left = Math.max(m, Math.min(b.left, window.innerWidth - w - m));
    let top = b.bottom + 4;
    if (top + h > window.innerHeight - m) top = Math.max(m, b.top - h - 4);
    setPos({ left, top });
  }, [open]);

  // Focus the list once it is placed (a hidden element can't take focus), so
  // the arrow keys move in the list rather than switching datasets.
  useEffect(() => {
    if (open && pos) listRef.current?.focus({ preventScroll: true });
  }, [open, pos]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (!listRef.current?.contains(t) && !buttonRef.current?.contains(t)) setOpen(false);
    };
    const onAway = () => setOpen(false);
    window.addEventListener("mousedown", onDown);
    window.addEventListener("resize", onAway);
    window.addEventListener("blur", onAway);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("resize", onAway);
      window.removeEventListener("blur", onAway);
    };
  }, [open]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    const n = names.length;
    if (e.key === "ArrowDown") setActive((a) => (a + 1) % n);
    else if (e.key === "ArrowUp") setActive((a) => (a + n - 1) % n);
    else if (e.key === "Home") setActive(0);
    else if (e.key === "End") setActive(n - 1);
    else if (e.key === "Enter" || e.key === " ") choose(names[active]);
    else if (e.key === "Escape") close(true);
    else if (e.key === "Tab") close();
    else return;
    e.preventDefault();
    e.stopPropagation();
  };

  return (
    <>
      <button
        ref={buttonRef}
        type="button"
        className={`colormap-button${open ? " active" : ""}`}
        onClick={() => (open ? close() : show())}
        onKeyDown={(e) => {
          if (!open && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
            e.preventDefault();
            show();
          }
        }}
        title={`Colormap: ${label(currentName)}`}
        aria-label={`Colormap: ${label(currentName)}`}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className="colormap-swatch" style={{ background: colormapGradient(current) }} />
      </button>
      {open &&
        createPortal(
          <div
            ref={listRef}
            className="colormap-list"
            role="listbox"
            tabIndex={-1}
            aria-label="Colormap"
            aria-activedescendant={`colormap-opt-${names[active]}`}
            style={{ left: pos?.left ?? 0, top: pos?.top ?? 0, visibility: pos ? "visible" : "hidden" }}
            onKeyDown={onKeyDown}
          >
            {names.map((name, i) => (
              <div
                key={name}
                id={`colormap-opt-${name}`}
                role="option"
                aria-selected={name === currentName}
                className={`colormap-option${name === currentName ? " current" : ""}${i === active ? " active" : ""}`}
                onMouseMove={() => i !== active && setActive(i)}
                onClick={() => choose(name)}
              >
                <span
                  className="colormap-swatch"
                  style={{ background: colormapGradient(name === SOLID ? solidColour : name) }}
                />
                {label(name)}
              </div>
            ))}
          </div>,
          document.body,
        )}
    </>
  );
}

const GAMMA_MIN = 0.01;
const GAMMA_MAX = 100;

/** Gamma typed in (the slider covers 0.25–4): applied as soon as it is a
 * valid number; Enter, Esc or leaving the field closes it (``onDone``). */
function GammaInput({ gamma, onChange, onDone }: { gamma: number; onChange: (gamma: number) => void; onDone: () => void }) {
  const [draft, setDraft] = useState<string | null>(null);
  const edit = (text: string) => {
    setDraft(text);
    const v = Number(text);
    if (text.trim() !== "" && v >= GAMMA_MIN && v <= GAMMA_MAX) onChange(v);
  };
  return (
    <input
      type="number"
      className="gamma-input"
      autoFocus
      min={GAMMA_MIN}
      max={GAMMA_MAX}
      step={0.05}
      value={draft ?? Number(gamma.toFixed(2))}
      onChange={(e) => edit(e.target.value)}
      onFocus={(e) => e.currentTarget.select()}
      onBlur={onDone}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === "Escape") onDone();
        e.stopPropagation();
      }}
      aria-label="Gamma value"
    />
  );
}

/** Typing the limits (right-click on the histogram): min and max, or ±one
 * value when locked symmetric about 0. Applied as typed; Enter, Esc or a
 * click elsewhere closes it. */
function LimitsPopover({ at, display, onChange, onClose }: {
  at: { x: number; y: number };
  display: DisplayParams;
  onChange: (patch: Partial<DisplayParams>) => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const m = 6;
    setPos({
      left: Math.max(m, Math.min(at.x, window.innerWidth - el.offsetWidth - m)),
      top: Math.max(m, Math.min(at.y + 4, window.innerHeight - el.offsetHeight - m)),
    });
  }, [at.x, at.y]);
  // Focus the first field once placed (a hidden element can't take focus).
  useEffect(() => {
    if (!pos) return;
    const input = ref.current?.querySelector<HTMLInputElement>("input[type=number]");
    input?.focus();
    input?.select();
  }, [pos === null]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const onDown = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && onClose();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  const sym = !!display.symmetric;
  const field = (key: "lo" | "hi" | "pm", label: string, value: number, apply: (v: number) => void) => (
    <label>
      <span>{label}</span>
      <input
        type="number"
        className="cell-input"
        step="any"
        value={drafts[key] ?? formatValue(value)}
        onChange={(e) => {
          setDrafts({ ...drafts, [key]: e.target.value });
          const v = Number(e.target.value);
          if (e.target.value.trim() !== "" && Number.isFinite(v)) apply(v);
        }}
        onFocus={(e) => e.currentTarget.select()}
        onKeyDown={(e) => {
          if (e.key === "Enter") onClose();
          e.stopPropagation();
        }}
      />
    </label>
  );

  return createPortal(
    <div
      ref={ref}
      className="contrast-limits"
      style={{ left: pos?.left ?? at.x, top: pos?.top ?? at.y, visibility: pos ? "visible" : "hidden" }}
      role="dialog"
      aria-label="Display limits"
    >
      {sym ? (
        field("pm", "±", display.hi, (v) => onChange({ lo: -Math.abs(v), hi: Math.abs(v) }))
      ) : (
        <>
          {field("lo", "Min", display.lo, (v) => onChange({ lo: v }))}
          {field("hi", "Max", display.hi, (v) => onChange({ hi: v }))}
        </>
      )}
      <label className="q-check" title="Keep the limits at ±the same value (e.g. strain)">
        <input
          type="checkbox"
          checked={sym}
          onChange={(e) => {
            setDrafts({});
            if (e.target.checked) {
              const [lo, hi] = symmetricOf(display.lo, display.hi);
              onChange({ symmetric: true, lo, hi });
            } else onChange({ symmetric: false });
          }}
        />
        Symmetric about 0
      </label>
    </div>,
    document.body,
  );
}
