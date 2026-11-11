import {
  useEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type MutableRefObject,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { createPortal } from "react-dom";
import type { DatasetSummary, InsituView } from "../api/client";
import { useShortcuts } from "../keyboard";
import { reducing, useInsitu } from "../state/insitu";
import { frameTime } from "../format";

type Mode = InsituView["mode"];
type Range = [number, number];

// A slider move (or a key) is read this long after the last change, so
// stepping quickly doesn't ask for every frame in between.
const DEBOUNCE_MS = 150;
// Frames on each side of the current one in a new range.
const DEFAULT_HALF = 2;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Frame bar of an in-situ series, under its navigator: a slider over the
 * frames (a frame is read when the slider is released, or shortly after a key
 * or a click), or a frame range reduced to its mean or sum. Frames count from 1. */
export function InsituFrameBar({ ds }: { ds: DatasetSummary }) {
  const info = ds.insitu!;
  const view = info.view!;
  const n = info.frames;
  const times = info.frame_times_s ?? [];
  const { showFrame, reduce, cancel } = useInsitu.getState();
  const reading = useInsitu((s) => s.reading[ds.id]);
  const error = useInsitu((s) => s.errors[ds.id]);
  const job = useInsitu((s) => s.jobs[ds.id]);
  const running = reducing(job);

  const [mode, setMode] = useState<Mode>(view.mode);
  const [index, setIndex] = useState(view.start);
  const [range, setRange] = useState<Range>(view.mode === "frame" ? around(view.start, n) : [view.start, view.end]);
  // The From/To bubble over the Mean or Sum button: the button's top centre on screen.
  const [asking, setAsking] = useState<{ x: number; y: number } | null>(null);

  const dragging = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout>>();
  const latest = useRef(index);
  latest.current = index;
  useEffect(() => () => clearTimeout(timer.current), []);

  // Follow the view when it changes under us (a finished range job), and go
  // back to it when a range job ends without changing it (cancelled, failed).
  useEffect(() => {
    if (running || reading !== undefined || dragging.current) return;
    setMode(view.mode);
    if (view.mode === "frame") setIndex(view.start);
    else setRange([view.start, view.end]);
  }, [view.mode, view.start, view.end, running]); // eslint-disable-line react-hooks/exhaustive-deps

  const goTo = (k: number, now = false) => {
    const next = clamp(k, 0, n - 1);
    setIndex(next);
    latest.current = next;
    clearTimeout(timer.current);
    if (dragging.current) return; // read on release
    if (now) showFrame(ds.id, next);
    else timer.current = setTimeout(() => showFrame(ds.id, latest.current), DEBOUNCE_MS);
  };

  /** Reduce ``r`` with the current mode, unless the view already holds it. */
  const applyRange = (r: Range, m: Mode = mode) => {
    setRange(r);
    if (m === "frame" || running) return;
    if (view.mode === m && view.start === r[0] && view.end === r[1]) return;
    void reduce(ds.id, m, r[0], r[1]);
  };
  const shiftRange = (d: number) => {
    const [s, e] = range;
    const k = clamp(d, -s, n - 1 - e);
    if (k) applyRange([s + k, e + k]);
  };

  // ←/→ step one frame (⇧ ten), Home/End go to the ends; a range moves as a
  // whole. Not while typing.
  useShortcuts(
    {
      ArrowLeft: (e) => (mode === "frame" ? goTo(latest.current - (e.shiftKey ? 10 : 1)) : !running && shiftRange(e.shiftKey ? -10 : -1)),
      ArrowRight: (e) => (mode === "frame" ? goTo(latest.current + (e.shiftKey ? 10 : 1)) : !running && shiftRange(e.shiftKey ? 10 : 1)),
      Home: () => (mode === "frame" ? goTo(0) : !running && shiftRange(-n)),
      End: () => (mode === "frame" ? goTo(n - 1) : !running && shiftRange(n)),
    },
    true,
  );

  const choose = (m: Mode, button: HTMLElement) => {
    if (running) return;
    if (m === "frame") {
      setAsking(null);
      if (m === mode) return;
      setMode(m);
      const k = Math.round((range[0] + range[1]) / 2);
      if (!(view.mode === "frame" && view.start === k)) goTo(k, true);
      return;
    }
    // Into a range: around the frame shown; the bubble asks for its frames.
    if (mode === "frame") setRange(around(latest.current, n));
    setMode(m);
    const r = button.getBoundingClientRect();
    setAsking({ x: r.left + r.width / 2, y: r.top });
  };

  const target = clamp(index, 0, n - 1);
  const timeText = times[target] !== undefined ? ` · ${frameTime(times[target])}` : "";
  const progress = job && running ? Math.round(job.progress * 100) : 0;
  // Room for the longest frame text, so the buttons after it stay put.
  const lastTime = times.length ? ` · ${frameTime(Math.max(...times))}` : "";
  const indexWidth = { minWidth: `${Math.max(`${n}–${n} / ${n}`.length, `${n} / ${n}${lastTime}`.length)}ch` };

  // Between the slider and the frame text, in a slot of its own so nothing
  // moves: a range job's progress (click to cancel), a frame being read, or
  // why the last one failed.
  const status = (
    <span className="insitu-slot">
      {running && job ? (
        <button
          className="progress-ring"
          onClick={() => cancel(ds.id)}
          title={`${job.message || "Starting…"} ${progress}% · click to cancel`}
          aria-label={`${progress}% done; cancel`}
        >
          <svg viewBox="0 0 16 16" aria-hidden>
            <circle cx="8" cy="8" r="6" className="ring-track" />
            <circle cx="8" cy="8" r="6" className="ring-fill" pathLength={100} strokeDasharray={`${progress} 100`} />
          </svg>
        </button>
      ) : reading !== undefined ? (
        <span className="spinner" title={`Reading frame ${reading + 1}…`} aria-label={`Reading frame ${reading + 1}`} />
      ) : error ? (
        <span className="slot-error" role="alert" title={error} aria-label={error}>
          !
        </span>
      ) : null}
    </span>
  );

  return (
    <div className="insitu-bar">
      {mode === "frame" ? (
        <>
          <input
            type="range"
            className="stack-slider"
            min={0}
            max={n - 1}
            step={1}
            value={target}
            onChange={(e) => goTo(Number(e.target.value))}
            onPointerDown={() => (dragging.current = true)}
            onPointerUp={() => {
              dragging.current = false;
              goTo(latest.current, true);
            }}
            onPointerCancel={() => (dragging.current = false)}
            aria-label="Frame"
            title="Frame (← / → step, Home / End)"
          />
          {status}
          <span className="stack-index" style={indexWidth} title="Frame / frames · time of the frame">
            {target + 1} / {n}
            {timeText}
          </span>
        </>
      ) : (
        <>
          <RangeSlider n={n} range={range} disabled={running} onDrag={setRange} onDone={applyRange} dragging={dragging} />
          {status}
          <span className="stack-index" style={indexWidth} title="Frames in the range / frames">
            {range[0] + 1}–{range[1] + 1} / {n}
          </span>
        </>
      )}
      <div className="insitu-modes">
        <div className="segmented" role="group" aria-label="Frame or range">
          {(["frame", "mean", "sum"] as Mode[]).map((m) => (
            <button
              key={m}
              className={mode === m ? "active" : undefined}
              aria-pressed={mode === m}
              disabled={running}
              onClick={(e) => choose(m, e.currentTarget)}
              title={m === "frame" ? "One frame" : m === "mean" ? "Mean of a range of frames" : "Sum of a range of frames"}
            >
              {m === "frame" ? "Frame" : m === "mean" ? "Mean" : "Sum"}
            </button>
          ))}
        </div>
        {asking !== null && mode !== "frame" && (
          <RangeBubble
            at={asking}
            n={n}
            range={range}
            onApply={(r) => {
              setAsking(null);
              applyRange(r);
            }}
            onClose={() => setAsking(null)}
          />
        )}
      </div>
    </div>
  );
}

/** ``DEFAULT_HALF`` frames each side of ``k``, inside 0…n-1. */
function around(k: number, n: number): Range {
  const half = Math.min(DEFAULT_HALF, Math.floor((n - 1) / 2));
  const s = clamp(k - half, 0, Math.max(0, n - 1 - 2 * half));
  return [s, Math.min(n - 1, s + 2 * half)];
}

type Grip = "centre" | "start" | "end";
// Least distance (px) of a limit from the centre handle's middle.
const LIMIT_GAP = 12;

/** A frame range on a track of ``n`` frames: the band of frames, a handle at
 * its centre that moves it, and a limit at each end that widens or narrows it
 * about its centre. While held, the two limits' frames are shown above them.
 * ``onDrag`` follows the pointer; ``onDone`` gets the range on release. A
 * press on the track away from the handles moves the range there. */
function RangeSlider({ n, range, disabled, onDrag, onDone, dragging }: {
  n: number;
  range: Range;
  disabled: boolean;
  onDrag: (r: Range) => void;
  onDone: (r: Range) => void;
  dragging: MutableRefObject<boolean>;
}) {
  const track = useRef<HTMLDivElement>(null);
  const drag = useRef<{ grip: Grip; at: number; start: Range; now: Range } | null>(null);
  const [held, setHeld] = useState(false);
  const [s, e] = range;

  /** The pointer's position in frames along the track (0 … n). */
  const frameAt = (clientX: number) => {
    const r = track.current!.getBoundingClientRect();
    return (clamp(clientX - r.left, 0, r.width) / Math.max(1, r.width)) * n;
  };
  const rangeFor = (grip: Grip, d: number, [s0, e0]: Range): Range => {
    if (grip === "centre") {
      const k = clamp(d, -s0, n - 1 - e0);
      return [s0 + k, e0 + k];
    }
    // Both limits move by the same amount, out or in.
    const out = clamp(grip === "start" ? -d : d, -Math.floor((e0 - s0) / 2), Math.min(s0, n - 1 - e0));
    return [s0 - out, e0 + out];
  };

  const onDown = (ev: ReactPointerEvent<HTMLDivElement>) => {
    if (disabled || ev.button !== 0) return;
    ev.preventDefault();
    const grip = ((ev.target as HTMLElement).dataset.grip as Grip | undefined) ?? null;
    const at = frameAt(ev.clientX);
    let start = range;
    if (!grip) {
      // Centre the range on the frame pressed, then drag it from there.
      start = rangeFor("centre", Math.floor(at) - Math.round((s + e) / 2), range);
      onDrag(start);
    }
    drag.current = { grip: grip ?? "centre", at, start, now: start };
    dragging.current = true;
    setHeld(true);
    ev.currentTarget.setPointerCapture(ev.pointerId);
  };
  const onMove = (ev: ReactPointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d) return;
    const next = rangeFor(d.grip, Math.round(frameAt(ev.clientX) - d.at), d.start);
    if (next[0] !== d.now[0] || next[1] !== d.now[1]) {
      d.now = next;
      onDrag(next);
    }
  };
  const onUp = () => {
    const d = drag.current;
    drag.current = null;
    dragging.current = false;
    setHeld(false);
    if (d) onDone(d.now);
  };

  const pct = (k: number) => `${(k / n) * 100}%`;
  // The limits stay clear of the centre handle, however narrow the range.
  const mid = pct((s + e + 1) / 2);
  return (
    <div
      className={`range-slider${disabled ? " disabled" : ""}${held ? " held" : ""}`}
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerCancel={onUp}
      aria-label="Frame range"
      title="Drag the middle to move the range, an end to widen or narrow it (← / → move it)"
    >
      <div ref={track} className="range-track">
        <div className="range-band" style={{ left: pct(s), width: pct(e + 1 - s) }} />
        <div className="range-limit" data-grip="start" style={{ left: `min(${pct(s)}, calc(${mid} - ${LIMIT_GAP}px))` }}>
          {held && <span className="range-tip">{s + 1}</span>}
        </div>
        <div className="range-limit" data-grip="end" style={{ left: `max(${pct(e + 1)}, calc(${mid} + ${LIMIT_GAP}px))` }}>
          {held && <span className="range-tip">{e + 1}</span>}
        </div>
        <div className="range-centre" data-grip="centre" style={{ left: mid }} />
      </div>
    </div>
  );
}

/** The From / To bubble above the Mean and Sum buttons: Enter or Apply sets
 * the range (frames from 1); Esc or a click elsewhere closes it. */
function RangeBubble({ at, n, range, onApply, onClose }: {
  at: { x: number; y: number };
  n: number;
  range: Range;
  onApply: (r: Range) => void;
  onClose: () => void;
}) {
  const [from, setFrom] = useState(String(range[0] + 1));
  const [to, setTo] = useState(String(range[1] + 1));
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const away = (e: PointerEvent) => {
      const t = e.target as Element;
      if (!box.current?.contains(t) && !t.closest?.(".insitu-modes")) onClose();
    };
    document.addEventListener("pointerdown", away);
    // Placed on the page, not in the bar: it would be left behind.
    window.addEventListener("resize", onClose);
    return () => {
      document.removeEventListener("pointerdown", away);
      window.removeEventListener("resize", onClose);
    };
  }, [onClose]);

  const a = Number(from);
  const b = Number(to);
  const valid = Number.isInteger(a) && Number.isInteger(b) && a >= 1 && b <= n && a <= b;
  const apply = () => valid && onApply([a - 1, b - 1]);
  const keys = (e: ReactKeyboardEvent) => {
    e.stopPropagation();
    if (e.key === "Enter") apply();
    if (e.key === "Escape") onClose();
  };
  return createPortal(
    // On the page itself, above every pane (the viewer's pane would clip it).
    <div className="range-bubble" style={{ left: at.x, top: at.y }} ref={box} role="dialog" aria-label="Frames to reduce">
      <label>
        From
        <input type="number" min={1} max={n} value={from} autoFocus onFocus={(e) => e.currentTarget.select()} onChange={(e) => setFrom(e.target.value)} onKeyDown={keys} />
      </label>
      <label>
        To
        <input type="number" min={1} max={n} value={to} onChange={(e) => setTo(e.target.value)} onKeyDown={keys} />
      </label>
      <button className="primary small-button" disabled={!valid} onClick={apply}>
        Apply
      </button>
    </div>,
    document.body,
  );
}
