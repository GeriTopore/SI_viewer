import { useEffect, useRef, useState } from "react";
import type { DatasetSummary, InsituView } from "../api/client";
import { useShortcuts } from "../keyboard";
import { reducing, useInsitu } from "../state/insitu";
import { frameTime, insituViewLabel } from "../format";

type Mode = InsituView["mode"];

// A slider move (or a key) is read this long after the last change, so
// stepping quickly doesn't ask for every frame in between.
const DEBOUNCE_MS = 150;

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
  const [from, setFrom] = useState(String(view.mode === "frame" ? 1 : view.start + 1));
  const [to, setTo] = useState(String(view.mode === "frame" ? n : view.end + 1));

  const dragging = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout>>();
  const latest = useRef(index);
  latest.current = index;
  useEffect(() => () => clearTimeout(timer.current), []);

  // Follow the view when it changes under us (a finished range job).
  useEffect(() => {
    if (running || reading !== undefined || dragging.current) return;
    setMode(view.mode);
    if (view.mode === "frame") setIndex(view.start);
    else {
      setFrom(String(view.start + 1));
      setTo(String(view.end + 1));
    }
  }, [view.mode, view.start, view.end]); // eslint-disable-line react-hooks/exhaustive-deps

  const goTo = (k: number, now = false) => {
    const next = clamp(k, 0, n - 1);
    setIndex(next);
    latest.current = next;
    clearTimeout(timer.current);
    if (dragging.current) return; // read on release
    if (now) showFrame(ds.id, next);
    else timer.current = setTimeout(() => showFrame(ds.id, latest.current), DEBOUNCE_MS);
  };

  // ←/→ step one frame (⇧ ten), Home/End go to the ends; not while typing.
  useShortcuts(
    {
      ArrowLeft: (e) => mode === "frame" && goTo(latest.current - (e.shiftKey ? 10 : 1)),
      ArrowRight: (e) => mode === "frame" && goTo(latest.current + (e.shiftKey ? 10 : 1)),
      Home: () => mode === "frame" && goTo(0),
      End: () => mode === "frame" && goTo(n - 1),
    },
    true,
  );

  const choose = (m: Mode) => {
    if (m === mode || running) return;
    setMode(m);
    if (m === "frame" && !(view.mode === "frame" && view.start === latest.current)) goTo(latest.current, true);
  };

  const a = Number(from);
  const b = Number(to);
  const validRange = Number.isInteger(a) && Number.isInteger(b) && a >= 1 && b <= n && a <= b;
  const apply = () => validRange && reduce(ds.id, mode as "mean" | "sum", a - 1, b - 1);

  const target = clamp(index, 0, n - 1);
  const timeText = times[target] !== undefined ? ` · ${frameTime(times[target])}` : "";
  const progress = job && running ? Math.round(job.progress * 100) : 0;

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
          <span className="stack-index" title="Frame / frames · time of the frame">
            {target + 1} / {n}
            {timeText}
          </span>
        </>
      ) : (
        <div className="insitu-range">
          <label title="First frame of the range">
            From{" "}
            <input type="number" min={1} max={n} value={from} disabled={running} onChange={(e) => setFrom(e.target.value)} />
          </label>
          <label title="Last frame of the range">
            To{" "}
            <input type="number" min={1} max={n} value={to} disabled={running} onChange={(e) => setTo(e.target.value)} />
          </label>
          <span title="Frames in the series">/ {n}</span>
          <button
            className="secondary small-button"
            onClick={apply}
            disabled={!validRange || running}
            title={`${mode === "mean" ? "Mean" : "Sum"} of the frames in the range: reads every one of them`}
          >
            Apply
          </button>
        </div>
      )}
      <div className="segmented" role="group" aria-label="Frame or range">
        {(["frame", "mean", "sum"] as Mode[]).map((m) => (
          <button
            key={m}
            className={mode === m ? "active" : undefined}
            aria-pressed={mode === m}
            disabled={running}
            onClick={() => choose(m)}
            title={m === "frame" ? "One frame" : m === "mean" ? "Mean of a range of frames" : "Sum of a range of frames"}
          >
            {m === "frame" ? "Frame" : m === "mean" ? "Mean" : "Sum"}
          </button>
        ))}
      </div>
      {running && job ? (
        <div className="insitu-progress">
          <div className="q-progress" role="progressbar" aria-valuenow={progress}>
            <div style={{ width: `${progress}%` }} />
          </div>
          <span>{job.message || "Starting…"}</span>
          <button className="link-button" onClick={() => cancel(ds.id)}>
            Cancel
          </button>
        </div>
      ) : reading !== undefined ? (
        <span className="insitu-status">
          <span className="spinner" aria-hidden />
          Reading frame {reading + 1}…
        </span>
      ) : error ? (
        <span className="insitu-status error" role="alert">
          {error}
        </span>
      ) : (
        <span className="insitu-status" title="What the view holds">
          {insituViewLabel(info)}
        </span>
      )}
    </div>
  );
}
