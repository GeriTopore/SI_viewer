import { useCallback, useEffect, useRef, useState } from "react";
import { fetchFrame, type Frame } from "../api/arrays";
import { api, type DatasetSummary } from "../api/client";
import { LayeredViewer } from "../components/LayeredViewer";
import { useDatasets } from "../state/datasets";
import { imageGeometry, isAbort } from "./geometry";

type Projection = "frame" | "mean" | "sum";

interface Request {
  index: number;
  projection: Projection;
  bin: number;
}

const CACHE_SIZE = 32;
// While dragging the slider, frames larger than this are fetched binned.
const PREVIEW_MAX_SIZE = 1024;

const keyOf = (r: Request) => (r.projection === "frame" ? `${r.index}@${r.bin}` : `${r.projection}`);

export function StackView({ ds, images }: { ds: DatasetSummary; images: DatasetSummary[] }) {
  const n = ds.navigation_shape[0] ?? 1;
  const g = imageGeometry(ds, "signal");
  const previewBin = Math.max(1, Math.ceil(Math.max(g.width, g.height) / PREVIEW_MAX_SIZE));

  const [index, setIndex] = useState(0);
  const [projection, setProjection] = useState<Projection>("frame");
  const [scrubbing, setScrubbing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Frames share one contrast setting (base key) so scrubbing doesn't flicker;
  // projections (with different ranges) get their own. The shown image is
  // kept with its projection, so right after switching the previous one isn't
  // used (e.g. to auto-contrast the new projection).
  const [shown, setShown] = useState<{ projection: Projection; frame: Frame } | null>(null);
  const frame = shown && shown.projection === projection ? shown.frame : null;

  // At most one request in flight; when it lands, fetch whatever is wanted now.
  // Every frame that arrives is shown, so dragging stays responsive.
  const cache = useRef(new Map<string, Frame>());
  const wanted = useRef<Request>({ index: 0, projection: "frame", bin: 1 });
  const inFlight = useRef(false);

  const remember = (key: string, f: Frame) => {
    cache.current.delete(key);
    cache.current.set(key, f);
    if (cache.current.size > CACHE_SIZE) cache.current.delete(cache.current.keys().next().value!);
  };

  const get = (r: Request) =>
    fetchFrame(ds.id, r.projection === "frame" ? { index: r.index, bin: r.bin } : { projection: r.projection });

  const pump = useCallback(async () => {
    if (inFlight.current) return;
    const r = wanted.current;
    const key = keyOf(r);
    const hit = cache.current.get(key);
    if (hit) {
      setShown({ projection: r.projection, frame: hit });
      return;
    }
    inFlight.current = true;
    try {
      const f = await get(r);
      remember(key, f);
      setShown({ projection: r.projection, frame: f });
      setError(null);
    } catch (e) {
      if (!isAbort(e)) setError((e as Error).message);
    } finally {
      inFlight.current = false;
      if (keyOf(wanted.current) !== key) pump();
    }
  }, [ds.id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const bin = scrubbing ? previewBin : 1;
    wanted.current = { index, projection, bin };
    pump();
    // Prefetch the neighbours at full resolution once the slider settles.
    if (!scrubbing && projection === "frame") {
      for (const i of [index + 1, index - 1]) {
        const r = { index: i, projection, bin: 1 } as Request;
        if (i >= 0 && i < n && !cache.current.has(keyOf(r))) {
          get(r).then((f) => remember(keyOf(r), f)).catch(() => {});
        }
      }
    }
  }, [index, projection, scrubbing, previewBin, n, pump]); // eslint-disable-line react-hooks/exhaustive-deps

  // The backend loads small stacks into memory on the first frame request;
  // refresh the summary so the Inspector says so.
  const updateDataset = useDatasets((s) => s.updateDataset);
  const loadedFrame = frame !== null;
  useEffect(() => {
    if (ds.lazy && loadedFrame) api.getDataset(ds.id).then(updateDataset).catch(() => {});
  }, [ds.id, ds.lazy, loadedFrame, updateDataset]);

  // ←/→ step through frames (unless typing in a field).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest("input, textarea, select")) return;
      if (projection !== "frame") return;
      if (e.key === "ArrowRight") setIndex((i) => Math.min(n - 1, i + 1));
      else if (e.key === "ArrowLeft") setIndex((i) => Math.max(0, i - 1));
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [n, projection]);

  const frameLabel = projection === "frame" ? `Frame ${index + 1} / ${n}` : `${projection} of ${n} frames`;

  return (
    <div className="view">
      <LayeredViewer
        viewKey={ds.id}
        base={{ key: `${ds.id}:${projection}`, label: ds.title, frame }}
        images={images}
        geometry={g}
        loading={frame ? null : "Loading frame…"}
        error={error}
      >
        <span className="frame-chip">{frameLabel}</span>
      </LayeredViewer>
      <div className="stack-bar">
        <input
          type="range"
          className="stack-slider"
          min={0}
          max={n - 1}
          step={1}
          value={index}
          disabled={projection !== "frame"}
          onChange={(e) => setIndex(Number(e.target.value))}
          onPointerDown={() => setScrubbing(true)}
          onPointerUp={() => setScrubbing(false)}
          onBlur={() => setScrubbing(false)}
          aria-label="Frame"
        />
        <span className="stack-index">
          {index + 1} / {n}
        </span>
        <div className="segmented" role="group" aria-label="Projection">
          {(["frame", "mean", "sum"] as Projection[]).map((p) => (
            <button
              key={p}
              className={projection === p ? "active" : undefined}
              aria-pressed={projection === p}
              onClick={() => setProjection(p)}
            >
              {p === "frame" ? "Frame" : p === "mean" ? "Mean" : "Sum"}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
