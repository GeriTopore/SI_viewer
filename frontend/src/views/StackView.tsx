import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchStackFrame, type Frame } from "../api/arrays";
import { api, dataVersion, type DatasetSummary } from "../api/client";
import { BASE_LAYER, LayeredViewer, type StackLayer } from "../components/LayeredViewer";
import { AlignmentSummary } from "../components/AlignmentSummary";
import { initialColormap, useDatasets } from "../state/datasets";
import { formatValue } from "../format";
import { ComponentSpectrum, ScreePlot } from "../components/PcaTabs";
import { keptOf, usePca } from "../state/pca";
import { imageGeometry, isAbort } from "./geometry";
import { isTyping } from "../keyboard";

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

/** A frame of a stack shown as a panel of the view (``key``). */
interface FramePanel {
  key: string;
  dataset: DatasetSummary;
  /** Its frame (clamped to the stack's length). */
  index: number;
}

/** The frames of ``panels`` (other stacks shown with the main one, or frames
 * of one stack side by side), each at its own frame or all at the
 * projection: per panel, at most one request in flight, the latest wanted
 * fetched next, the last frame kept on screen meanwhile. Returns the frame of
 * a panel (by key). */
function usePanelFrames(panels: FramePanel[], projection: Projection, scrubbing: boolean) {
  const [shown, setShown] = useState<Record<string, { projection: Projection; version: string; frame: Frame }>>({});
  const caches = useRef(new Map<string, Map<string, Frame>>());
  const busy = useRef(new Set<string>());
  const wanted = useRef(new Map<string, { key: string; run: () => Promise<Frame>; projection: Projection; version: string }>());

  const pump = useCallback((id: string) => {
    const w = wanted.current.get(id);
    if (!w || busy.current.has(id)) return;
    const cache = caches.current.get(id) ?? new Map<string, Frame>();
    caches.current.set(id, cache);
    const show = (frame: Frame) =>
      setShown((s) => ({ ...s, [id]: { projection: w.projection, version: w.version, frame } }));
    const hit = cache.get(w.key);
    if (hit) {
      show(hit);
      return;
    }
    busy.current.add(id);
    w.run()
      .then((f) => {
        cache.delete(w.key);
        cache.set(w.key, f);
        if (cache.size > CACHE_SIZE) cache.delete(cache.keys().next().value!);
        show(f);
      })
      .catch(() => {})
      .finally(() => {
        busy.current.delete(id);
        if (wanted.current.get(id)?.key !== w.key) pump(id);
      });
  }, []);

  const versions = panels
    .map((p) => `${p.key}:${dataVersion(p.dataset)}:${p.dataset.navigation_shape[0]}:${p.index}`)
    .join(",");
  useEffect(() => {
    for (const { key, dataset: d, index } of panels) {
      const n = d.navigation_shape[0] ?? 1;
      const i = Math.min(index, n - 1);
      const g = imageGeometry(d, "signal");
      const bin = scrubbing ? Math.max(1, Math.ceil(Math.max(g.width, g.height) / PREVIEW_MAX_SIZE)) : 1;
      const version = dataVersion(d);
      const r: Request = { index: i, projection, bin };
      wanted.current.set(key, {
        key: `${version}|${keyOf(r)}`,
        projection,
        version,
        run: () => fetchStackFrame(d, projection === "frame" ? { index: i, bin } : { projection }),
      });
      pump(key);
    }
  }, [versions, projection, scrubbing, pump]); // eslint-disable-line react-hooks/exhaustive-deps

  return (key: string): Frame | null => {
    const s = shown[key];
    // Another projection's frame is not shown (its contrast is another's).
    return s && s.projection === projection ? s.frame : null;
  };
}

export function StackView({ ds, images, others = [] }: {
  ds: DatasetSummary;
  images: DatasetSummary[];
  /** Other stacks shown with it, at the same frame (one slider). */
  others?: DatasetSummary[];
}) {
  const n = ds.navigation_shape[0] ?? 1;
  const g = imageGeometry(ds, "signal");
  const previewBin = Math.max(1, Math.ceil(Math.max(g.width, g.height) / PREVIEW_MAX_SIZE));

  const [index, setIndex] = useState(0);
  // A PCA: its kept components' maps side by side.
  const [pcaGrid, setPcaGrid] = useState(false);
  const [projection, setProjection] = useState<Projection>("frame");
  const [scrubbing, setScrubbing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Frames share one contrast setting (base key) so scrubbing doesn't flicker;
  // projections (with different ranges) get their own. The shown image is
  // kept with its projection, so right after switching the previous one isn't
  // used (e.g. to auto-contrast the new projection).
  // Frames are also kept with the filters they were made with.
  const [shown, setShown] = useState<{ projection: Projection; filterRev: number; index: number; frame: Frame } | null>(null);
  const frame = shown && shown.projection === projection && shown.filterRev === ds.filter_rev ? shown.frame : null;
  const filterRev = useRef(ds.filter_rev);

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
    fetchStackFrame(ds, r.projection === "frame" ? { index: r.index, bin: r.bin } : { projection: r.projection });

  const pump = useCallback(async () => {
    if (inFlight.current) return;
    const r = wanted.current;
    const key = keyOf(r);
    const rev = filterRev.current;
    const hit = cache.current.get(key);
    if (hit) {
      setShown({ projection: r.projection, filterRev: rev, index: r.index, frame: hit });
      return;
    }
    inFlight.current = true;
    try {
      const f = await get(r);
      if (rev !== filterRev.current) return; // filtered differently meanwhile
      remember(key, f);
      setShown({ projection: r.projection, filterRev: rev, index: r.index, frame: f });
      setError(null);
    } catch (e) {
      if (!isAbort(e)) setError((e as Error).message);
    } finally {
      inFlight.current = false;
      if (keyOf(wanted.current) !== key || rev !== filterRev.current) pump();
    }
  }, [ds.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // New filters: cached frames are stale.
  if (filterRev.current !== ds.filter_rev) {
    filterRev.current = ds.filter_rev;
    cache.current.clear();
  }

  useEffect(() => {
    const bin = scrubbing ? previewBin : 1;
    wanted.current = { index, projection, bin };
    pump();
    // Prefetch the neighbours at full resolution once the slider settles.
    if (!scrubbing && projection === "frame") {
      for (const i of [index + 1, index - 1]) {
        const r = { index: i, projection, bin: 1 } as Request;
        if (i >= 0 && i < n && !cache.current.has(keyOf(r))) {
          const rev = filterRev.current;
          get(r).then((f) => rev === filterRev.current && remember(keyOf(r), f)).catch(() => {});
        }
      }
    }
  }, [index, projection, scrubbing, previewBin, n, pump, ds.filter_rev]); // eslint-disable-line react-hooks/exhaustive-deps

  // The backend loads small stacks into memory on the first frame request;
  // refresh the summary so the Inspector says so.
  const updateDataset = useDatasets((s) => s.updateDataset);
  const loadedFrame = frame !== null;
  useEffect(() => {
    if (ds.lazy && loadedFrame) api.getDataset(ds.id).then(updateDataset).catch(() => {});
  }, [ds.id, ds.lazy, loadedFrame, updateDataset]);

  // Drift-corrected stacks: switch between the aligned frames (fewer,
  // cropped) and the frames as loaded.
  const replaceFile = useDatasets((s) => s.replaceFile);
  const alignment = ds.alignment;
  const showAligned = async (active: boolean) => {
    try {
      replaceFile((await api.showAlignment(ds.id, active)).file);
    } catch (e) {
      setError((e as Error).message);
    }
  };
  useEffect(() => setIndex((i) => Math.min(i, n - 1)), [n]);

  // ←/→ step through frames (unless typing in a field).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || isTyping(e)) return;
      if (projection !== "frame") return;
      if (e.key === "ArrowRight") setIndex((i) => Math.min(n - 1, i + 1));
      else if (e.key === "ArrowLeft") setIndex((i) => Math.max(0, i - 1));
      else if (e.key === "Home") setIndex(0);
      else if (e.key === "End") setIndex(n - 1);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [n, projection]);

  const isPca = ds.origin === "pca" && !!ds.pca;
  const pcaKept = usePca((s) => (isPca ? keptOf(ds, s.kept[ds.id]) : 0));
  // A PCA's kept components side by side: a panel per component (the first
  // as the base), each a frame of this stack with its own contrast.
  const showGrid = isPca && pcaGrid && projection === "frame";
  const componentKey = (i: number) => (i === 0 ? BASE_LAYER : `${ds.id}#${i}`);
  const componentLabel = (i: number) => `Component ${i + 1} · ${formatValue((ds.pca?.explained[i] ?? 0) * 100)} %`;
  const components = showGrid ? Array.from({ length: pcaKept }, (_, i) => i) : [];
  const panelFrame = usePanelFrames(
    [
      ...others.map((d) => ({ key: d.id, dataset: d, index })),
      ...components.map((i) => ({ key: `${ds.id}#${i}`, dataset: ds, index: i })),
    ],
    projection,
    scrubbing,
  );
  const stacks: StackLayer[] = showGrid
    ? components.slice(1).map((i) => ({
        dataset: ds,
        key: componentKey(i),
        label: componentLabel(i),
        displayKey: `${ds.id}:pc${i}`,
        frame: panelFrame(`${ds.id}#${i}`),
        version: String(ds.filter_rev),
        removable: false,
      }))
    : others.map((d) => ({
        dataset: d,
        displayKey: `${d.id}:${projection}`,
        frame: panelFrame(d.id),
        version: String(d.filter_rev),
      }));
  // In the grid the component shown is the panel being edited.
  const activeRequest = useMemo(() => (showGrid ? { key: componentKey(index) } : null), [showGrid, index]); // eslint-disable-line react-hooks/exhaustive-deps
  const frameLabel = isPca && projection === "frame"
    ? `Component ${index + 1} / ${n} · ${formatValue((ds.pca!.explained[index] ?? 0) * 100)} %`
    : (projection === "frame" ? `Frame ${index + 1} / ${n}` : `${projection} of ${n} frames`) +
      (alignment?.active ? " · aligned" : "");

  const stackBar = (
    <div className="stack-bar">
      <input
        type="range"
        className="stack-slider"
        min={0}
        max={n - 1}
        step={1}
        value={index}
        disabled={projection !== "frame" || showGrid}
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
      {isPca && (
        <button
          className={showGrid ? "active" : undefined}
          aria-pressed={showGrid}
          onClick={() => {
            setPcaGrid(!showGrid);
            setProjection("frame");
          }}
          title="The kept components' maps side by side (the one shown is outlined)"
        >
          Grid · 1–{pcaKept}
        </button>
      )}
      {alignment && (
        <div className="segmented" role="group" aria-label="Drift correction">
          {[false, true].map((on) => (
            <button
              key={String(on)}
              className={alignment.active === on ? "active" : undefined}
              aria-pressed={alignment.active === on}
              onClick={() => alignment.active !== on && showAligned(on)}
              title={on ? "Frames after drift correction (cropped)" : "Frames as loaded"}
            >
              {on ? "Aligned" : "Original"}
            </button>
          ))}
        </div>
      )}
    </div>
  );

  return (
    <div className="view">
      <LayeredViewer
        viewKey={ds.id}
        base={
          showGrid
            ? {
                key: `${ds.id}:pc0`,
                label: componentLabel(0),
                frame: panelFrame(`${ds.id}#0`),
                version: String(ds.filter_rev),
                colormap: () => initialColormap(ds),
              }
            : {
                key: `${ds.id}:${projection}`,
                label: ds.title,
                frame,
                // A PCA's maps differ in range: each is auto-contrasted as it is shown.
                version: isPca && projection === "frame" ? `${ds.filter_rev}:${shown?.index ?? index}` : String(ds.filter_rev),
                colormap: () => initialColormap(ds),
              }
        }
        images={images}
        stacks={stacks}
        gridContext={others.length || showGrid ? "stacks" : undefined}
        requestActive={activeRequest}
        onActiveChange={(key) => {
          if (!showGrid) return;
          const i = key === BASE_LAYER ? 0 : Number(key.split("#")[1]);
          if (Number.isInteger(i) && i !== index) setIndex(i);
        }}
        geometry={g}
        loading={(showGrid ? panelFrame(`${ds.id}#0`) : frame) ? null : "Loading frame…"}
        error={error}
        footer={stackBar}
        overlays={false}
        bottomTabs={[
          ...(alignment ? [{ id: "alignment" as const, label: "Alignment", content: <AlignmentSummary ds={ds} calX={g.calX} /> }] : []),
          ...(isPca
            ? [
                { id: "pca-component" as const, label: "Component", content: <ComponentSpectrum pca={ds} index={index} /> },
                {
                  id: "pca-scree" as const,
                  label: "Scree",
                  content: (
                    <ScreePlot
                      pca={ds}
                      current={projection === "frame" ? index : undefined}
                      onShow={(i) => {
                        setProjection("frame");
                        setIndex(i);
                      }}
                    />
                  ),
                },
              ]
            : []),
        ]}
      >
        <span className="frame-chip">{frameLabel}</span>
      </LayeredViewer>
    </div>
  );
}
