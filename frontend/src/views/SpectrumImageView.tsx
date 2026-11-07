import { useEffect, useMemo, useRef, useState } from "react";
import { fetchEnergyMap, fetchFrame, type Frame, type PixelsSpectrum } from "../api/arrays";
import { Coalescer } from "../api/coalesce";
import { api, dataVersion, type DatasetSummary, type NavigatorSource } from "../api/client";
import { ElementsPicker } from "../components/ElementsPicker";
import { LayeredViewer } from "../components/LayeredViewer";
import { RoiOverlay } from "../components/RoiOverlay";
import { SeriesLegend } from "../components/SeriesLegend";
import { SpectrumPlot, type EnergyRange, type Marker, type SpectrumSeries } from "../components/SpectrumPlot";
import { allDatasets, useDatasets } from "../state/datasets";
import { useDisplay } from "../state/display";
import { CURSOR_COLOR, getSiState, useSi, useSiState, type Reduce, type Tool } from "../state/si";
import { autoRange, sortedSample } from "../viewer/contrast";
import { isColour } from "../viewer/colormaps";
import { bytesLabel, formatValue } from "../format";
import { imageGeometry, isAbort } from "./geometry";
import { useNavigatorTools } from "./si/useNavigatorTools";
import { CURSOR_CHANNEL, LINE_CHANNEL, useSiSpectra, type LineBand } from "./si/useSiSpectra";
import { useViewProfiles } from "../state/profiles";
import { useUi } from "../state/ui";
import { useShortcuts } from "../keyboard";
import { bandPolygon, pixelLength, pointAt } from "../viewer/profile";
import { fwhmAt, useLineMarkers } from "./si/xray";
import { SPECTRUM_COLOR, useEnergyAxis, useSpectrum } from "./SpectrumView";
import { useFitSeries } from "./useFitSeries";

/** Navigator source meaning "no base image": only the overlaid maps are shown. */
const NONE = "none";
/** Navigator source showing the map of the energy window picked on the spectrum. */
const WINDOW = "window";

const TOOLS: { tool: Tool; label: string; title: string }[] = [
  { tool: "cursor", label: "✛", title: "Cursor: click or drag to see the spectrum at a pixel (arrow keys nudge it)" },
  { tool: "rect", label: "▭", title: "Rectangle region: drag to draw, drag inside to move" },
  { tool: "ellipse", label: "◯", title: "Ellipse region: drag to draw, drag inside to move" },
  { tool: "polygon", label: "⬠", title: "Polygon region: click vertices; click the first, double-click or Enter to close" },
  {
    tool: "line",
    label: "╱",
    title: "Line profile: drag to draw (Shift snaps the angle); click on the profile for the spectrum there",
  },
  { tool: "pan", label: "✋", title: "Pan (or hold Space / middle-drag with any tool)" },
];

const scaled = (data: Float32Array, k: number) => (k === 1 ? data : data.map((v) => v * k));

/** Navigator image (ADF, total counts, a map or an energy window) with the
 * spectrum cursor and regions, beside their spectra. */
export function SpectrumImageView({ ds, images }: { ds: DatasetSummary; images: DatasetSummary[] }) {
  const updateDataset = useDatasets((s) => s.updateDataset);
  const [loadError, setLoadError] = useState<string | null>(null);
  const ready = !ds.lazy;
  const g = imageGeometry(ds, "navigation");

  // Spectrum images are loaded into RAM up front: everything else is fast after.
  useEffect(() => {
    if (!ds.lazy) return;
    api
      .loadDataset(ds.id)
      .then(updateDataset)
      .catch((e) => setLoadError((e as Error).message));
  }, [ds.id, ds.lazy, updateDataset]);

  // ---------- interaction state ----------
  const st = useSiState(ds.id);
  const { tool, setTool, cursorSize, reduce, setReduce, patch } = useSi();
  useEffect(() => {
    // Start with the cursor in the middle so a pixel spectrum shows at once.
    if (!getSiState(ds.id).cursor && g.width) {
      patch(ds.id, { cursor: { x: Math.floor(g.width / 2), y: Math.floor(g.height / 2) } });
    }
  }, [ds.id, g.width, g.height, patch]);
  const tools = useNavigatorTools(ds.id, g.width, g.height);
  useShortcuts({
    c: () => setTool("cursor"),
    r: () => setTool("rect"),
    e: () => setTool("ellipse"),
    p: () => setTool("polygon"),
    l: () => setTool("line"),
    h: () => setTool("pan"),
  });
  // Cursor and region tools bring the spectrum tab back (the line tool shows profiles).
  const setBottomTab = useUi((s) => s.setBottomTab);
  useEffect(() => {
    if (tool !== "line" && tool !== "pan") setBottomTab("spectrum");
  }, [tool, setBottomTab]);

  // ---------- navigator ----------
  const [sources, setSources] = useState<NavigatorSource[]>([]);
  const [sourceId, setSourceId] = useState<string | null>(null);
  const files = useDatasets((s) => s.files);
  // Refetched when the file's maps change (computed maps added or removed).
  const fileKey = files.find((f) => f.id === ds.file_id)?.datasets.map((d) => d.id).join(",");
  useEffect(() => {
    api.navigators(ds.id).then((s) => {
      setSources(s);
      // Keep the current source unless it was deleted.
      setSourceId((cur) =>
        cur && (cur === NONE || cur === WINDOW || s.some((x) => x.id === cur)) ? cur : (s[0]?.id ?? ds.id),
      );
    });
  }, [ds.id, fileKey]);

  // The frame is kept with the source it came from: right after switching
  // source the previous image must not be used (e.g. for auto contrast).
  const [nav, setNav] = useState<{ source: string; version: string; frame: Frame } | null>(null);
  const [navError, setNavError] = useState<string | null>(null);
  // The total-counts map (source = the SI itself) needs the SI in memory.
  const sourceReady =
    sourceId !== null && sourceId !== NONE && sourceId !== WINDOW && (sourceId !== ds.id || ready);
  const sourceSummary = allDatasets(files).find((d) => d.id === sourceId);
  const sourceVersion = sourceSummary ? dataVersion(sourceSummary) : "";
  useEffect(() => {
    if (!sourceReady) return;
    const ac = new AbortController();
    const source = sourceId!;
    const version = sourceVersion;
    fetchFrame(source, {}, ac.signal)
      .then((frame) => setNav({ source, version, frame }))
      .catch((e) => !isAbort(e) && setNavError((e as Error).message));
    return () => ac.abort();
  }, [sourceId, sourceReady, sourceVersion]);

  // Energy-window maps follow the window live (latest-only while dragging).
  const windowMaps = useRef<Coalescer<Frame>>();
  windowMaps.current ??= new Coalescer(
    (_c, frame) => setNav({ source: WINDOW, version: "", frame }),
    (_c, e) => setNavError(e.message),
  );
  const win = st.window;
  useEffect(() => {
    if (sourceId !== WINDOW || !win || !ready) return;
    // Keyed by the filter too: the map follows the spectrum image's spatial filter.
    windowMaps.current!.request("map", `${win.lo},${win.hi}@${ds.filter_rev}`, () =>
      fetchEnergyMap(ds.id, win.lo, win.hi),
    );
  }, [sourceId, win, ready, ds.id, ds.filter_rev]);

  const navFrame =
    nav && nav.source === sourceId && (sourceId === WINDOW || nav.version === sourceVersion) ? nav.frame : null;
  const sourceDataset =
    sourceId && sourceId !== ds.id ? allDatasets(files).find((d) => d.id === sourceId) : undefined;
  // A navigator image shares its display settings (contrast, element colour)
  // with the dataset itself; total counts and window maps have their own.
  const navKey =
    sourceId === ds.id ? `${ds.id}:total` : sourceId === WINDOW ? `${ds.id}:window` : (sourceId ?? "");

  // Each new window map is auto-contrasted; a window taken from an element's
  // line is shown in that element's colour.
  useEffect(() => {
    if (sourceId !== WINDOW || !navFrame) return;
    const [lo, hi] = autoRange(sortedSample(navFrame.data));
    const { display, elementColours, setDisplay, initDisplay } = useDisplay.getState();
    // A manual window drops an element colour left over from a line window,
    // but keeps a named colormap the user picked.
    const current = display[navKey]?.colormap;
    const colormap = win?.element
      ? elementColours[win.element]
      : current && isColour(current)
        ? "gray"
        : undefined;
    if (!display[navKey]) initDisplay(navKey, { lo, hi, gamma: 1, colormap: colormap ?? "gray" });
    else setDisplay(navKey, { lo, hi, ...(colormap && { colormap }) });
  }, [navFrame]); // eslint-disable-line react-hooks/exhaustive-deps

  const windowLabel = win
    ? win.label
      ? `${win.label} (${formatValue(win.lo)}–${formatValue(win.hi)} keV)`
      : `${formatValue(win.lo)}–${formatValue(win.hi)} keV`
    : "";
  const sourceOptions = [...sources, ...(win ? [{ id: WINDOW, label: `Energy window: ${windowLabel}` }] : [])];

  // ---------- spectra ----------
  const axis = useEnergyAxis(ds);
  const whole = useSpectrum(ds, ready);
  // A point or stretch picked on a line's profile: its spectrum (the band
  // across the line's width; a point covers one pixel along it).
  const navView = `${ds.id}:nav`;
  const profiles = useViewProfiles(navView);
  const pickLine = profiles.pick && profiles.lines.find((l) => l.id === profiles.pick!.line && l.visible);
  const lineBand = useMemo<LineBand | null>(() => {
    if (!pickLine || !profiles.pick) return null;
    let { t0, t1 } = profiles.pick;
    const half = 0.5 / Math.max(1, pixelLength(pickLine));
    if (t1 - t0 < 2 * half) [t0, t1] = [(t0 + t1) / 2 - half, (t0 + t1) / 2 + half];
    const [x, y] = pointAt(pickLine, (t0 + t1) / 2);
    const clamp = (v: number, n: number) => Math.min(n - 1, Math.max(0, Math.floor(v)));
    return { points: bandPolygon(pickLine, Math.max(1, pickLine.width), t0, t1), pixel: [clamp(x, g.width), clamp(y, g.height)] };
  }, [pickLine, profiles.pick, g.width, g.height]);
  const spectra = useSiSpectra(ds.id, dataVersion(ds), st, cursorSize, ready, lineBand);
  const totalPixels = g.width * g.height;
  const fitSeries = useFitSeries(ds, axis.x, reduce === "mean");
  const series = useMemo<SpectrumSeries[]>(() => {
    const k = (n: number) => (reduce === "mean" ? 1 / n : 1);
    const out: SpectrumSeries[] = [];
    if (st.showWhole && whole.data) {
      out.push({ label: "All pixels", data: scaled(whole.data, k(totalPixels)), color: SPECTRUM_COLOR });
    }
    for (const r of st.rois) {
      const s: PixelsSpectrum | undefined = spectra[r.id];
      if (r.visible && s && s.pixels > 0) out.push({ label: r.name, data: scaled(s.data, k(s.pixels)), color: r.color });
    }
    const ls = spectra[LINE_CHANNEL];
    if (pickLine && lineBand && ls) {
      out.push({ label: `${pickLine.name} (profile)`, data: scaled(ls.data, k(ls.pixels)), color: pickLine.color });
    }
    const c = spectra[CURSOR_CHANNEL];
    if (st.showCursor && st.cursor && c) out.push({ label: "Cursor", data: scaled(c.data, k(c.pixels)), color: CURSOR_COLOR });
    return [...out, ...fitSeries];
  }, [st.showWhole, st.showCursor, st.cursor, st.rois, whole.data, spectra, reduce, totalPixels, fitSeries, pickLine, lineBand]);

  // ---------- elements, line markers and energy windows ----------
  const [preview, setPreview] = useState<string | null>(null);
  const markers = useLineMarkers(ds.eds?.elements ?? [], preview, ds.eds?.beam_energy ?? null, ds.eds?.xray_lines, [axis.x[0], axis.x[axis.x.length - 1]]);
  const showWindow = (w: EnergyRange & { element?: string; label?: string }) => {
    patch(ds.id, { window: { ...w, lo: Math.min(w.lo, w.hi), hi: Math.max(w.lo, w.hi) } });
    setSourceId(WINDOW);
  };
  const onMarkerClick = (m: Marker) => {
    const fw = fwhmAt(m.energy, ds.eds?.energy_resolution_MnKa ?? 130);
    showWindow({ lo: m.energy - fw, hi: m.energy + fw, element: m.element, label: m.label });
  };

  const loadingMsg = `Loading spectrum image into memory (${bytesLabel(ds.nbytes)})…`;

  const spectrumPlot = (
    <SpectrumPlot
      name={`${ds.title}_spectra`}
      x={axis.x}
      series={series}
      xLabel={axis.label}
      xUnits={axis.units}
      loading={whole.data ? null : ready ? "Summing spectra…" : loadingMsg}
      error={whole.error ?? loadError}
      markers={markers}
      window={win}
      onWindow={(range) => showWindow(range)}
      onMarkerClick={onMarkerClick}
      toolbar={
        <>
          <ElementsPicker ds={ds} onPreview={setPreview} />
          <div className="segmented" role="group" aria-label="Spectrum normalisation">
            {(["mean", "sum"] as Reduce[]).map((r) => (
              <button
                key={r}
                className={reduce === r ? "active" : undefined}
                aria-pressed={reduce === r}
                onClick={() => setReduce(r)}
                title={r === "mean" ? "Average spectrum per pixel (comparable across regions)" : "Summed counts"}
              >
                {r === "mean" ? "Mean" : "Sum"}
              </button>
            ))}
          </div>
        </>
      }
      legend={
        <SeriesLegend
          si={ds.id}
          state={st}
          spectra={spectra}
          totalPixels={totalPixels}
          wholeColor={SPECTRUM_COLOR}
        />
      }
    />
  );

  return (
    <div className="view si-view">
      <section className="si-pane">
        <div className="si-pane-header">
          <label className="field">
            Navigator
            <select value={sourceId ?? ""} onChange={(e) => setSourceId(e.target.value)}>
              {sourceOptions.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
              <option value={NONE}>None (maps only)</option>
            </select>
          </label>
          <div className="segmented tool-buttons" role="group" aria-label="Navigator tool">
            {TOOLS.map((t) => (
              <button
                key={t.tool}
                className={tool === t.tool ? "active" : undefined}
                aria-pressed={tool === t.tool}
                title={t.title}
                onClick={() => setTool(t.tool)}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>
        <LayeredViewer
          viewKey={navView}
          lineTool={tool === "line"}
          profilePick
          base={
            sourceId === NONE
              ? null
              : {
                  key: navKey,
                  label:
                    sourceId === WINDOW
                      ? windowLabel
                      : (sources.find((s) => s.id === sourceId)?.label ?? "Navigator"),
                  frame: navFrame,
                  dataset: sourceDataset,
                  version: sourceVersion,
                }
          }
          onRemoveBase={() => setSourceId(NONE)}
          images={images.filter((d) => d.id !== sourceId)}
          geometry={g}
          loading={
            navFrame || sourceId === NONE
              ? null
              : (sourceId === ds.id || sourceId === WINDOW) && !ready
                ? loadingMsg
                : "Loading navigator…"
          }
          error={navError ?? loadError}
          empty="No navigator image. Click element maps in the dataset pane to show them here."
          interaction={tools.interaction}
          renderOverlay={(t) => (
            <RoiOverlay t={t} state={st} cursorSize={cursorSize} draft={tools.draft} hover={tools.hover} />
          )}
          bottomTabs={[{ id: "spectrum", label: "Spectrum", content: spectrumPlot }]}
        />
      </section>
    </div>
  );
}
