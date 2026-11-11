import { useEffect, useMemo, useRef, useState } from "react";
import { fetchEnergyMap, fetchFrame, fetchInsituAdf, type Frame, type PixelsSpectrum } from "../api/arrays";
import { Coalescer } from "../api/coalesce";
import { api, contrastVersion, dataVersion, type DatasetSummary, type DeconvolutionSelection, type NavigatorSource } from "../api/client";
import { ContextMenu } from "../components/ContextMenu";
import { AnnotationsToggle } from "../components/AnnotationsToggle";
import { ElementsPicker } from "../components/ElementsPicker";
import { InsituFrameBar } from "../components/InsituFrameBar";
import { BASE_LAYER, FFT_TOOL_TITLE, LayeredViewer } from "../components/LayeredViewer";
import { RoiOverlay } from "../components/RoiOverlay";
import { SeriesLegend } from "../components/SeriesLegend";
import { SpectrumPlot, type EnergyRange, type Marker, type PlotWindow, type SpectrumSeries } from "../components/SpectrumPlot";
import { allDatasets, useDatasets } from "../state/datasets";
import { useDisplay } from "../state/display";
import { baseLayerKey, useLayers } from "../state/layers";
import { CURSOR_COLOR, containsPoint, getSiState, scaleShape, shapeToRegion, useSi, useSiState, type Reduce, type Tool } from "../state/si";
import { reducing, useInsitu } from "../state/insitu";
import type { MenuItem } from "../components/ContextMenu";
import type { Pt } from "../components/ImageView";
import { autoRange, sortedSample } from "../viewer/contrast";
import { isColour } from "../viewer/colormaps";
import { bytesLabel, formatValue, insituViewLabel } from "../format";
import { imageGeometry, isAbort } from "./geometry";
import { useNavigatorTools } from "./si/useNavigatorTools";
import { navigatorGeometry, polygonToScan, scanInteraction, scanPixel, scanTransform } from "./si/scan";
import { CURSOR_CHANNEL, LINE_CHANNEL, useSiSpectra, type LineBand } from "./si/useSiSpectra";
import { getViewProfiles, useProfiles, useViewProfiles, type ProfileLine } from "../state/profiles";
import { eelsTab, useUi, type BottomTab } from "../state/ui";
import { stepKeys, useShortcuts } from "../keyboard";
import { bandPolygon, pixelLength, pointAt } from "../viewer/profile";
import { fwhmAt, useMarkers } from "./si/xray";
import { SPECTRUM_COLOR, useEnergyAxis, useSpectrum } from "./SpectrumView";
import { memberLabel, memberTabs, useGroup, useLossLog } from "./si/LinkedSpectrum";
import { useFitSeries } from "./useFitSeries";
import { useBackgroundPreview } from "./si/useBackgroundPreview";
import { useDeconvPreview } from "./si/useDeconvPreview";
import { useThresholdLine } from "./si/useThresholdLine";

/** Navigator source meaning "no base image": only the overlaid maps are shown. */
const NONE = "none";
/** Navigator source: the spectrum image summed over the main window. */
const VIRTUAL = "virtual";
/** Navigator source (in-situ series): the simultaneous ADF of the frame held (the backend's id). */
const ADF = "insitu-adf";

const TOOLS: { tool: Tool; label: string; title: string }[] = [
  { tool: "cursor", label: "✛", title: "Cursor: drag its marker to see the spectrum at another pixel (arrow keys nudge it)" },
  { tool: "rect", label: "▭", title: "Rectangle region: drag to draw, drag inside to move" },
  { tool: "ellipse", label: "◯", title: "Ellipse region: drag to draw, drag inside to move" },
  { tool: "polygon", label: "⬠", title: "Polygon region: click vertices; click the first, double-click or Enter to close" },
  {
    tool: "line",
    label: "╱",
    title: "Line profile: drag to draw (Shift snaps the angle); shift-click on the profile for the spectrum there",
  },
  { tool: "fft", label: "FFT", title: FFT_TOOL_TITLE },
];

/** Band colour of a window with no solid colour of its own. */
const NEUTRAL_WINDOW = "#4c8dff";
/** Id of the main window (the virtual image's) among the plot's windows. */
const MAIN = "main";
/** Id of the window being drawn. */
const DRAFT = "draft";
/** Band colour of the main window: neutral, apart from the coloured windows. */
const MAIN_COLOUR = "#8a8f98";

const scaled = (data: Float32Array, k: number) => (k === 1 ? data : data.map((v) => v * k));

/** Navigator image (ADF, the virtual image, a map) with the spectrum cursor
 * and regions, beside their spectra. */
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
  const tools = useNavigatorTools(ds.id, g.width, g.height, !!ds.insitu?.view);
  useShortcuts({
    c: () => setTool("cursor"),
    r: () => setTool("rect"),
    e: () => setTool("ellipse"),
    p: () => setTool("polygon"),
    l: () => setTool("line"),
    // Shift +/-: the selected region bigger or smaller (lines: the viewer's).
    ...stepKeys((step) => {
      if (tool === "line") return false;
      const { selected, rois } = getSiState(ds.id);
      const roi = rois.find((r) => r.id === selected);
      if (!roi) return false;
      const shape = scaleShape(roi.shape, step);
      if (shape) useSi.getState().updateRoi(ds.id, roi.id, { shape });
      return true;
    }),
  });
  // Cursor and region tools bring the spectrum tab back (the line tool shows profiles).
  // A linked EELS group has a tab for each member instead of "Spectrum".
  const group = useGroup(ds);
  const linked = !!ds.eels && group.length > 1;
  const setBottomTab = useUi((s) => s.setBottomTab);
  // Only on a change of tool (or group), not on opening the view: entering a
  // workspace brings back the tab it was left on.
  const lastTool = useRef({ tool, linked });
  useEffect(() => {
    const changed = lastTool.current.tool !== tool || lastTool.current.linked !== linked;
    lastTool.current = { tool, linked };
    if (!changed) return;
    if (tool === "line" || tool === "fft") return;
    const tab = useUi.getState().bottomTab;
    const mine: BottomTab[] = linked ? group.map((m) => eelsTab(m.id)) : ["spectrum"];
    if (!mine.includes(tab)) setBottomTab(linked ? eelsTab(ds.id) : "spectrum");
  }, [tool, setBottomTab, linked]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---------- navigator ----------
  const axis = useEnergyAxis(ds);
  const [sources, setSources] = useState<NavigatorSource[]>([]);
  const [sourceId, setSourceId] = useState<string | null>(null);
  const files = useDatasets((s) => s.files);
  // Refetched when the file's maps change (computed maps added or removed).
  const fileKey = files.find((f) => f.id === ds.file_id)?.datasets.map((d) => `${d.id}:${d.title}`).join(",");
  // In a linked group, the virtual image of every member can be the navigator.
  const members = linked ? group.filter((m) => m.id !== ds.id) : [];
  const membersKey = members.map((m) => m.id).join(",");
  useEffect(() => {
    api.navigators(ds.id).then((list) => {
      // The spectrum image itself (total counts) is the virtual image.
      const own = linked ? `Virtual image · ${memberLabel(ds, group)}` : "Virtual image";
      const s = list.map((x) => (x.id === ds.id ? { ...x, id: VIRTUAL, label: own } : x));
      s.splice(1, 0, ...members.map((m) => ({
        id: m.id, label: `Virtual image · ${memberLabel(m, group)}`, factor: 1, dataset_id: m.id,
      })));
      setSources(s);
      // Keep the current source unless it was deleted.
      setSourceId((cur) => (cur && (cur === NONE || s.some((x) => x.id === cur)) ? cur : (s[0]?.id ?? VIRTUAL)));
    });
  }, [ds.id, fileKey, membersKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // The main window: the range the virtual image sums (null: all of it).
  const isVirtual = sourceId === VIRTUAL;
  const main = st.main;
  const energyUnits = ds.axes.find((a) => !a.navigate)?.units ?? "";
  const rangeText = (r: { lo: number; hi: number }) =>
    `${formatValue(r.lo)}–${formatValue(r.hi)}${energyUnits && ` ${energyUnits}`}`;
  const setMain = (r: { lo: number; hi: number } | null) => {
    const x = axis.x;
    // The whole range is the total counts.
    if (r && x.length > 1 && r.lo <= x[0] + (x[1] - x[0]) / 2 && r.hi >= x[x.length - 1] - (x[1] - x[0]) / 2) r = null;
    if (r?.lo !== main?.lo || r?.hi !== main?.hi) patch(ds.id, { main: r });
  };

  // The frame is kept with the source it came from: right after switching
  // source the previous image must not be used (e.g. for auto contrast).
  const [nav, setNav] = useState<{ source: string; version: string; windowed: boolean; frame: Frame } | null>(null);
  const [navError, setNavError] = useState<string | null>(null);
  // Fetched as a plain image: a navigator image, or the total counts (the
  // virtual image over the whole range), which needs the SI in memory.
  const isAdf = sourceId === ADF;
  const frameId = isVirtual ? (main ? null : ds.id) : sourceId;
  const member = members.find((m) => m.id === sourceId);
  const sourceReady = frameId !== null && frameId !== NONE && (frameId !== ds.id || ready) && !member?.lazy;
  // A member shown as the navigator is loaded first (its total counts need it in memory).
  useEffect(() => {
    if (!member?.lazy) return;
    api.loadDataset(member.id).then(updateDataset).catch((e) => setNavError((e as Error).message));
  }, [member?.id, member?.lazy, updateDataset]); // eslint-disable-line react-hooks/exhaustive-deps
  // The ADF of a series follows the frame held: it has the dataset's version.
  const sourceSummary = allDatasets(files).find((d) => d.id === (isVirtual || isAdf ? ds.id : sourceId));
  const sourceVersion = sourceSummary ? dataVersion(sourceSummary) : "";
  // Contrast is redone when this changes: not for the next frame.
  const sourceContrast = sourceSummary ? contrastVersion(sourceSummary) : "";
  useEffect(() => {
    if (!sourceReady) return;
    const ac = new AbortController();
    const source = sourceId!;
    const version = sourceVersion;
    (isAdf ? fetchInsituAdf(ds.id, ac.signal) : fetchFrame(frameId!, {}, ac.signal))
      .then((frame) => {
        setNav({ source, version, windowed: false, frame });
        setNavError(null);
      })
      .catch((e) => !isAbort(e) && setNavError((e as Error).message));
    return () => ac.abort();
  }, [sourceId, frameId, sourceReady, sourceVersion]); // eslint-disable-line react-hooks/exhaustive-deps

  // The virtual image over a narrower main window follows it live (latest-only
  // while dragging).
  const windowMaps = useRef<Coalescer<{ frame: Frame; version: string }>>();
  windowMaps.current ??= new Coalescer(
    (_c, r) => setNav({ source: VIRTUAL, version: r.version, windowed: true, frame: r.frame }),
    (_c, e) => setNavError(e.message),
  );
  useEffect(() => {
    if (!isVirtual || !main || !ready) {
      windowMaps.current!.forget("map");
      return;
    }
    // Keyed by the version too: the map follows the spectrum image's filters.
    const version = sourceVersion;
    windowMaps.current!.request("map", `${main.lo},${main.hi}@${version}`, () =>
      fetchEnergyMap(ds.id, main.lo, main.hi).then((frame) => ({ frame, version })),
    );
  }, [isVirtual, main?.lo, main?.hi, ready, ds.id, sourceVersion]); // eslint-disable-line react-hooks/exhaustive-deps

  // An in-situ series keeps showing the last frame's image until the next one arrives.
  const keepLast = !!ds.insitu?.view;
  const navFrame =
    nav && nav.source === sourceId && (keepLast || nav.version === sourceVersion) && (!isVirtual || nav.windowed === !!main)
      ? nav.frame
      : null;
  const sourceDataset = sourceId && sourceId !== VIRTUAL ? allDatasets(files).find((d) => d.id === sourceId) : undefined;
  // A navigator finer than the scan (a sub-scan ADF) is shown at its own
  // resolution: k image pixels per spectrum pixel along each axis. The view
  // works in the image's pixels; the cursor, regions and spectra stay in
  // spectrum pixels (see ./si/scan.ts).
  const navSource = sourceId && sourceId !== VIRTUAL && sourceId !== NONE ? sources.find((s) => s.id === sourceId) : undefined;
  const scanK = navSource?.factor ?? 1;
  const imageDataset = navSource && scanK > 1 ? allDatasets(files).find((d) => d.id === navSource.dataset_id) : undefined;
  const navGeometry = navigatorGeometry(g, scanK, imageDataset && imageGeometry(imageDataset, "signal"));
  // A navigator image shares its display settings (contrast, element colour)
  // with the dataset itself; the virtual image has its own.
  const navKey = isVirtual ? `${ds.id}:total` : isAdf ? `${ds.id}:adf` : (sourceId ?? "");

  // The virtual image is auto-contrasted whenever its window changes (it
  // stays greyscale); the whole range keeps the viewer's own contrast.
  const windowed = useRef(false);
  useEffect(() => {
    if (!isVirtual) windowed.current = false;
    if (!isVirtual || !navFrame) return;
    if (!main && !windowed.current) return;
    windowed.current = !!main;
    const [lo, hi] = autoRange(sortedSample(navFrame.data));
    const { display, setDisplay, initDisplay } = useDisplay.getState();
    if (!display[navKey]) initDisplay(navKey, { lo, hi, gamma: 1, colormap: "gray" });
    else setDisplay(navKey, { lo, hi });
  }, [navFrame]); // eslint-disable-line react-hooks/exhaustive-deps

  const virtualName = linked ? `Virtual image · ${memberLabel(ds, group)}` : "Virtual image";
  const virtualLabel = main ? `${virtualName} (${rangeText(main)})` : virtualName;

  // A line scan along a profile line (drawn in the navigator's pixels: a
  // finer navigator has scanK of them per spectrum pixel), and along the
  // same line of the spectrum images linked to it.
  const cutLineScan = async (l: ProfileLine) => {
    const { replaceFile } = useDatasets.getState();
    const k = scanK;
    try {
      const r = await api.cutLineScan(
        ds.id,
        { x0: l.x0 / k, y0: l.y0 / k, x1: l.x1 / k, y1: l.y1 / k, width: Math.max(1, l.width / k) },
        l.name,
      );
      replaceFile(r.file);
      r.partner_files.forEach(replaceFile);
      // Kept with the line: deleting the line deletes them.
      const line = getViewProfiles(navView).lines.find((x) => x.id === l.id);
      if (line) {
        const made = [r.dataset.id, ...r.partners.map((p) => p.id)];
        useProfiles.getState().updateLine(navView, l.id, { lineScans: [...(line.lineScans ?? []), ...made] });
      }
      useLayers.getState().notify(`${r.dataset.title} added to the dataset pane.`);
    } catch (e) {
      useLayers.getState().notify(`Could not make the LineScan: ${(e as Error).message}`);
    }
  };

  // In-situ series: right-clicking the image offers time scans (line scans
  // over the frames) at the cursor and of the regions under the pointer.
  const insituJob = useInsitu((s) => s.jobs[ds.id]);
  const timeScanMenu = (p: Pt): MenuItem[] => {
    const s = getSiState(ds.id);
    const at = { x: p.x / scanK, y: p.y / scanK }; // scan pixels
    const busy = reducing(insituJob);
    const title = busy ? "Wait for the frames being read" : "A LineScan over the frames: in each, the mean spectrum of these pixels";
    const scan = (source: Parameters<typeof api.timeScan>[1], name: string) =>
      void useInsitu.getState().timeScan(ds.id, source, name);
    const items: MenuItem[] = [];
    if (s.cursor && s.showCursor) {
      const { x, y } = s.cursor;
      const size = cursorSize;
      items.push({
        label: `Generate LineScan over time at the cursor (${x}, ${y}${size > 1 ? `, ${size}×${size}` : ""})`,
        title, disabled: busy,
        onSelect: () => scan({ pixel: { x, y, size } }, `cursor (${x}, ${y})`),
      });
    }
    for (const r of s.rois.filter((r) => r.visible && containsPoint(r.shape, at.x, at.y))) {
      items.push({
        label: `Generate LineScan over time of ${r.name}`, title, disabled: busy,
        onSelect: () => scan({ region: shapeToRegion(r.shape) }, r.name),
      });
    }
    return items;
  };

  // ---------- spectra ----------
  const whole = useSpectrum(ds, ready);
  // A point or stretch picked on a line's profile: its spectrum (the band
  // across the line's width; a point covers one pixel along it).
  // Lines are drawn in the navigator's pixels: a finer navigator has its own.
  const navView = scanK > 1 ? `${ds.id}:nav:x${scanK}` : `${ds.id}:nav`;
  // Annotations hidden with the eye: picking a drawing tool shows them again.
  const annotationsHidden = useUi((s) => !!s.annotationsHidden[navView]);
  useEffect(() => {
    if (tool !== "cursor") useUi.getState().setAnnotationsHidden(navView, false);
  }, [tool, navView]);
  const profiles = useViewProfiles(navView);
  const pickLine = profiles.pick && profiles.lines.find((l) => l.id === profiles.pick!.line && l.visible);
  const lineBand = useMemo<LineBand | null>(() => {
    if (!pickLine || !profiles.pick) return null;
    let { t0, t1 } = profiles.pick;
    // A picked point and the line's width cover at least one spectrum pixel (scanK image pixels).
    const half = (0.5 * scanK) / Math.max(scanK, pixelLength(pickLine));
    if (t1 - t0 < 2 * half) [t0, t1] = [(t0 + t1) / 2 - half, (t0 + t1) / 2 + half];
    const [x, y] = pointAt(pickLine, (t0 + t1) / 2);
    const clamp = (v: number, n: number) => Math.min(n - 1, Math.max(0, scanPixel(v, scanK)));
    return {
      points: polygonToScan(bandPolygon(pickLine, Math.max(scanK, pickLine.width), t0, t1), scanK),
      pixel: [clamp(x, g.width), clamp(y, g.height)],
    };
  }, [pickLine, profiles.pick, g.width, g.height, scanK]);
  const spectra = useSiSpectra(ds.id, dataVersion(ds), st, cursorSize, ready, lineBand);
  const totalPixels = g.width * g.height;
  const fitCursor = useMemo(
    () => (st.showCursor && st.cursor ? { x: st.cursor.x, y: st.cursor.y, size: cursorSize } : null),
    [st.showCursor, st.cursor, cursorSize],
  );
  const fitSeries = useFitSeries(ds, axis.x, reduce === "mean", fitCursor, st);
  // Each series with the pixels it sums (for the deconvolution preview; none for fits).
  const { series, sel } = useMemo(() => {
    const k = (n: number) => (reduce === "mean" ? 1 / n : 1);
    const out: SpectrumSeries[] = [];
    const sel: (DeconvolutionSelection | null)[] = [];
    if (st.showWhole && whole.data) {
      out.push({ label: "All pixels", data: scaled(whole.data, k(totalPixels)), color: SPECTRUM_COLOR });
      sel.push({ kind: "whole" });
    }
    for (const r of st.rois) {
      const s: PixelsSpectrum | undefined = spectra[r.id];
      if (r.visible && s && s.pixels > 0) {
        out.push({ label: r.name, data: scaled(s.data, k(s.pixels)), color: r.color });
        sel.push({ kind: "region", region: shapeToRegion(r.shape) });
      }
    }
    const ls = spectra[LINE_CHANNEL];
    if (pickLine && lineBand && ls) {
      out.push({ label: `${pickLine.name} (profile)`, data: scaled(ls.data, k(ls.pixels)), color: pickLine.color });
      sel.push(null);
    }
    const c = spectra[CURSOR_CHANNEL];
    if (st.showCursor && st.cursor && c) {
      out.push({ label: "Cursor", data: scaled(c.data, k(c.pixels)), color: CURSOR_COLOR });
      sel.push({ kind: "pixel", x: st.cursor.x, y: st.cursor.y, size: cursorSize });
    }
    return { series: [...out, ...fitSeries], sel };
  }, [st.showWhole, st.showCursor, st.cursor, st.rois, whole.data, spectra, reduce, totalPixels, fitSeries, pickLine, lineBand, cursorSize]);

  // ---------- elements, line markers and energy windows ----------
  const [preview, setPreview] = useState<string | null>(null);
  const markers = useMarkers(ds, preview, axis.x);

  // Windows drawn on the spectrum: the main one (grey, the virtual image's)
  // and the window maps in the view, each in its map's colour. Selecting one
  // makes its layer the one edited (and the other way round).
  const displays = useDisplay((s) => s.display);
  const windowColours = useDisplay((s) => s.windowColours);
  const layerProps = useLayers((s) => s.props);
  const showMain = useSi((s) => s.showMain);
  const [edit, setEdit] = useState<{ id: string; lo: number; hi: number } | null>(null); // a window map being dragged
  const [draft, setDraft] = useState<EnergyRange | null>(null); // a window being drawn
  const [activate, setActivate] = useState<{ key: string } | null>(null);
  const [menu, setMenu] = useState<{ x: number; y: number; id: string } | null>(null);
  const plotWindows = useMemo<PlotWindow[]>(() => {
    const solid = (c: string | undefined) => (c && isColour(c) ? c : NEUTRAL_WINDOW);
    const out: PlotWindow[] = [];
    const x = axis.x;
    if (isVirtual && showMain && x.length && layerProps[baseLayerKey(navView)]?.visible !== false) {
      out.push({ id: MAIN, lo: main?.lo ?? x[0], hi: main?.hi ?? x[x.length - 1], color: MAIN_COLOUR, label: virtualLabel });
    }
    for (const d of allDatasets(files)) {
      if (d.origin !== "window" || d.parent_id !== ds.id || !d.window) continue;
      const asBase = d.id === sourceId;
      if (!asBase && !images.some((i) => i.id === d.id)) continue;
      if (layerProps[asBase ? baseLayerKey(navView) : d.id]?.visible === false) continue;
      const cm = displays[d.id]?.colormap;
      const w = edit?.id === d.id ? edit : d.window;
      out.push({ id: d.id, lo: w.lo, hi: w.hi, color: cm ? solid(cm) : (windowColours[d.id] ?? NEUTRAL_WINDOW), label: d.title });
    }
    if (draft) out.push({ id: DRAFT, ...draft, color: useDisplay.getState().nextWindowColour(), label: "", draft: true });
    return out;
  }, [files, images, sourceId, isVirtual, main, axis.x, displays, windowColours, layerProps, edit, draft, navView, ds.id, virtualLabel, showMain]);
  // Background removal (Process tab): its fits and bands on this plot.
  const bgPixels = useMemo(() => sel.map((s) => (s?.kind === "pixel" ? { x: s.x, y: s.y, size: s.size } : null)), [sel]);
  const bg = useBackgroundPreview(ds, axis.x, series, bgPixels);
  // Deconvolution (Process tab): the deconvolved spectra, dashed.
  const plotted = useDeconvPreview(ds, bg.series, sel, reduce === "mean");
  // Thickness (Process tab): the zero-loss threshold on a low loss.
  const zlp = useThresholdLine(ds);
  const allWindows = useMemo(() => [...plotWindows, ...bg.bands, ...zlp.windows], [plotWindows, bg.bands, zlp.windows]);
  const windowMapIds = plotWindows.filter((w) => w.id !== MAIN && w.id !== DRAFT).map((w) => w.id);
  const selectedWindow = plotWindows.some((w) => w.id === st.selectedWindow) ? st.selectedWindow : null;
  /** Selects a window; a window map also deselects the region (Delete acts on one at a time). */
  const pickWindow = (id: string | null) => {
    if (id === MAIN && !useSi.getState().showMain) id = null; // its band is hidden
    const s = getSiState(ds.id);
    if (s.selectedWindow !== id || (id && id !== MAIN && s.selected)) {
      patch(ds.id, { selectedWindow: id, ...(id && id !== MAIN && s.selected && { selected: null }) });
    }
  };
  const selectWindow = (id: string | null) => {
    pickWindow(id);
    const key = id === MAIN ? (isVirtual ? BASE_LAYER : null) : id === sourceId ? BASE_LAYER : id;
    if (key) setActivate({ key });
  };
  // The layer being edited changed: select its window, if it has one.
  const onActiveLayer = (key: string) => {
    const id = key === BASE_LAYER ? (isVirtual ? MAIN : sourceId) : key;
    const isWindow =
      id === MAIN || allDatasets(files).some((d) => d.id === id && d.origin === "window" && d.parent_id === ds.id);
    pickWindow(isWindow ? id : null);
  };
  // Selecting a region leaves the window selection (to the main one).
  useEffect(() => {
    const s = getSiState(ds.id);
    if (s.selected && s.selectedWindow && s.selectedWindow !== MAIN) {
      patch(ds.id, { selectedWindow: isVirtual && useSi.getState().showMain ? MAIN : null });
    }
  }, [st.selected]); // eslint-disable-line react-hooks/exhaustive-deps
  // Hiding the main band deselects it.
  useEffect(() => {
    if (!showMain && getSiState(ds.id).selectedWindow === MAIN) patch(ds.id, { selectedWindow: null });
  }, [showMain]); // eslint-disable-line react-hooks/exhaustive-deps
  // Shift-drag: the new window shows as a provisional band; on release it
  // becomes a window map, added to the view and selected.
  const drawWindow = async (range: EnergyRange, final: boolean) => {
    const r = { lo: Math.min(range.lo, range.hi), hi: Math.max(range.lo, range.hi) };
    setDraft(r);
    if (!final || !ready) return;
    try {
      const res = await api.addWindowMap(ds.id, r.lo, r.hi, null);
      const { replaceFile, addToSelection } = useDatasets.getState();
      replaceFile(res.file);
      const made = allDatasets(useDatasets.getState().files).find((d) => d.id === res.dataset.id) ?? res.dataset;
      addToSelection(made);
      pickWindow(made.id);
    } catch (e) {
      useLayers.getState().notify(`Could not make the window: ${(e as Error).message}`);
    } finally {
      setDraft(null);
    }
  };
  // Dragging the selected window's edge or inside: the main window follows
  // live; a window map's band follows the pointer and its map is remade on release.
  const editWindow = async (id: string, range: EnergyRange, final: boolean) => {
    if (bg.onEditBand(id, range, final) || zlp.onEdit(id, range, final)) return;
    if (id === MAIN) return setMain(range);
    setEdit({ id, ...range });
    if (!final) return;
    try {
      const colour = useDisplay.getState().display[id]?.colormap;
      const r = await api.setWindowMap(id, range.lo, range.hi);
      // The remade map starts afresh, but keeps the colour the user gave it.
      if (colour && isColour(colour)) useDisplay.setState((s) => ({ windowColours: { ...s.windowColours, [id]: colour } }));
      useDatasets.getState().replaceFile(r.file);
    } catch (e) {
      useLayers.getState().notify(`Could not move the window: ${(e as Error).message}`);
    } finally {
      setEdit(null);
    }
  };
  // Delete windows (window maps; the main window stays).
  const removeWindows = async (ids: string[]) => {
    for (const id of ids) {
      const d = allDatasets(useDatasets.getState().files).find((x) => x.id === id);
      if (d) await useDatasets.getState().deleteDataset(d);
    }
    pickWindow(isVirtual ? MAIN : null);
  };
  const deleteSelectedWindow = (e: KeyboardEvent) => {
    // A focused dataset-list row handles Delete itself.
    if (e.target instanceof Element && e.target.closest("[role=button][aria-pressed]")) return false;
    const id = getSiState(ds.id).selectedWindow;
    if (!id || id === MAIN || !windowMapIds.includes(id) || !useSi.getState().showWindows) return false;
    void removeWindows([id]);
    return true;
  };
  useShortcuts({ Delete: deleteSelectedWindow, Backspace: deleteSelectedWindow });
  const onWindowMenu = (id: string, x: number, y: number) => {
    if (id === MAIN || !windowMapIds.includes(id)) return false;
    selectWindow(id);
    setMenu({ x, y, id });
    return true;
  };
  // Double-click on the main band: back to the whole range.
  const onWindowDoubleClick = (id: string) => {
    if (id !== MAIN || !main) return false;
    setMain(null);
    return true;
  };
  const onMarkerClick = (m: Marker) => {
    const fw = fwhmAt(m.energy, ds.eds?.energy_resolution_MnKa ?? 130);
    setMain({ lo: Math.max(m.energy - fw, axis.x[0]), hi: Math.min(m.energy + fw, axis.x[axis.x.length - 1]) });
    setSourceId(VIRTUAL);
    useSi.getState().setShowMain(true); // so the new range is seen
    patch(ds.id, { selectedWindow: MAIN });
    setActivate({ key: BASE_LAYER });
  };

  const ownLog = useLossLog(ds.eels?.role, linked);
  const loadingMsg = `Loading spectrum image into memory (${bytesLabel(ds.nbytes)})…`;

  const legend = (
    <SeriesLegend si={ds.id} state={st} spectra={spectra} totalPixels={totalPixels} wholeColor={SPECTRUM_COLOR} />
  );
  const spectrumPlot = (
    <SpectrumPlot
      name={`${ds.title}_spectra`}
      zoomKey={`${ds.id}:spectra`}
      {...ownLog}
      x={axis.x}
      series={plotted}
      xLabel={axis.label}
      xUnits={axis.units}
      loading={whole.data ? null : ready ? "Summing spectra…" : loadingMsg}
      error={whole.error ?? loadError}
      markers={markers}
      windows={allWindows}
      mainToggle={isVirtual}
      selectedId={selectedWindow}
      onSelectWindow={selectWindow}
      onWindow={drawWindow}
      onEditWindow={editWindow}
      onWindowMenu={onWindowMenu}
      onWindowDoubleClick={onWindowDoubleClick}
      onMarkerClick={ds.eds ? onMarkerClick : undefined}
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
      legend={legend}
    />
  );

  return (
    <div className="view si-view">
      <section className="si-pane">
        <div className="si-pane-header">
          <label className="field">
            Navigator
            <select value={sourceId ?? ""} onChange={(e) => setSourceId(e.target.value)}>
              {sources.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
              <option value={NONE}>None (maps only)</option>
            </select>
          </label>
          <div className="tool-row">
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
            <AnnotationsToggle view={navView} />
          </div>
        </div>
        <LayeredViewer
          viewKey={navView}
          lineTool={tool === "line"}
          fftTool={tool === "fft"}
          profilePick
          onLineScan={cutLineScan}
          imageMenu={ds.insitu?.view ? timeScanMenu : undefined}
          gridContext="si"
          base={
            sourceId === NONE
              ? null
              : {
                  key: navKey,
                  label: isVirtual ? virtualLabel : (sources.find((s) => s.id === sourceId)?.label ?? "Navigator"),
                  frame: navFrame,
                  dataset: sourceDataset,
                  version: sourceContrast,
                }
          }
          images={images.filter((d) => d.id !== sourceId)}
          geometry={navGeometry}
          scanFactor={scanK}
          loading={
            navFrame || sourceId === NONE
              ? null
              : isVirtual && !ready
                ? loadingMsg
                : "Loading navigator…"
          }
          error={navError ?? loadError}
          empty="No navigator image. Click element maps in the dataset pane to show them here."
          requestActive={activate}
          onActiveChange={onActiveLayer}
          interaction={scanInteraction(tools.interaction, scanK)}
          renderOverlay={(t) => (
            <RoiOverlay
              t={scanTransform(t, scanK)}
              state={st}
              cursorSize={cursorSize}
              draft={tools.draft}
              hover={tools.hover}
              hideRois={annotationsHidden}
            />
          )}
          bottomTabs={memberTabs(ds, group, spectrumPlot, st, false, legend)}
          footer={ds.insitu?.view ? <InsituFrameBar ds={ds} /> : undefined}
        >
          {ds.insitu?.view && <span className="frame-chip">{insituViewLabel(ds.insitu)}</span>}
        </LayeredViewer>
      </section>
      {menu && (
        <ContextMenu
          x={menu.x}
          y={menu.y}
          onClose={() => setMenu(null)}
          items={[
            { label: "Delete window", danger: true, onSelect: () => removeWindows([menu.id]) },
            ...(windowMapIds.length > 1
              ? [{ label: "Delete all windows", danger: true, onSelect: () => removeWindows(windowMapIds) }]
              : []),
          ]}
        />
      )}
    </div>
  );
}
