import { useEffect, useMemo, useRef, useState } from "react";
import { fetchFrame, type Frame } from "../api/arrays";
import { api, dataVersion, type DatasetSummary, type DeconvolutionSelection } from "../api/client";
import { ContrastControls } from "../components/ContrastControls";
import { ElementsPicker } from "../components/ElementsPicker";
import { LineScanImage, type ScanAxis } from "../components/LineScanImage";
import { BottomPane } from "../components/BottomPane";
import { HistogramPanel } from "../components/HistogramPanel";
import { ExplodedPanel, marksKey, type EnergyMark } from "../components/ExplodedPanel";
import { SpectrumPlot, type SpectrumSeries } from "../components/SpectrumPlot";
import { formatValue } from "../format";
import { useShortcuts } from "../keyboard";
import { useDatasets } from "../state/datasets";
import { useDisplay } from "../state/display";
import { CURSOR_COLOR, getSiState, shapeToRegion, useSi, useSiState } from "../state/si";
import { eelsTab, useUi } from "../state/ui";
import { isAbort, spectralAxis } from "./geometry";
import { CURSOR_CHANNEL, useSiSpectra } from "./si/useSiSpectra";
import { useBackgroundPreview } from "./si/useBackgroundPreview";
import { useDeconvPreview } from "./si/useDeconvPreview";
import { useThresholdLine } from "./si/useThresholdLine";
import { SPECTRUM_COLOR, useEnergyAxis, useSpectrum } from "./SpectrumView";
import { ReduceToggle, memberLabel, memberTabs, useGroup, useLossLog } from "./si/LinkedSpectrum";
import { useFitSeries } from "./useFitSeries";
import { useImageDisplay } from "./useImageDisplay";
import { useMarkers } from "./si/xray";
import { NormaliseToggle, useNormBand } from "../components/NormaliseSection";
import { useCurveFitSeries } from "./useCurveFitSeries";
import { useEelsBgBand } from "./si/useEelsBgBand";
import { normaliseSeries, useNormSettings } from "../state/normalise";

const scaled = (data: Float32Array, k: number) => (k === 1 ? data : data.map((v) => v * k));

/** log10 of the positive values; zeros and negatives take the smallest. */
export function logFrame(f: Frame): Frame {
  const data = f.data.map((v) => (v > 0 ? Math.log10(v) : NaN));
  let floor = Infinity;
  for (const v of data) if (v < floor) floor = v;
  return { ...f, data: data.map((v) => (Number.isNaN(v) ? (Number.isFinite(floor) ? floor : 0) : v)) };
}

const axisOf = (a: DatasetSummary["axes"][number] | undefined, fallback: string): ScanAxis => ({
  label: a?.name || fallback,
  units: a?.units ?? "",
  size: a?.size ?? 0,
  scale: a?.scale ?? 1,
  offset: a?.offset ?? 0,
});

/**
 * A line scan (a spectrum image with one navigation axis): the image of
 * energy against position above, the spectra below. Dragging the cursor line
 * moves the cursor spectrum (hovering only reads out; arrow keys nudge it);
 * Shift+drag makes a range of positions to average, any number of them, each
 * in its own colour. Ranges are edited by dragging their edges or inside,
 * deleted with Delete or the right-click menu.
 *
 * The cursor and the ranges live in the SI interaction state, as for a 2D
 * scan, with positions along x (the cursor at (i, 0), a range a rectangle
 * {x0: i0, x1: i1 + 1, y0: 0, y1: 1}, which the backend sums over i0…i1).
 */
export function LineScanView({ ds }: { ds: DatasetSummary }) {
  const updateDataset = useDatasets((s) => s.updateDataset);
  const [loadError, setLoadError] = useState<string | null>(null);
  const ready = !ds.lazy;
  const version = dataVersion(ds);
  const n = ds.navigation_shape[0] ?? 0;

  // Spectrum images are loaded into RAM up front: everything else is fast after.
  useEffect(() => {
    if (!ds.lazy) return;
    api
      .loadDataset(ds.id)
      .then(updateDataset)
      .catch((e) => setLoadError((e as Error).message));
  }, [ds.id, ds.lazy, updateDataset]);

  // ---------- cursor and selection ----------
  const st = useSiState(ds.id);
  const { patch, addRoi, updateRoi, deleteRois } = useSi.getState();
  useEffect(() => {
    // Start with the cursor in the middle so a spectrum shows at once.
    if (!getSiState(ds.id).cursor && n) patch(ds.id, { cursor: { x: Math.floor(n / 2), y: 0 } });
  }, [ds.id, n, patch]);
  const reduceMode = useSi((s) => s.reduce);

  /** Ranges are rectangles over positions; the first/last position they sum. */
  const ranges = useMemo(
    () =>
      st.rois.flatMap((r) =>
        r.shape.kind === "polygon" ? [] : [{ roi: r, lo: r.shape.x0, hi: r.shape.x1 - 1 }],
      ),
    [st.rois],
  );
  const rangeShape = (lo: number, hi: number) => ({ kind: "rect" as const, x0: lo, x1: hi + 1, y0: 0, y1: 1 });
  const setCursor = (row: number) => {
    const cur = getSiState(ds.id).cursor;
    if (!cur || cur.x !== row || cur.y !== 0) patch(ds.id, { cursor: { x: row, y: 0 }, showCursor: true });
  };
  const addRange = (lo: number, hi: number) => {
    const rois = getSiState(ds.id).rois;
    const num = rois.reduce((m, r) => Math.max(m, Number(r.name.match(/\d+$/)?.[0] ?? 0)), 0) + 1;
    const id = addRoi(ds.id, rangeShape(lo, hi));
    updateRoi(ds.id, id, { name: `Range ${num}` });
    return id;
  };
  const changeRange = (id: string, lo: number, hi: number) => updateRoi(ds.id, id, { shape: rangeShape(lo, hi) });
  const removeRange = (id: string) => deleteRois(ds.id, [id]);
  const clearRanges = () => deleteRois(ds.id, getSiState(ds.id).rois.map((r) => r.id));
  const hasSelected = () => {
    const s = getSiState(ds.id);
    return s.selected !== null && s.rois.some((r) => r.id === s.selected);
  };
  useShortcuts({
    Delete: () => hasSelected() && removeRange(getSiState(ds.id).selected!),
    Backspace: () => hasSelected() && removeRange(getSiState(ds.id).selected!),
    Escape: () => hasSelected() && patch(ds.id, { selected: null }),
  });

  // ---------- the image ----------
  // A linked group shows the image of the member whose spectrum tab is open;
  // the frames are kept, so switching tabs back does not fetch again.
  const group = useGroup(ds);
  const linked = !!ds.eels && group.length > 1;
  const bottomTab = useUi((s) => s.bottomTab);
  const other = group.find((m) => m.id !== ds.id && eelsTab(m.id) === bottomTab);
  const otherId = other?.id ?? null;
  const otherReady = !!other && !other.lazy;
  const otherVersion = other ? dataVersion(other) : "";

  const [ownShown, setOwnShown] = useState<{ version: string; frame: Frame } | null>(null);
  const [frameError, setFrameError] = useState<string | null>(null);
  useEffect(() => {
    if (!ready) return;
    const ac = new AbortController();
    setFrameError(null);
    fetchFrame(ds.id, {}, ac.signal)
      .then((frame) => setOwnShown({ version, frame }))
      .catch((e) => !isAbort(e) && setFrameError((e as Error).message));
    return () => ac.abort();
  }, [ds.id, ready, version]);

  const [otherFrames, setOtherFrames] = useState<Record<string, { version: string; frame: Frame }>>({});
  const [otherError, setOtherError] = useState<string | null>(null);
  const otherFrame = otherId ? otherFrames[otherId] : undefined;
  const otherCached = otherFrame?.version === otherVersion;
  useEffect(() => {
    if (!otherId || !otherReady || otherCached) return;
    const ac = new AbortController();
    setOtherError(null);
    fetchFrame(otherId, {}, ac.signal)
      .then((frame) => setOtherFrames((f) => ({ ...f, [otherId]: { version: otherVersion, frame } })))
      .catch((e) => !isAbort(e) && setOtherError((e as Error).message));
    return () => ac.abort();
  }, [otherId, otherReady, otherVersion, otherCached]);

  const imgDs = other ?? ds;
  const imgNorm = useNormSettings(imgDs);
  const shown = other ? (otherCached ? otherFrame! : null) : ownShown;
  const imgReady = other ? otherReady : ready;

  // Linear/Log is kept per dataset; the low-loss image starts in Log (the zero-loss peak dominates).
  const storedLog = useDisplay((s) => s.imageLog[imgDs.id]);
  const setImageLog = useDisplay((s) => s.setImageLog);
  const log = storedLog ?? (linked && imgDs.eels?.role === "low_loss");
  const drawn = useMemo(() => shown && (log ? logFrame(shown.frame) : shown.frame), [shown, log]);
  const display = useImageDisplay(`${imgDs.id}:linescan${log ? ":log" : ""}`, drawn, undefined, shown?.version);

  const rows = axisOf(ds.axes.find((a) => a.navigate), "Position");
  const cols = axisOf(spectralAxis(imgDs), "Energy");

  // ---------- spectra ----------
  const axis = useEnergyAxis(ds);
  const whole = useSpectrum(ds, ready);
  const spectra = useSiSpectra(ds.id, version, st, 1, ready);
  // Each series with the positions it sums (for the deconvolution preview).
  const { series, sel } = useMemo(() => {
    const k = (count: number) => (reduceMode === "mean" ? 1 / count : 1);
    const out: SpectrumSeries[] = [];
    const sel: DeconvolutionSelection[] = [];
    if (st.showWhole && whole.data) {
      out.push({ label: "All positions", data: scaled(whole.data, k(n)), color: SPECTRUM_COLOR });
      sel.push({ kind: "whole" });
    }
    for (const { roi, lo, hi } of ranges) {
      const s = roi.visible ? spectra[roi.id] : undefined;
      if (s) {
        out.push({ label: roi.name, data: scaled(s.data, k(hi - lo + 1)), color: roi.color });
        sel.push({ kind: "region", region: shapeToRegion(roi.shape) });
      }
    }
    const c = spectra[CURSOR_CHANNEL];
    if (st.showCursor && st.cursor && c) {
      out.push({ label: "Cursor", data: c.data, color: CURSOR_COLOR });
      sel.push({ kind: "pixel", x: st.cursor.x, y: st.cursor.y, size: 1 });
    }
    return { series: out, sel };
  }, [st.showWhole, st.showCursor, st.cursor, ranges, whole.data, spectra, reduceMode, n]); // eslint-disable-line react-hooks/exhaustive-deps

  // Background removal (Process tab): its fits and bands on this plot.
  const bgPixels = useMemo(() => sel.map((s) => (s?.kind === "pixel" ? { x: s.x, y: s.y, size: s.size } : null)), [sel]);
  const bg = useBackgroundPreview(ds, axis.x, series, bgPixels);
  // Deconvolution (Process tab): the deconvolved spectra, dashed.
  const plotted = useDeconvPreview(ds, bg.series, sel, reduceMode === "mean");
  // Thickness (Process tab): the zero-loss threshold on a low loss.
  const zlp = useThresholdLine(ds);
  const normBand = useNormBand(ds);
  const curve = useCurveFitSeries(ds, axis.x, reduceMode === "mean");
  const eelsBg = useEelsBgBand(ds);
  const bands = useMemo(
    () => [...bg.bands, ...zlp.windows, ...normBand.bands, ...curve.bands, ...eelsBg.bands],
    [bg.bands, zlp.windows, normBand.bands, curve.bands, eelsBg.bands],
  );
  // The model fit (Quantify tab) and the elements' edges.
  const fitSeries = useFitSeries(ds, axis.x, reduceMode === "mean", null, st);
  // Normalised (Process tab, the plot's toggle): the spectra alone, each normalised.
  const norm = useNormSettings(ds);
  const withFit = useMemo(
    () => (norm.on ? normaliseSeries(series, axis.x, norm) : [...plotted, ...fitSeries, ...curve.series]),
    [norm, series, axis.x, plotted, fitSeries, curve.series],
  );
  const [preview, setPreview] = useState<string | null>(null);
  const markers = useMarkers(ds, preview, axis.x);

  const position = (i: number) => `${i}${rows.units ? ` · ${formatValue(rows.offset + i * rows.scale)} ${rows.units}` : ""}`;
  const legend = (
    <div className="series-legend">
      <label className={`series-chip${st.showWhole ? "" : " off"}`} title="Spectrum summed over every position">
        <input type="checkbox" checked={st.showWhole} onChange={(e) => patch(ds.id, { showWhole: e.target.checked })} />
        <span className="series-swatch" style={{ background: SPECTRUM_COLOR }} />
        All positions
      </label>
      <label className={`series-chip${st.showCursor ? "" : " off"}`} title="Spectrum at the cursor row">
        <input type="checkbox" checked={st.showCursor} onChange={(e) => patch(ds.id, { showCursor: e.target.checked })} />
        <span className="series-swatch" style={{ background: CURSOR_COLOR }} />
        Cursor <span className="muted">{st.cursor ? position(st.cursor.x) : "–"}</span>
      </label>
      {ranges.map(({ roi, lo, hi }) => (
        <span
          key={roi.id}
          className={`series-chip roi-chip${roi.visible ? "" : " off"}${st.selected === roi.id ? " selected" : ""}`}
          onClick={() => patch(ds.id, { selected: roi.id })}
          title="Click to select (Delete to remove)"
        >
          <input
            type="checkbox"
            checked={roi.visible}
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => updateRoi(ds.id, roi.id, { visible: e.target.checked })}
            aria-label={`Show ${roi.name}`}
          />
          <span className="series-swatch" style={{ background: roi.color }} />
          {roi.name}
          <span className="muted">
            {lo === hi ? lo : `${lo}–${hi}`}
          </span>
          <button
            className="icon-button"
            title={`Remove ${roi.name}`}
            onClick={(e) => {
              e.stopPropagation();
              removeRange(roi.id);
            }}
          >
            ×
          </button>
        </span>
      ))}
    </div>
  );

  const ownLog = useLossLog(ds.eels?.role, linked);
  const spectrumPlot = (
    <SpectrumPlot
      {...curve.plot}
      name={`${ds.title}_spectra`}
      zoomKey={`${ds.id}:spectra`}
      {...ownLog}
      x={axis.x}
      series={withFit}
      markers={markers}
      windows={bands}
      onEditWindow={(id, range, final) =>
        normBand.onEdit(id, range) || curve.onEdit(id, range) || eelsBg.onEdit(id, range) || zlp.onEdit(id, range, final) || bg.onEditBand(id, range, final)}
      xLabel={axis.label}
      xUnits={axis.units}
      loading={whole.data ? null : ready ? "Summing spectra…" : "Loading LineScan into memory…"}
      error={whole.error ?? loadError}
      toolbar={
        <>
          <ElementsPicker ds={ds} onPreview={setPreview} />
          <ReduceToggle unit="position" />
          <NormaliseToggle d={ds} />
        </>
      }
      legend={legend}
    />
  );

  // A linked group has a tab for each member (T cycles them).
  // The exploded view (right-click the image): the spectra of the image shown, as a cascade.
  const [exploded, setExploded] = useState(false);
  // Lines and windows on the energy axis made in the exploded view (also drawn on the image).
  const [marks, setMarks] = useState<EnergyMark[]>([]);
  const [markSel, setMarkSel] = useState<string | null>(null);
  useEffect(() => {
    setExploded(false);
    setMarks([]);
    setMarkSel(null);
  }, [ds.id]);
  const changeMark = (m: EnergyMark) => setMarks((ms) => ms.map((x) => (x.id === m.id ? m : x)));
  const marksHidden = useUi((s) => !!s.annotationsHidden[marksKey(ds.id)]);
  // Delete removes the selected mark, Esc deselects it (a selected range goes first).
  const deleteMark = () => {
    if (!markSel) return false;
    setMarks((ms) => ms.filter((m) => m.id !== markSel));
    setMarkSel(null);
  };
  useShortcuts({
    Delete: deleteMark,
    Backspace: deleteMark,
    Escape: () => (markSel ? setMarkSel(null) : false),
  });
  const setBottomTab = useUi((s) => s.setBottomTab);
  const openExploded = () => {
    setExploded(true);
    setBottomTab("exploded");
  };
  const tabs = [
    ...memberTabs(ds, group, spectrumPlot, st, true, legend),
    // The values of the image shown (log-scaled when it is shown in Log).
    { id: "histogram" as const, label: "Histogram", content: <HistogramPanel frame={drawn} label={imgDs.title} display={display.display} /> },
    ...(exploded
      ? [{
          id: "exploded" as const,
          label: "Exploded",
          content: (
            <ExplodedPanel
              frame={shown?.frame ?? null}
              rows={rows}
              cols={cols}
              label={imgDs.title}
              norm={imgNorm}
              normToggle={<NormaliseToggle d={imgDs} />}
              marksView={marksKey(ds.id)}
              marks={marks}
              onMarksChange={setMarks}
              selected={markSel}
              onSelect={setMarkSel}
              onClose={() => {
                setExploded(false);
                setMarks([]);
                setBottomTab(linked ? eelsTab(ds.id) : "spectrum");
              }}
            />
          ),
        }]
      : []),
  ];

  // Image above, spectrum below, with the same draggable split as the other views.
  const viewerRef = useRef<HTMLDivElement>(null);

  return (
    <div className="view si-view">
      <section className="si-pane">
        <div ref={viewerRef} className="layered-viewer">
          <div className="viewer-top">
            <div className="layer-toolbar">
              <ContrastControls frame={drawn} display={display.display} onChange={display.setDisplay} />
              <div className="segmented" role="group" aria-label="Intensity scale">
                <button className={!log ? "active" : undefined} aria-pressed={!log} onClick={() => setImageLog(imgDs.id, false)}>
                  Linear
                </button>
                <button
                  className={log ? "active" : undefined}
                  aria-pressed={log}
                  onClick={() => setImageLog(imgDs.id, true)}
                  title="Logarithm of the counts (zeros take the lowest value)"
                >
                  Log
                </button>
              </div>
              {linked && <span className="muted">{memberLabel(imgDs, group)}</span>}
            </div>
            <LineScanImage
              frame={drawn}
              values={shown?.frame ?? null}
              display={display.display}
              rows={rows}
              cols={cols}
              cursor={st.cursor && st.showCursor ? st.cursor.x : null}
              cursorColor={CURSOR_COLOR}
              ranges={ranges
                .filter((r) => r.roi.visible)
                .map((r) => ({ id: r.roi.id, lo: r.lo, hi: r.hi, color: r.roi.color, selected: st.selected === r.roi.id }))}
              loading={drawn ? null : imgReady ? "Loading LineScan…" : "Loading LineScan into memory…"}
              error={(other ? otherError : frameError) ?? loadError}
              onCursor={setCursor}
              onRangeAdd={addRange}
              onRangeChange={changeRange}
              onSelect={(id) => patch(ds.id, { selected: id })}
              onRangeRemove={removeRange}
              onRangesClear={clearRanges}
              energyMarks={marksHidden ? [] : marks}
              selectedMark={markSel}
              onMarkSelect={setMarkSel}
              onMarkChange={changeMark}
              menuItems={[
                {
                  label: "Exploded view",
                  title: "The spectra as a cascade, coloured blue to red",
                  onSelect: openExploded,
                },
              ]}
            />
          </div>
          <BottomPane tabs={tabs} viewerRef={viewerRef} resizeLabel="Resize the image and the spectrum below it" />
        </div>
      </section>
    </div>
  );
}
