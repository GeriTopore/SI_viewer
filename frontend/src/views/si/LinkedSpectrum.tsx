import { useEffect, useMemo, useState, type ReactNode } from "react";
import { api, dataVersion, type DatasetSummary, type EelsRole } from "../../api/client";
import { SpectrumPlot, type SpectrumSeries } from "../../components/SpectrumPlot";
import { allDatasets, useDatasets } from "../../state/datasets";
import { useDisplay } from "../../state/display";
import { CURSOR_COLOR, useSi, type Reduce, type SiState } from "../../state/si";
import type { BottomTab } from "../../state/ui";
import { SPECTRUM_COLOR, useEnergyAxis, useSpectrum } from "../SpectrumView";
import { CURSOR_CHANNEL, useSiSpectra } from "./useSiSpectra";

/** Tabs of a linked low/core loss pair. */
export const LOSS_TAB: Record<EelsRole, { id: BottomTab; label: string }> = {
  low_loss: { id: "low-loss", label: "Low loss" },
  core_loss: { id: "core-loss", label: "Core loss" },
};

/** The EELS spectrum image ``ds`` is linked with, if it is open. */
export function usePartner(ds: DatasetSummary): DatasetSummary | undefined {
  const id = ds.eels?.partner_id;
  return useDatasets((s) => (id ? allDatasets(s.files).find((d) => d.id === id) : undefined));
}

/** The pane's tabs: "Spectrum" alone, or, for a linked pair, the shown
 * dataset's own spectrum and its partner's (the shown one first). */
export function lossTabs(
  ds: DatasetSummary, partner: DatasetSummary | undefined, own: ReactNode, linked: ReactNode,
): { id: BottomTab; label: string; content: ReactNode }[] {
  if (!ds.eels || !partner?.eels) return [{ id: "spectrum", label: "Spectrum", content: own }];
  return [
    { ...LOSS_TAB[ds.eels.role], content: own },
    { ...LOSS_TAB[partner.eels.role], content: linked },
  ];
}

/** Linear/Log of a low-loss plot in a linked pair (log by default: the
 * zero-loss peak dominates); empty for any other plot, which uses the shared one. */
export function useLossLog(role: EelsRole | undefined, linked: boolean): { log?: boolean; onLog?: (log: boolean) => void } {
  const log = useDisplay((s) => s.lowLossLog);
  const setLog = useDisplay((s) => s.setLowLossLog);
  return linked && role === "low_loss" ? { log, onLog: setLog } : {};
}

/** Mean / Sum toggle, shared by every spectrum plot. */
export function ReduceToggle({ unit = "pixel" }: { unit?: string }) {
  const reduce = useSi((s) => s.reduce);
  const setReduce = useSi((s) => s.setReduce);
  return (
    <div className="segmented" role="group" aria-label="Spectrum normalisation">
      {(["mean", "sum"] as Reduce[]).map((r) => (
        <button
          key={r}
          className={reduce === r ? "active" : undefined}
          aria-pressed={reduce === r}
          onClick={() => setReduce(r)}
          title={r === "mean" ? `Average spectrum per ${unit} (comparable across selections)` : "Summed counts"}
        >
          {r === "mean" ? "Mean" : "Sum"}
        </button>
      ))}
    </div>
  );
}

interface Props {
  /** The linked partner of the spectrum image in the viewer. */
  partner: DatasetSummary;
  /** Cursor and regions of the spectrum image in the viewer (navigation
   * coordinates are the same in both). */
  st: SiState;
  lineScan: boolean;
  legend: ReactNode;
}

/** The partner's spectra at the viewer's cursor and regions, plus its sum
 * spectrum, drawn like the shown dataset's own. */
export function LinkedSpectrum({ partner, st, lineScan, legend }: Props) {
  const updateDataset = useDatasets((s) => s.updateDataset);
  const [loadError, setLoadError] = useState<string | null>(null);
  const ready = !partner.lazy;
  useEffect(() => {
    if (!partner.lazy) return;
    api
      .loadDataset(partner.id)
      .then(updateDataset)
      .catch((e) => setLoadError((e as Error).message));
  }, [partner.id, partner.lazy, updateDataset]);

  const reduce = useSi((s) => s.reduce);
  const cursorSize = useSi((s) => s.cursorSize);
  const axis = useEnergyAxis(partner);
  const whole = useSpectrum(partner, ready);
  const spectra = useSiSpectra(partner.id, dataVersion(partner), st, lineScan ? 1 : cursorSize, ready);
  const total = partner.navigation_shape.reduce((a, b) => a * b, 1);
  const lossLog = useLossLog(partner.eels?.role, true);

  const series = useMemo<SpectrumSeries[]>(() => {
    const k = (count: number) => (reduce === "mean" ? 1 / Math.max(1, count) : 1);
    const scaled = (data: Float32Array, f: number) => (f === 1 ? data : data.map((v) => v * f));
    const out: SpectrumSeries[] = [];
    if (st.showWhole && whole.data) {
      out.push({ label: lineScan ? "All positions" : "All pixels", data: scaled(whole.data, k(total)), color: SPECTRUM_COLOR });
    }
    for (const roi of st.rois) {
      const s = roi.visible ? spectra[roi.id] : undefined;
      if (!s || roi.shape.kind === "polygon") {
        if (s && s.pixels > 0) out.push({ label: roi.name, data: scaled(s.data, k(s.pixels)), color: roi.color });
        continue;
      }
      // A line scan's range sums the positions lo…hi.
      const count = lineScan ? roi.shape.x1 - roi.shape.x0 : s.pixels;
      if (count > 0) out.push({ label: roi.name, data: scaled(s.data, k(count)), color: roi.color });
    }
    const c = spectra[CURSOR_CHANNEL];
    if (st.showCursor && st.cursor && c) {
      out.push({ label: "Cursor", data: lineScan ? c.data : scaled(c.data, k(c.pixels)), color: CURSOR_COLOR });
    }
    return out;
  }, [st.showWhole, st.showCursor, st.cursor, st.rois, whole.data, spectra, reduce, total, lineScan]);

  const role = partner.eels?.role === "low_loss" ? "low loss" : "core loss";
  return (
    <SpectrumPlot
      name={`${partner.title}_spectra`}
      zoomKey={`${partner.id}:spectra`}
      x={axis.x}
      series={series}
      xLabel={axis.label}
      xUnits={axis.units}
      loading={whole.data ? null : ready ? "Summing spectra…" : `Loading ${role} into memory…`}
      error={whole.error ?? loadError}
      toolbar={<ReduceToggle unit={lineScan ? "position" : "pixel"} />}
      legend={legend}
      {...lossLog}
    />
  );
}
