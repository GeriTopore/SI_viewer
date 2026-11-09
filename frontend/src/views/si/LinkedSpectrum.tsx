import { useEffect, useMemo, useState, type ReactNode } from "react";
import { api, dataVersion, type DatasetSummary, type EelsRole } from "../../api/client";
import { SpectrumPlot, type SpectrumSeries } from "../../components/SpectrumPlot";
import { allDatasets, groupOf, useDatasets } from "../../state/datasets";
import { useDisplay } from "../../state/display";
import { CURSOR_COLOR, useSi, type Reduce, type SiState } from "../../state/si";
import { eelsTab, type BottomTab } from "../../state/ui";
import { SPECTRUM_COLOR, useEnergyAxis, useSpectrum } from "../SpectrumView";
import { CURSOR_CHANNEL, useSiSpectra } from "./useSiSpectra";
import { useThresholdLine } from "./useThresholdLine";

/** The link group ``ds`` is in, itself included (low loss first, then core
 * losses by energy offset); just ``ds`` when it is not linked. */
export function useGroup(ds: DatasetSummary): DatasetSummary[] {
  const files = useDatasets((s) => s.files);
  return useMemo(() => groupOf(ds, new Map(allDatasets(files).map((d) => [d.id, d]))), [ds, files]);
}

/** A group member's tab: "Low loss", "Core loss" (the only one), or the
 * energy range of one of several core losses, e.g. "430–1640 eV". */
export function memberLabel(m: DatasetSummary, group: DatasetSummary[]): string {
  if (m.eels?.role === "low_loss") return "Low loss";
  if (group.filter((d) => d.eels?.role !== "low_loss").length < 2) return "Core loss";
  const axis = m.axes.find((a) => !a.navigate);
  if (!axis) return m.title;
  const lo = axis.offset;
  const hi = axis.offset + axis.scale * (axis.size - 1);
  return `${Math.round(lo)}–${Math.round(hi)}${axis.units ? ` ${axis.units}` : ""}`;
}

/** The pane's tabs: "Spectrum" alone, or, for a linked group, one per member
 * (the shown dataset's own spectrum, the others' as ``LinkedSpectrum``). The
 * ids stay the same when another member is shown. */
export function memberTabs(
  ds: DatasetSummary, group: DatasetSummary[], own: ReactNode, st: SiState, lineScan: boolean, legend: ReactNode,
): { id: BottomTab; label: string; content: ReactNode }[] {
  if (!ds.eels || group.length < 2) return [{ id: "spectrum", label: "Spectrum", content: own }];
  return group.map((m) => ({
    id: eelsTab(m.id),
    label: memberLabel(m, group),
    content: m.id === ds.id ? own : <LinkedSpectrum member={m} st={st} lineScan={lineScan} legend={legend} />,
  }));
}

/** Linear/Log of a low-loss plot in a linked group (its own setting, linear
 * by default); empty for any other plot, which uses the shared one. */
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
  /** A spectrum image linked to the one in the viewer. */
  member: DatasetSummary;
  /** Cursor and regions of the spectrum image in the viewer (navigation
   * coordinates are the same in both). */
  st: SiState;
  lineScan: boolean;
  legend: ReactNode;
}

/** A linked member's spectra at the viewer's cursor and regions, plus its sum
 * spectrum, drawn like the shown dataset's own. */
export function LinkedSpectrum({ member, st, lineScan, legend }: Props) {
  const updateDataset = useDatasets((s) => s.updateDataset);
  const [loadError, setLoadError] = useState<string | null>(null);
  const ready = !member.lazy;
  useEffect(() => {
    if (!member.lazy) return;
    api
      .loadDataset(member.id)
      .then(updateDataset)
      .catch((e) => setLoadError((e as Error).message));
  }, [member.id, member.lazy, updateDataset]);

  const reduce = useSi((s) => s.reduce);
  const cursorSize = useSi((s) => s.cursorSize);
  const axis = useEnergyAxis(member);
  const whole = useSpectrum(member, ready);
  const spectra = useSiSpectra(member.id, dataVersion(member), st, lineScan ? 1 : cursorSize, ready);
  const total = member.navigation_shape.reduce((a, b) => a * b, 1);
  const lossLog = useLossLog(member.eels?.role, true);
  // Thickness (Process tab): the zero-loss threshold, drawn on the low loss.
  const zlp = useThresholdLine(member);

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

  const role = member.eels?.role === "low_loss" ? "low loss" : "core loss";
  return (
    <SpectrumPlot
      name={`${member.title}_spectra`}
      zoomKey={`${member.id}:spectra`}
      x={axis.x}
      series={series}
      windows={zlp.windows}
      onEditWindow={zlp.onEdit}
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
