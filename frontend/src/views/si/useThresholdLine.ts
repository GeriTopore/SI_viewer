import { useMemo } from "react";
import type { DatasetSummary } from "../../api/client";
import type { EnergyRange, PlotWindow } from "../../components/SpectrumPlot";
import { THRESHOLD_LINE, snapThreshold, useThickness, useThicknessState } from "../../state/thickness";

const ZLP_COLOUR = "#e11d48";

export interface ThresholdTool {
  /** The threshold line (empty while the tool is off or there is no value to draw). */
  windows: PlotWindow[];
  /** Handles an edit of the line; false if ``id`` is not the line. */
  onEdit: (id: string, range: EnergyRange, final: boolean) => boolean;
}

/** The Thickness tool's zero-loss threshold on the spectrum plot of ``ds``, if
 * ``ds`` is a low loss and "Show on spectrum" is on (Process tab): a vertical
 * line at the fixed threshold, or at the estimate for the summed spectrum when
 * it is automatic. Dragging it, in either case, sets a fixed threshold. */
export function useThresholdLine(ds: DatasetSummary): ThresholdTool {
  const low = ds.eels?.role === "low_loss";
  const st = useThicknessState(low ? ds : undefined);
  const estimate = useThickness((s) => (low ? s.estimate[ds.id]?.value : null)) ?? null;
  const value = st.threshold === "auto" ? estimate : st.threshold;
  const show = low && st.on && value !== null && Number.isFinite(value);

  const windows = useMemo<PlotWindow[]>(
    () =>
      show ? [{ id: THRESHOLD_LINE, kind: "threshold", lo: value!, hi: value!, color: ZLP_COLOUR, label: "Threshold" }] : [],
    [show, value],
  );

  const onEdit = (id: string, range: EnergyRange) => {
    if (id !== THRESHOLD_LINE) return false;
    useThickness.getState().patch(ds, { threshold: snapThreshold(ds, range.lo) });
    return true;
  };
  return { windows, onEdit };
}
