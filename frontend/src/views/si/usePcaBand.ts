import { useMemo } from "react";
import type { DatasetSummary } from "../../api/client";
import type { EnergyRange, PlotWindow } from "../../components/SpectrumPlot";
import { PCA_BAND, canDecompose, snapToAxis, usePca, usePcaState } from "../../state/pca";

const PCA_COLOUR = "#0891b2";

/** The PCA tool's energy window on a spectrum plot: a band (while the PCA
 * section is open) whose edges and inside are dragged to change it. */
export function usePcaBand(ds: DatasetSummary): {
  bands: PlotWindow[];
  onEdit: (id: string, range: EnergyRange, final: boolean) => boolean;
} {
  const st = usePcaState(ds);
  const { lo, hi } = st.options;
  const on = st.on && canDecompose(ds) && lo !== null && hi !== null;
  const bands = useMemo<PlotWindow[]>(
    () => (on ? [{ id: PCA_BAND, kind: "signal", lo: lo!, hi: hi!, color: PCA_COLOUR, label: "PCA" }] : []),
    [on, lo, hi],
  );
  const onEdit = (id: string, r: EnergyRange) => {
    if (id !== PCA_BAND) return false;
    usePca.getState().setOptions(ds, { lo: snapToAxis(ds, r.lo), hi: snapToAxis(ds, r.hi) });
    return true;
  };
  return { bands, onEdit };
}
