import { useMemo } from "react";
import type { DatasetSummary } from "../../api/client";
import type { EnergyRange, PlotWindow } from "../../components/SpectrumPlot";
import { EELS_BG_COLOUR } from "../../components/EelsFitPanel";
import { useEelsFit } from "../../state/eelsFit";
import { snapToAxis } from "../../state/pca";

const BAND = "eels-fit:background";

/** The EELS edge fit's background window (Quantify tab) on a spectrum plot,
 * hatched, while the fit panel is open; dragged by its edges or inside. */
export function useEelsBgBand(ds: DatasetSummary): {
  bands: PlotWindow[];
  onEdit: (id: string, range: EnergyRange) => boolean;
} {
  const shown = useEelsFit((s) => s.shown === ds.id);
  const opts = useEelsFit((s) => s.options[ds.id]);
  const lo = opts?.bg_lo ?? null;
  const hi = opts?.bg_hi ?? null;
  const bands = useMemo<PlotWindow[]>(
    () => (shown && lo !== null && hi !== null
      ? [{ id: BAND, kind: "background", lo, hi, color: EELS_BG_COLOUR, label: "Background" }]
      : []),
    [shown, lo, hi],
  );
  const onEdit = (id: string, r: EnergyRange) => {
    if (id !== BAND) return false;
    useEelsFit.getState().setOptions(ds.id, { bg_lo: snapToAxis(ds, r.lo), bg_hi: snapToAxis(ds, r.hi) });
    return true;
  };
  return { bands, onEdit };
}
