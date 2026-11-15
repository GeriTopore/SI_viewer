import { useMemo } from "react";
import type { DatasetSummary } from "../api/client";
import type { SpectrumSeries } from "../components/SpectrumPlot";
import { useModelAtCursor, useModelColours, useModelView } from "../state/curveModel";
import { useSiState } from "../state/si";

const DATA_COLOR = "#9ca3af";

/** A curve model (origin "curve-model") on its spectrum plot: at the cursor
 * (whose own spectrum is the model), the data it was fitted to and each
 * component ticked in the Info tab, aligned to the plot's channels ``x``
 * (per pixel with ``perPixel``, to match spectra plotted as means). */
export function useCurveModelSeries(ds: DatasetSummary, x: Float64Array, perPixel = false): SpectrumSeries[] {
  const isModel = ds.origin === "curve-model";
  const { result } = useModelAtCursor(ds);
  const view = useModelView(ds);
  const colours = useModelColours(ds);
  const showCursor = useSiState(ds.id).showCursor;
  return useMemo(() => {
    if (!isModel || !result || !showCursor || x.length < 2) return [];
    const k = perPixel ? 1 / Math.max(1, result.pixels) : 1;
    const i0 = Math.round((result.x[0] - x[0]) / (x[1] - x[0]));
    const place = (curve: number[]) => {
      const out: (number | null)[] = new Array(x.length).fill(null);
      curve.forEach((v, j) => {
        const i = i0 + j;
        if (i >= 0 && i < out.length) out[i] = v * k;
      });
      return out;
    };
    const out: SpectrumSeries[] = [{ label: "Data", data: place(result.data), color: DATA_COLOR, width: 1.25, fill: false }];
    for (const c of result.components) {
      if (!c.curve || view.hidden.includes(c.name)) continue;
      out.push({ label: c.name, data: place(c.curve), color: colours[c.name], dash: [4, 3], width: 1.25, fill: false });
    }
    return out;
  }, [isModel, result, showCursor, x, perPixel, view.hidden, colours]);
}
