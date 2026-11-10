import { useMemo } from "react";
import type { DatasetSummary } from "../api/client";
import type { SpectrumSeries } from "../components/SpectrumPlot";
import { useDisplay } from "../state/display";
import { useEelsFit } from "../state/eelsFit";
import { datasetKey, useQuant } from "../state/quant";

export const FIT_COLOR = "#f43f5e";
const BACKGROUND_COLOR = "#9ca3af";

/** The latest fit of ``ds`` (EDS, or the EELS model) as plot series (model, background and, if asked,
 * each element's contribution), aligned to the plot's channels ``x``.
 * With ``perPixel`` the fitted counts are divided by the pixels they sum, to
 * match spectra plotted as per-pixel means. */
export function useFitSeries(ds: DatasetSummary, x: Float64Array, perPixel = false): SpectrumSeries[] {
  const edsFit = useQuant((s) => s.fits[datasetKey(ds)]);
  const eelsFit = useEelsFit((s) => s.fits[datasetKey(ds)]);
  const fit = ds.edges ? eelsFit : edsFit;
  const showFit = useQuant((s) => s.showFit);
  const showComponents = useQuant((s) => s.showComponents);
  const colours = useDisplay((s) => s.elementColours);

  return useMemo(() => {
    if (!fit || !showFit || x.length < 2) return [];
    const r = fit.result;
    const scale = perPixel ? 1 / r.pixels : 1;
    // The fit covers channels from its first energy onwards (same axis).
    const i0 = Math.round((r.x[0] - x[0]) / (x[1] - x[0]));
    const place = (curve: number[]) => {
      const out: (number | null)[] = new Array(x.length).fill(null);
      curve.forEach((v, j) => {
        const i = i0 + j;
        if (i >= 0 && i < out.length) out[i] = v * scale;
      });
      return out;
    };
    const series: SpectrumSeries[] = [
      { label: "Background", data: place(r.background), color: BACKGROUND_COLOR, dash: [4, 3], fill: false },
    ];
    if (showComponents) {
      for (const [el, curve] of Object.entries(r.elements)) {
        series.push({ label: `${el} fit`, data: place(curve), color: colours[el] ?? "#e0b84a", dash: [2, 2], fill: false });
      }
    }
    series.push({ label: "Fit", data: place(r.model), color: FIT_COLOR, width: 1.6, fill: false });
    return series;
  }, [fit, showFit, showComponents, colours, x, perPixel]);
}
