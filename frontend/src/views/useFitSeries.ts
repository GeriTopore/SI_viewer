import { useEffect, useMemo, useState } from "react";
import { api, dataVersion, type DatasetSummary, type EelsFitResult } from "../api/client";
import type { SpectrumSeries } from "../components/SpectrumPlot";
import { useDisplay } from "../state/display";
import { useEelsFit } from "../state/eelsFit";
import { datasetKey, useQuant } from "../state/quant";
import type { SiState } from "../state/si";

export const FIT_COLOR = "#f43f5e";
const BACKGROUND_COLOR = "#9ca3af";

/** The pixel block the cursor spectrum sums. */
export interface CursorBlock {
  x: number;
  y: number;
  size: number;
}

/** The fit of every pixel of ``ds`` at ``cursor`` (null while there is none
 * or nothing asks for it). The last one stays while the next is fetched. */
function usePixelFit(ds: DatasetSummary, cursor: CursorBlock | null | undefined, on: boolean) {
  const [fit, setFit] = useState<EelsFitResult | null>(null);
  const want = on && ds.pixel_fit && cursor ? `${ds.id}|${dataVersion(ds)}|${cursor.x}|${cursor.y}|${cursor.size}` : null;
  useEffect(() => {
    if (!want || !cursor) {
      setFit(null);
      return;
    }
    const abort = new AbortController();
    api
      .eelsPixelFit(ds.id, cursor.x, cursor.y, cursor.size, abort.signal)
      .then(setFit)
      .catch(() => !abort.signal.aborted && setFit(null)); // e.g. the fit failed at that pixel
    return () => abort.abort();
  }, [want]); // eslint-disable-line react-hooks/exhaustive-deps
  return want ? fit : null;
}

/** The curves of a fit, as both fits (EDS and the EELS model) give them. */
interface FitCurves {
  x: number[];
  model: number[];
  background: number[];
  elements: Record<string, number[]>;
  pixels: number;
}

/** The latest fits of ``ds`` (EDS, or the EELS model) as plot series (model, background and, if asked,
 * each element's contribution), aligned to the plot's channels ``x``.
 * With ``perPixel`` the fitted counts are divided by the pixels they sum, to
 * match spectra plotted as per-pixel means. EELS fits of a spectrum image's
 * spectra are drawn while ``shown`` shows that spectrum (the whole, the
 * cursor, a region); when every pixel was fitted, the fit at ``cursor`` is
 * drawn on the cursor spectrum. */
export function useFitSeries(
  ds: DatasetSummary, x: Float64Array, perPixel = false, cursor?: CursorBlock | null,
  shown?: Pick<SiState, "showWhole" | "showCursor" | "rois">,
): SpectrumSeries[] {
  const edsFit = useQuant((s) => s.fits[datasetKey(ds)]);
  const eelsFits = useEelsFit((s) => s.fits[datasetKey(ds)]);
  const showFit = useQuant((s) => s.showFit);
  const pixelFit = usePixelFit(ds, cursor, showFit && !!ds.edges);
  const showComponents = useQuant((s) => s.showComponents);
  const colours = useDisplay((s) => s.elementColours);

  return useMemo(() => {
    if (!showFit || x.length < 2) return [];
    const visible = (source: string) =>
      !shown ||
      (source === "all"
        ? shown.showWhole
        : source === "cursor"
          ? shown.showCursor && !pixelFit
          : shown.rois.some((r) => r.id === source && r.visible));
    const drawn: { label: string; result: FitCurves }[] = !ds.edges
      ? edsFit ? [{ label: "", result: edsFit.result }] : []
      : [
          ...(eelsFits ?? []).filter((f) => visible(f.source)).map((f) => ({ label: f.label, result: f.result })),
          ...(pixelFit ? [{ label: "Cursor", result: pixelFit }] : []),
        ];
    // Several fits are told apart by their spectrum's name.
    const named = (label: string, of: string) => (drawn.length > 1 && of ? `${label} · ${of}` : label);
    return drawn.flatMap(({ label, result: r }) => {
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
        { label: named("Background", label), data: place(r.background), color: BACKGROUND_COLOR, dash: [4, 3], fill: false },
      ];
      if (showComponents) {
        for (const [el, curve] of Object.entries(r.elements)) {
          series.push({ label: named(`${el} fit`, label), data: place(curve), color: colours[el] ?? "#e0b84a", dash: [2, 2], fill: false });
        }
      }
      series.push({ label: named("Fit", label), data: place(r.model), color: FIT_COLOR, width: 1.6, fill: false });
      return series;
    });
  }, [ds.edges, edsFit, eelsFits, pixelFit, shown, showFit, showComponents, colours, x, perPixel]);
}
