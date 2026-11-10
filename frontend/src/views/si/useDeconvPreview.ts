import { useEffect, useMemo, useState } from "react";
import { api, type DatasetSummary, type DeconvolutionPreview, type DeconvolutionSelection } from "../../api/client";
import type { SpectrumSeries } from "../../components/SpectrumPlot";
import { useDatasets } from "../../state/datasets";
import {
  canDeconvolve, deconvolutionLowLoss, methodProblem, optionsProblem, useDeconvState, useDeconvolution,
} from "../../state/deconvolution";
import { isAbort } from "../geometry";

const DEBOUNCE_MS = 200;

/** The Deconvolution tool on a spectrum plot: while its preview is on for
 * ``ds`` (switched in the Process tab), the deconvolved spectrum of every
 * series with a selection of pixels (``selections``, one per series: null
 * for one that has none, e.g. a fit) is requested (debounced, stale
 * responses dropped) and drawn dashed in the series' colour. ``mean``: the
 * series are per pixel (else summed). */
export function useDeconvPreview(
  ds: DatasetSummary, series: SpectrumSeries[], selections: (DeconvolutionSelection | null)[], mean: boolean,
): SpectrumSeries[] {
  const st = useDeconvState(ds);
  const { options } = st;
  const files = useDatasets((s) => s.files);
  const ll = deconvolutionLowLoss(ds, files);
  const eligible = canDeconvolve(ds) && !ds.deconvolution;
  const problem = methodProblem(options.method, options.psf, ds, ll) ?? optionsProblem(options);
  const active = eligible && st.on && !problem;
  const wanted = selections.map((s, i) => (s ? { s, i } : null)).filter((x): x is { s: DeconvolutionSelection; i: number } => !!x);
  const key = JSON.stringify({ options, sel: wanted.map((w) => w.s), rev: [ds.revision, ds.data_rev, ll?.revision, ll?.data_rev] });

  const [result, setResult] = useState<{ key: string; index: number[]; preview: DeconvolutionPreview } | null>(null);
  useEffect(() => {
    const { setInfo } = useDeconvolution.getState();
    if (!active || !wanted.length) {
      setResult(null);
      setInfo(ds.id, { threshold: null, fwhm: null, error: null });
      return;
    }
    const ac = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const preview = await api.deconvolutionPreview(ds.id, options, wanted.map((w) => w.s), ac.signal);
        setResult({ key, index: wanted.map((w) => w.i), preview });
        setInfo(ds.id, { threshold: preview.threshold, fwhm: preview.fwhm, error: null });
      } catch (e) {
        if (!isAbort(e)) setInfo(ds.id, { threshold: null, fwhm: null, error: (e as Error).message });
      }
    }, DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      ac.abort();
    };
  }, [active, key, ds.id]); // eslint-disable-line react-hooks/exhaustive-deps

  return useMemo(() => {
    if (!active || !result) return series;
    const lines: SpectrumSeries[] = [];
    result.index.forEach((i, k) => {
      const s = series[i];
      const data = result.preview.spectra[k];
      const n = result.preview.pixels[k];
      if (!s || !data || !n) return;
      const f = mean ? 1 / n : 1;
      lines.push({
        label: `${s.label} (deconvolved)`,
        data: f === 1 ? data : data.map((v) => v * f),
        color: s.color, dash: [5, 4], width: 1.25, fill: false,
      });
    });
    return [...series, ...lines];
  }, [active, result, series, mean]);
}
