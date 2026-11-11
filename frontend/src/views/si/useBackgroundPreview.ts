import { useEffect, useMemo, useState } from "react";
import { api, type BackgroundPixel, type BackgroundPreview, type DatasetSummary } from "../../api/client";
import type { EnergyRange, PlotWindow, SpectrumSeries } from "../../components/SpectrumPlot";
import {
  BG_BAND, SIGNAL_BAND, canRemoveBackground, energyRange, isBgBand, snapEnergy, useBackground, useBgState,
  validateWindows,
} from "../../state/background";
import { isAbort } from "../geometry";

const BG_COLOUR = "#f97316";
const SIGNAL_COLOUR = "#22c55e";
const DEBOUNCE_MS = 150;

/** What the preview was computed for: the spectra sent, and their fits. */
interface Fitted {
  inputs: SpectrumSeries[];
  fit: BackgroundPreview;
}

export interface BackgroundTool {
  /** The series to draw: as given, plus a dashed fit under each (or, when
   * "Subtracted" is on, the remainders over the signal window). */
  series: SpectrumSeries[];
  /** The Background and Signal bands (empty while the tool is off). */
  bands: PlotWindow[];
  /** Handles an edit of a band; false if ``id`` is not one of them. */
  onEditBand: (id: string, range: EnergyRange, final: boolean) => boolean;
}

/** The Background removal tool on a spectrum plot: while it is on for ``ds``
 * (switched in the Process tab), the fitted background of every series is
 * requested (debounced, stale responses dropped) and drawn dashed, and the
 * Background and Signal bands are offered, kept in step with the section's
 * inputs. ``series`` are the raw spectra on the energy axis ``x``; ``pixels``
 * the block of pixels each is from, if any (with LBA, these are fitted to
 * their neighbourhood's average). */
export function useBackgroundPreview(
  ds: DatasetSummary, x: Float64Array, series: SpectrumSeries[], pixels?: (BackgroundPixel | null)[],
): BackgroundTool {
  const st = useBgState(ds);
  const { options } = st;
  const eligible = canRemoveBackground(ds) && !ds.background;
  const invalid = validateWindows(options);
  const active = eligible && st.on && !invalid && x.length > 1;
  const optionsKey = JSON.stringify(options);

  const [fitted, setFitted] = useState<Fitted | null>(null);
  useEffect(() => {
    const { setInfo } = useBackground.getState();
    if (!active || !series.length) {
      setFitted(null);
      setInfo(ds.id, { r: null, error: null });
      return;
    }
    const ac = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const fit = await api.backgroundPreview(
          ds.id, options, series.map((s) => Array.from(s.data, (v) => v ?? 0)),
          options.lba > 0 ? series.map((_, i) => pixels?.[i] ?? null) : undefined, ac.signal,
        );
        setFitted({ inputs: series, fit });
        // The exponent shown is the cursor's (or the only spectrum's).
        const i = series.findIndex((s) => s.label === "Cursor");
        setInfo(ds.id, { r: fit.r[i >= 0 ? i : ds.kind === "spectrum" ? 0 : -1] ?? null, error: null });
      } catch (e) {
        if (!isAbort(e)) setInfo(ds.id, { r: null, error: (e as Error).message });
      }
    }, DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      ac.abort();
    };
  }, [active, optionsKey, series, ds.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // No preview while the tool is off or the windows are invalid.
  useEffect(() => {
    if (!eligible || !st.on) useBackground.getState().setInfo(ds.id, { r: null, error: null });
  }, [eligible, st.on, ds.id]);

  const out = useMemo<SpectrumSeries[]>(() => {
    if (!active) return series;
    const n = x.length;
    const empty = () => new Array<number | null>(n).fill(null);
    // The fits belong to the spectra they were computed for (the latest ones
    // follow within a moment); until the first arrives the lines are empty.
    const current = fitted && fitted.inputs.length === series.length ? fitted : null;
    const lines = series.map((s, i) => {
      const data = empty();
      const bg = current?.fit.background[i];
      if (bg) bg.forEach((v, k) => (data[current!.fit.start + k] = v));
      return { s, raw: current?.inputs[i].data ?? s.data, data };
    });
    if (st.subtracted) {
      const first = options.signal[0] - (x[1] - x[0]) / 2;
      return lines.map(({ s, raw, data }) => ({
        ...s,
        data: data.map((bg, k) => (bg === null || x[k] < first ? null : (raw[k] ?? 0) - bg)),
      }));
    }
    return [
      ...series,
      ...lines.map(({ s, data }) => ({
        label: `${s.label} (bg)`, data, color: s.color, dash: [5, 4], width: 1.25, fill: false,
      })),
    ];
  }, [active, series, fitted, st.subtracted, x]); // eslint-disable-line react-hooks/exhaustive-deps

  const bands = useMemo<PlotWindow[]>(() => {
    if (!eligible || !st.on) return [];
    const [b0, b1] = options.background;
    const [s0, s1] = options.signal;
    return [
      { id: BG_BAND, kind: "background", lo: b0, hi: b1, color: BG_COLOUR, label: "Background", max: s0 },
      { id: SIGNAL_BAND, kind: "signal", lo: s0, hi: s1, color: SIGNAL_COLOUR, label: "Signal", min: b1 },
    ];
  }, [eligible, st.on, optionsKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const range = energyRange(ds);
  const onEditBand = (id: string, r: EnergyRange) => {
    if (!isBgBand(id)) return false;
    const snap = (v: number) => (range ? snapEnergy(range.e0, range.step, v) : v);
    const window: [number, number] = [snap(r.lo), snap(r.hi)];
    useBackground.getState().setOptions(ds, id === BG_BAND ? { background: window } : { signal: window });
    return true;
  };

  return { series: out, bands, onEditBand };
}
