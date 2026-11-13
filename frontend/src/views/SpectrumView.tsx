import { useEffect, useMemo, useState } from "react";
import { useZlp } from "../state/zlp";
import { fetchSpectrum } from "../api/arrays";
import { dataVersion, type DatasetSummary } from "../api/client";
import { ElementsPicker } from "../components/ElementsPicker";
import { SpectrumPlot, type SpectrumSeries } from "../components/SpectrumPlot";
import { isAbort, spectralAxis } from "./geometry";
import { useMarkers } from "./si/xray";
import { useFitSeries } from "./useFitSeries";
import { useBackgroundPreview } from "./si/useBackgroundPreview";
import { NormaliseToggle, useNormBand } from "../components/NormaliseSection";
import { canNormalise, normaliseSeries, useNormSettings } from "../state/normalise";

export const SPECTRUM_COLOR = "#4c8dff";

/** Energy axis values (channel centres) for a dataset's spectral axis. */
export function useEnergyAxis(ds: DatasetSummary) {
  const axis = spectralAxis(ds);
  // A shift by hand being typed (ZLP alignment without a low loss): shown live.
  const preview = useZlp((s) => s.preview[ds.id] ?? 0);
  const x = useMemo(() => {
    const out = new Float64Array(axis?.size ?? 0);
    for (let i = 0; i < out.length; i++) out[i] = (axis?.offset ?? 0) + preview + i * (axis?.scale ?? 1);
    return out;
  }, [axis?.size, axis?.offset, axis?.scale, preview]); // eslint-disable-line react-hooks/exhaustive-deps
  return { x, label: axis?.name || "Energy", units: axis?.units ?? "" };
}

/** Fetch the spectrum (or SI sum spectrum) of a dataset once ``ready``, and
 * again when its data changes (e.g. smoothing). */
export function useSpectrum(ds: DatasetSummary, ready = true) {
  const [data, setData] = useState<Float32Array | null>(null);
  const [error, setError] = useState<string | null>(null);
  const version = dataVersion(ds);
  useEffect(() => {
    if (!ready) return;
    const ac = new AbortController();
    fetchSpectrum(ds.id, ac.signal)
      .then(setData)
      .catch((e) => !isAbort(e) && setError((e as Error).message));
    return () => ac.abort();
  }, [ds.id, version, ready]);
  return { data, error };
}

export function SpectrumView({ ds }: { ds: DatasetSummary }) {
  const axis = useEnergyAxis(ds);
  const { data, error } = useSpectrum(ds);
  const fitSeries = useFitSeries(ds, axis.x);
  const series = useMemo<SpectrumSeries[]>(
    () => (data ? [{ label: ds.title, data, color: SPECTRUM_COLOR }, ...fitSeries] : []),
    [data, ds.title, fitSeries],
  );
  const bg = useBackgroundPreview(ds, axis.x, series);
  // Normalised (Process tab, the plot's checkbox): the spectrum alone, normalised.
  const norm = useNormSettings(ds);
  const plotted = useMemo(
    () => (norm.on && data ? normaliseSeries([series[0]], axis.x, norm) : bg.series),
    [norm, data, series, axis.x, bg.series],
  );
  const normBand = useNormBand(ds);
  const windows = useMemo(() => [...bg.bands, ...normBand.bands], [bg.bands, normBand.bands]);
  const [preview, setPreview] = useState<string | null>(null);
  const markers = useMarkers(ds, preview, axis.x);
  return (
    <div className="view">
      <SpectrumPlot
        name={ds.title}
        x={axis.x}
        series={plotted}
        windows={windows}
        onEditWindow={(id, range, final) => normBand.onEdit(id, range) || bg.onEditBand(id, range, final)}
        xLabel={axis.label}
        xUnits={axis.units}
        loading={data ? null : "Loading spectrum…"}
        error={error}
        markers={markers}
        toolbar={
          <>
            <ElementsPicker ds={ds} onPreview={setPreview} />
            {canNormalise(ds) && <NormaliseToggle d={ds} />}
          </>
        }
      />
    </div>
  );
}
