import { useEffect, useMemo, useState } from "react";
import { fetchSpectrum } from "../api/arrays";
import type { DatasetSummary } from "../api/client";
import { ElementsPicker } from "../components/ElementsPicker";
import { SpectrumPlot, type SpectrumSeries } from "../components/SpectrumPlot";
import { isAbort, spectralAxis } from "./geometry";
import { useLineMarkers } from "./si/xray";
import { useFitSeries } from "./useFitSeries";

export const SPECTRUM_COLOR = "#4c8dff";

/** Energy axis values (channel centres) for a dataset's spectral axis. */
export function useEnergyAxis(ds: DatasetSummary) {
  const axis = spectralAxis(ds);
  const x = useMemo(() => {
    const out = new Float64Array(axis?.size ?? 0);
    for (let i = 0; i < out.length; i++) out[i] = (axis?.offset ?? 0) + i * (axis?.scale ?? 1);
    return out;
  }, [axis?.size, axis?.offset, axis?.scale]); // eslint-disable-line react-hooks/exhaustive-deps
  return { x, label: axis?.name || "Energy", units: axis?.units ?? "" };
}

/** Fetch the spectrum (or SI sum spectrum) of a dataset once ``ready``. */
export function useSpectrum(ds: DatasetSummary, ready = true) {
  const [data, setData] = useState<Float32Array | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!ready) return;
    const ac = new AbortController();
    fetchSpectrum(ds.id, ac.signal)
      .then(setData)
      .catch((e) => !isAbort(e) && setError((e as Error).message));
    return () => ac.abort();
  }, [ds.id, ready]);
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
  const [preview, setPreview] = useState<string | null>(null);
  const markers = useLineMarkers(ds.eds?.elements ?? [], preview, ds.eds?.beam_energy ?? null, ds.eds?.xray_lines, [axis.x[0], axis.x[axis.x.length - 1]]);
  return (
    <div className="view">
      <SpectrumPlot
        x={axis.x}
        series={series}
        xLabel={axis.label}
        xUnits={axis.units}
        loading={data ? null : "Loading spectrum…"}
        error={error}
        markers={markers}
        toolbar={<ElementsPicker ds={ds} onPreview={setPreview} />}
      />
    </div>
  );
}
