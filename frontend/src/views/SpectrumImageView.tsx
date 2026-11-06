import { useEffect, useMemo, useState } from "react";
import { fetchFrame, type Frame } from "../api/arrays";
import { api, type DatasetSummary, type NavigatorSource } from "../api/client";
import { ContrastControls } from "../components/ContrastControls";
import { ImageView } from "../components/ImageView";
import { SpectrumPlot, type SpectrumSeries } from "../components/SpectrumPlot";
import { useDatasets } from "../state/datasets";
import { bytesLabel } from "../format";
import { imageGeometry, isAbort } from "./geometry";
import { SPECTRUM_COLOR, useEnergyAxis, useSpectrum } from "./SpectrumView";
import { useImageDisplay } from "./useImageDisplay";

/** Navigator image (ADF, total counts or a map) beside the SI's spectrum. */
export function SpectrumImageView({ ds }: { ds: DatasetSummary }) {
  const updateDataset = useDatasets((s) => s.updateDataset);
  const [loadError, setLoadError] = useState<string | null>(null);
  const ready = !ds.lazy;

  // Spectrum images are loaded into RAM up front: everything else is fast after.
  useEffect(() => {
    if (!ds.lazy) return;
    api
      .loadDataset(ds.id)
      .then(updateDataset)
      .catch((e) => setLoadError((e as Error).message));
  }, [ds.id, ds.lazy, updateDataset]);

  const [sources, setSources] = useState<NavigatorSource[]>([]);
  const [sourceId, setSourceId] = useState<string | null>(null);
  useEffect(() => {
    api.navigators(ds.id).then((s) => {
      setSources(s);
      setSourceId((cur) => cur ?? s[0]?.id ?? ds.id);
    });
  }, [ds.id]);

  // The total-counts map (source = the SI itself) needs the SI in memory.
  const sourceReady = sourceId !== null && (sourceId !== ds.id || ready);
  const [navFrame, setNavFrame] = useState<Frame | null>(null);
  const [navError, setNavError] = useState<string | null>(null);
  useEffect(() => {
    if (!sourceReady) return;
    const ac = new AbortController();
    setNavFrame(null);
    fetchFrame(sourceId!, {}, ac.signal)
      .then(setNavFrame)
      .catch((e) => !isAbort(e) && setNavError((e as Error).message));
    return () => ac.abort();
  }, [sourceId, sourceReady]);

  const navKey = `${ds.id}:nav:${sourceId}`;
  const { display, setDisplay } = useImageDisplay(navKey, navFrame);
  const g = imageGeometry(ds, "navigation");

  const axis = useEnergyAxis(ds);
  const spectrum = useSpectrum(ds, ready);
  const series = useMemo<SpectrumSeries[]>(
    () => (spectrum.data ? [{ label: "Sum", data: spectrum.data, color: SPECTRUM_COLOR }] : []),
    [spectrum.data],
  );

  const loadingMsg = `Loading spectrum image into memory (${bytesLabel(ds.nbytes)})…`;

  return (
    <div className="view si-view">
      <div className="si-layout">
        <section className="si-pane">
          <div className="si-pane-header">
            <label className="field">
              Navigator
              <select value={sourceId ?? ""} onChange={(e) => setSourceId(e.target.value)}>
                {sources.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <ContrastControls frame={navFrame} display={display} onChange={setDisplay} />
          <ImageView
            viewKey={`${ds.id}:nav`}
            frame={navFrame}
            fullWidth={g.width}
            fullHeight={g.height}
            display={display}
            calX={g.calX}
            calY={g.calY}
            loading={navFrame ? null : sourceId === ds.id && !ready ? loadingMsg : "Loading navigator…"}
            error={navError ?? loadError}
          />
        </section>
        <section className="si-pane">
          <SpectrumPlot
            x={axis.x}
            series={series}
            xLabel={axis.label}
            xUnits={axis.units}
            loading={spectrum.data ? null : ready ? "Summing spectra…" : loadingMsg}
            error={spectrum.error ?? loadError}
          />
        </section>
      </div>
    </div>
  );
}
