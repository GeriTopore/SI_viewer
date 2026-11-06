import { useEffect, useMemo, useState } from "react";
import { fetchFrame, type Frame } from "../api/arrays";
import { api, type DatasetSummary, type NavigatorSource } from "../api/client";
import { LayeredViewer } from "../components/LayeredViewer";
import { SpectrumPlot, type SpectrumSeries } from "../components/SpectrumPlot";
import { allDatasets, useDatasets } from "../state/datasets";
import { bytesLabel } from "../format";
import { imageGeometry, isAbort } from "./geometry";
import { SPECTRUM_COLOR, useEnergyAxis, useSpectrum } from "./SpectrumView";

/** Navigator source meaning "no base image": only the overlaid maps are shown. */
const NONE = "none";

/** Navigator image (ADF, total counts or a map) beside the SI's spectrum. */
export function SpectrumImageView({ ds, images }: { ds: DatasetSummary; images: DatasetSummary[] }) {
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
  const sourceReady = sourceId !== null && sourceId !== NONE && (sourceId !== ds.id || ready);
  // The frame is kept with the source it came from: right after switching
  // source the previous image must not be used (e.g. for auto contrast).
  const [nav, setNav] = useState<{ source: string; frame: Frame } | null>(null);
  const [navError, setNavError] = useState<string | null>(null);
  useEffect(() => {
    if (!sourceReady) return;
    const ac = new AbortController();
    const source = sourceId!;
    fetchFrame(source, {}, ac.signal)
      .then((frame) => setNav({ source, frame }))
      .catch((e) => !isAbort(e) && setNavError((e as Error).message));
    return () => ac.abort();
  }, [sourceId, sourceReady]);
  const navFrame = nav && nav.source === sourceId ? nav.frame : null;

  // A navigator image shares its display settings (contrast, element colour)
  // with the dataset itself; the SI's total-counts map has its own.
  const files = useDatasets((s) => s.files);
  const sourceDataset =
    sourceId && sourceId !== ds.id ? allDatasets(files).find((d) => d.id === sourceId) : undefined;
  const navKey = sourceId === ds.id ? `${ds.id}:total` : (sourceId ?? "");

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
                <option value={NONE}>None (maps only)</option>
              </select>
            </label>
          </div>
          <LayeredViewer
            viewKey={`${ds.id}:nav`}
            base={
              sourceId === NONE
                ? null
                : {
                    key: navKey,
                    label: sources.find((s) => s.id === sourceId)?.label ?? "Navigator",
                    frame: navFrame,
                    dataset: sourceDataset,
                  }
            }
            onRemoveBase={() => setSourceId(NONE)}
            images={images.filter((d) => d.id !== sourceId)}
            geometry={imageGeometry(ds, "navigation")}
            loading={
              navFrame || sourceId === NONE
                ? null
                : sourceId === ds.id && !ready
                  ? loadingMsg
                  : "Loading navigator…"
            }
            error={navError ?? loadError}
            empty="No navigator image. Click element maps in the dataset pane to show them here."
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
