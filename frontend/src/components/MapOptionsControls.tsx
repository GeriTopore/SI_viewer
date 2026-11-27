import { useState } from "react";
import { api, type DatasetSummary, type MapOptions } from "../api/client";
import { useDatasets } from "../state/datasets";
import { useLayers } from "../state/layers";

const WIDTHS = [1, 1.5, 2, 3, 4];
const SMOOTHING = [1, 3, 5, 7, 9];
const DEFAULTS: MapOptions = { background: true, width: 2, smoothing: 1 };

/** How a spectrum image's element maps are computed; changes recompute them. */
export function MapOptionsControls({ ds }: { ds: DatasetSummary }) {
  const options = ds.eds?.map_options ?? DEFAULTS;
  const [busy, setBusy] = useState(false);
  const computed = useDatasets(
    (s) => s.files.find((f) => f.id === ds.file_id)?.datasets.filter((d) => d.parent_id === ds.id).length ?? 0,
  );

  const change = async (patch: Partial<MapOptions>) => {
    setBusy(true);
    try {
      const { dataset, file } = await api.setMapOptions(ds.id, { ...options, ...patch });
      const { replaceFile, updateDataset } = useDatasets.getState();
      replaceFile(file);
      updateDataset(dataset);
    } catch (e) {
      useLayers.getState().notify(`Could not update maps: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="map-options" title="Maps are computed for elements you add here">
      <span className="map-options-title">Maps</span>
      <label title="Subtract the background from either side of the peak">
        <input
          type="checkbox"
          checked={options.background}
          disabled={busy}
          onChange={(e) => change({ background: e.target.checked })}
        />
        Background
      </label>
      <label title="Integration window width, in units of the peak FWHM">
        Width
        <select value={options.width} disabled={busy} onChange={(e) => change({ width: Number(e.target.value) })}>
          {WIDTHS.map((w) => (
            <option key={w} value={w}>
              {w}× FWHM
            </option>
          ))}
        </select>
      </label>
      <label title="Average each pixel with its n × n neighbourhood">
        Smoothing
        <select
          value={options.smoothing}
          disabled={busy}
          onChange={(e) => change({ smoothing: Number(e.target.value) })}
        >
          {SMOOTHING.map((n) => (
            <option key={n} value={n}>
              {n === 1 ? "none" : `${n}×${n}`}
            </option>
          ))}
        </select>
      </label>
      <span className="muted small">
        {busy ? "Updating maps…" : computed ? `${computed} map${computed > 1 ? "s" : ""} from raw data` : ""}
      </span>
    </div>
  );
}
