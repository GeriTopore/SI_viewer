import { api, type DatasetSummary, type FitQuantity } from "../api/client";
import { useDatasets } from "../state/datasets";
import { useLayers } from "../state/layers";

const QUANTITIES: FitQuantity[] = ["counts", "at%"];

/** Counts | at% for the per-pixel fits that ``ds`` come from (fitted maps
 * that have at%); switching one map switches every map of its fit. Nothing
 * when none of ``ds`` has at%. */
export function FitQuantitySwitch({ ds }: { ds: DatasetSummary[] }) {
  const maps = ds.filter((d) => d.origin === "fitted" && d.quantities.includes("at%"));
  if (!maps.length) return null;
  const fits = [...new Set(maps.map((d) => d.parent_id!))];
  const shown: FitQuantity = maps.every((d) => d.quantity === "at%") ? "at%" : "counts";
  const set = async (q: FitQuantity) => {
    try {
      for (const id of fits) useDatasets.getState().replaceFile((await api.setFitQuantity(id, q)).file);
    } catch (e) {
      useLayers.getState().notify(`Could not switch: ${(e as Error).message}`);
    }
  };
  return (
    <div className="segmented fit-quantity" role="group" aria-label="Fitted maps show">
      {QUANTITIES.map((q) => (
        <button
          key={q}
          className={shown === q ? "active" : undefined}
          aria-pressed={shown === q}
          onClick={(e) => {
            e.stopPropagation();
            if (shown !== q) void set(q);
          }}
          title={q === "counts" ? "Fitted maps: intensity" : "Fitted maps: composition (at%)"}
        >
          {q === "counts" ? "Counts" : "at%"}
        </button>
      ))}
    </div>
  );
}
