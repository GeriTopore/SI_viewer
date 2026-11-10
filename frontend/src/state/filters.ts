import { api, type DatasetSummary, type Filters } from "../api/client";
import { allDatasets, useDatasets } from "./datasets";
import { useLayers } from "./layers";

/** The part of the filters that EELS spectrum images linked to a dataset share. */
export const spatialKey = (f: Filters) => `${f.spatial}|${f.sigma}|${f.size}|${f.small}|${f.large}`;

/** Send ``filters`` to the datasets ``ids`` and update the store; the EELS
 * spectrum images linked to them follow. Throws if the backend refuses. */
export async function applyFilters(ids: string[], filters: Filters): Promise<void> {
  const { replaceFile } = useDatasets.getState();
  let notes: string[] = [];
  let partnerChanged = false;
  for (const id of ids) {
    const before = allDatasets(useDatasets.getState().files).find((x) => x.id === id);
    const r = await api.setFilters(id, filters);
    replaceFile(r.file);
    notes = [...notes, ...r.notes];
    // The EELS spectrum images linked to it get the same spatial filter.
    if (before?.eels?.linked.length && spatialKey(before.filters) !== spatialKey(filters)) partnerChanged = true;
  }
  if (partnerChanged) await useDatasets.getState().syncFiles();
  if (notes.length) useLayers.getState().notify(notes.join(" "));
}

/** The dataset whose filters the focused dataset shows: a map made from a
 * spectrum image follows that spectrum image's processing. */
export function useFilterTarget(): DatasetSummary | undefined {
  return useDatasets((s) => {
    const all = allDatasets(s.files);
    const focused = all.find((d) => d.id === s.focusId);
    if (focused?.parent_id && focused.origin !== "background") {
      return all.find((d) => d.id === focused.parent_id) ?? focused;
    }
    return focused;
  });
}
