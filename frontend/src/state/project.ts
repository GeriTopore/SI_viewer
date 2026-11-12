import type { StoreApi, UseBoundStore } from "zustand";
import { api, type SaveLevel } from "../api/client";
import { useBackground } from "./background";
import { layoutPane, paneFiles, useDatasets, workspaceOf } from "./datasets";
import { useDeconvolution } from "./deconvolution";
import { useDisplay } from "./display";
import { useEelsFit } from "./eelsFit";
import { useFftBoxes } from "./fftBoxes";
import { useLayers } from "./layers";
import { useProfiles } from "./profiles";
import { useQuant } from "./quant";
import { useSi } from "./si";
import { useThickness } from "./thickness";
import { useUi } from "./ui";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Store = UseBoundStore<StoreApi<any>>;

/** What of each store is saved with an analysis (the views): the state that
 * is about the open data, not jobs, errors or this browser's preferences. */
const PARTS: Record<string, [Store, string[]]> = {
  datasets: [useDatasets, ["active", "workspaces", "selection", "focusId"]],
  si: [useSi, ["bySi", "tool", "cursorSize", "reduce", "showWindows", "showMain"]],
  display: [useDisplay, ["display", "views", "windowColours", "spectrumLog", "lowLossLog", "imageLog"]],
  layers: [useLayers, ["props"]],
  profiles: [useProfiles, ["byView"]],
  fftBoxes: [useFftBoxes, ["byView"]],
  background: [useBackground, ["byId"]],
  deconvolution: [useDeconvolution, ["byId"]],
  thickness: [useThickness, ["byId"]],
  eelsFit: [useEelsFit, ["options", "fits", "sources", "composition"]],
  quant: [useQuant, ["options", "fits", "source", "lineChoice", "excluded", "showFit", "showComponents", "mapComposition", "minCounts"]],
  ui: [useUi, ["foldedNodes", "gridLayouts", "rightTab", "bottomTab", "bottomSplit", "splitDir"]],
};

/** Keys holding one entry per dataset, view or file (their keys contain its
 * id): only the saved ones' entries are saved, and opening adds to them. */
const BY_ID = new Set([
  "datasets.workspaces", "si.bySi", "display.display", "display.views", "display.windowColours",
  "display.imageLog", "layers.props", "profiles.byView", "fftBoxes.byView", "background.byId",
  "deconvolution.byId", "thickness.byId", "eelsFit.options", "eelsFit.fits", "eelsFit.sources",
  "quant.fits", "quant.source", "quant.lineChoice", "quant.excluded", "ui.foldedNodes", "ui.gridLayouts",
]);

const VERSION = 1;

/** The views of the datasets and files ``ids`` (the per-id entries of theirs
 * only, the rest as it is). */
export function viewsSnapshot(ids: Set<string>): Record<string, unknown> {
  const mine = (key: string) => [...ids].some((id) => key.includes(id));
  const out: Record<string, unknown> = { version: VERSION };
  for (const [name, [store, keys]] of Object.entries(PARTS)) {
    const s = store.getState();
    const part = keys.map((k) => {
      const v = s[k];
      if (!BY_ID.has(`${name}.${k}`) || !v) return [k, v];
      return [k, Object.fromEntries(Object.entries(v as Record<string, unknown>).filter(([key]) => mine(key)))];
    });
    out[name] = JSON.parse(JSON.stringify(Object.fromEntries(part)));
  }
  return out;
}

/** Put the saved views of an analysis opened next to what is open: per-id
 * entries added to the ones there, the rest (selection, tabs, tools) as
 * saved. */
function restoreViews(views: Record<string, unknown>) {
  for (const [name, [store, keys]] of Object.entries(PARTS)) {
    const saved = (views[name] ?? {}) as Record<string, unknown>;
    const s = store.getState();
    const patch = keys
      .filter((k) => k in saved)
      .map((k) => [k, BY_ID.has(`${name}.${k}`) ? { ...(s[k] ?? {}), ...(saved[k] as object) } : saved[k]]);
    store.setState(Object.fromEntries(patch));
  }
}

/** What Save analysis saves: the active workspace (an experiment, or a file
 * with the files linked to it), or the only one; its files and a name for it. */
export function workspaceToSave(): { fileIds: string[]; name: string } | null {
  const { files, active } = useDatasets.getState();
  const items = layoutPane(files).items;
  const ws = active ? workspaceOf(files, active) : items.length === 1 ? { files: paneFiles(items[0]) } : null;
  if (!ws?.files.length) return null;
  const first = ws.files[0];
  const name = first.experiment?.name ?? first.name.replace(/\.[^.]+$/, "");
  return { fileIds: ws.files.map((f) => f.id), name };
}

/** Save the files ``fileIds`` (as the estimate gave them: a workspace and
 * what it links to) as an analysis at ``path``. Throws (an ApiError, 409 if it exists and
 * ``overwrite`` isn't set). Returns where it was saved. */
export async function saveAnalysis(
  path: string, fileIds: string[], overwrite = false, level: SaveLevel = "link",
): Promise<string> {
  const files = useDatasets.getState().files.filter((f) => fileIds.includes(f.id));
  const ids = new Set([...fileIds, ...files.flatMap((f) => f.datasets.map((d) => d.id))]);
  const { path: saved } = await api.saveProject(path, viewsSnapshot(ids), overwrite, level, fileIds);
  useLayers.getState().notify(`Saved ${saved.split("/").pop()}`);
  return saved;
}

/** Open the analysis at ``path`` next to what is open, and show it. Throws
 * if it can't (nothing is opened then). */
export async function openAnalysis(path: string, locations: Record<string, string> = {}): Promise<void> {
  const { files, file_ids, views, warnings } = await api.openProject(path, locations);
  const store = useDatasets.getState();
  store.adoptFiles(files);
  if (file_ids[0]) store.enterWorkspace(file_ids[0]);
  restoreViews(views);
  const name = path.split("/").pop();
  useLayers.getState().notify(warnings.length ? `Opened ${name} · ${warnings.join(" · ")}` : `Opened ${name}`);
}
