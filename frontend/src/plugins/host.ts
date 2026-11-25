import React, { useMemo, type ReactNode } from "react";
import { create } from "zustand";
import { api, request, type DatasetSummary, type FileSummary, type JobSummary } from "../api/client";
import type { PaneTab } from "../components/BottomPane";
import { allDatasets, initialColormap, useDatasets } from "../state/datasets";
import { useDisplay } from "../state/display";
import { useLayers } from "../state/layers";
import { getViewProfiles, useProfiles, useViewProfiles, type LineGeometry, type ProfileLine } from "../state/profiles";
import { colormapLut } from "../viewer/colormaps";
import { useUi, type BottomTab, type BrowserPick } from "../state/ui";

/** A process a plugin adds to the Process tab. */
export interface PluginProcess {
  id: string;
  title: string;
  tip: string;
  section?: string;
  summary?: string | null;
  applies: (d: DatasetSummary) => boolean;
  render: (d: DatasetSummary) => ReactNode;
}

/** A tab a plugin adds to the pane under the image. */
export interface PluginBottomTab {
  id: string;
  label: string;
  applies: (d: DatasetSummary) => boolean;
  render: (d: DatasetSummary) => ReactNode;
}

/** A button a plugin adds to the top bar. */
export interface PluginCommand {
  id: string;
  label: string;
  title?: string;
  run: () => Promise<void> | void;
}

/** What the Process tab shows for a plugin's project with nothing selected. */
export interface PluginProjectPanel {
  type: string;
  render: (file: FileSummary) => ReactNode;
}

/** How the middle column shows a plugin's tree item: one tab. */
export interface PluginItemView {
  item: string;
  label: string;
  render: (d: DatasetSummary) => ReactNode;
}

export interface PluginInfo {
  id: string;
  name: string;
  version: string | null;
  status: "active" | "failed" | "incompatible";
  error: string | null;
  frontend: string | null;
}

/** What a plugin's ``activate(host)`` is given. */
export interface PluginHost {
  /** The app's React: a plugin must not bundle its own. */
  React: typeof React;
  addProcess: (process: PluginProcess) => void;
  addBottomTab: (tab: PluginBottomTab) => void;
  addCommand: (command: PluginCommand) => void;
  addProjectPanel: (panel: PluginProjectPanel) => void;
  addItemView: (view: PluginItemView) => void;
  /** Ask for a file or a new folder in the file browser (null: cancelled). */
  pickPath: (pick: Omit<BrowserPick, "resolve">) => Promise<string | null>;
  /** Show ``file`` (added if new) and, if given, one of its datasets. */
  showFile: (file: FileSummary, datasetId?: string) => void;
  /** Take a file's new summary (e.g. after a plugin changed it), changing nothing shown. */
  updateFile: (file: FileSummary) => void;
  request: typeof request;
  /** POST ``body`` to a route that starts a job, and follow it to its end. */
  runJob: (path: string, body: unknown, onUpdate?: (job: JobSummary) => void) => Promise<JobSummary>;
  /** Follow a job already started (for example, after a page reload). */
  followJob: (job: JobSummary, onUpdate?: (job: JobSummary) => void) => Promise<JobSummary>;
  cancelJob: (id: string) => Promise<JobSummary>;
  /** The profile lines drawn on a view (a stack's view key is its id), and the selected one. */
  useLines: (view: string) => { lines: ProfileLine[]; selected: string | null };
  /** The same, read once (outside rendering). */
  getLines: (view: string) => { lines: ProfileLine[]; selected: string | null };
  /** Draw a profile line on a view and select it (``fixed``: along an axis,
   * moving only across it, never deleted). Returns its id. */
  addLine: (view: string, geometry: LineGeometry, extra?: Partial<Pick<ProfileLine, "name" | "fixed" | "visible">>) => string;
  /** Change a line (its visibility, for example). */
  updateLine: (view: string, id: string, patch: Partial<ProfileLine>) => void;
  selectLine: (view: string, id: string) => void;
  /** The colormap a dataset's view uses, and colormaps as 256 × RGBA8 lookup tables. */
  useColormap: (d: DatasetSummary) => string;
  colormapLut: (name: string) => Uint8Array;
}

const useLines = (view: string) => {
  const { lines, selected } = useViewProfiles(view);
  return { lines, selected };
};

function useColormap(d: DatasetSummary) {
  const chosen = useDisplay((s) => (["frame", "sum", "mean"] as const).map((p) => s.display[`${d.id}:${p}`]?.colormap).find(Boolean));
  return chosen ?? initialColormap(d);
}

interface PluginsState {
  plugins: PluginInfo[];
  processes: PluginProcess[];
  bottomTabs: PluginBottomTab[];
  commands: PluginCommand[];
  projectPanels: (PluginProjectPanel & { plugin: string })[];
  itemViews: (PluginItemView & { plugin: string })[];
}

export const usePlugins = create<PluginsState>(() => ({
  plugins: [], processes: [], bottomTabs: [], commands: [], projectPanels: [], itemViews: [],
}));

const pickPath = (pick: Omit<BrowserPick, "resolve">) =>
  new Promise<string | null>((resolve) => {
    useUi.setState({ browserOpen: true, browserMode: "pick", browserPick: { ...pick, resolve } });
  });

function showFile(file: FileSummary, datasetId?: string) {
  const store = useDatasets.getState();
  const known = store.files.some((f) => f.id === file.id);
  store.adoptFiles(known ? store.files.map((f) => (f.id === file.id ? file : f)) : [...store.files, file]);
  useDatasets.getState().enterWorkspace(file.id);
  const d = datasetId ? file.datasets.find((x) => x.id === datasetId) : undefined;
  if (d) useDatasets.getState().showOnly(d);
  useUi.getState().setRightTab("process");
}

/** Run a plugin's command; what goes wrong is shown as a message. */
export async function runCommand(command: PluginCommand) {
  try {
    await command.run();
  } catch (e) {
    useLayers.getState().notify(`${command.label.replace(/…$/, "")}: ${(e as Error).message}`);
  }
}

const POLL_MS = 200;
const isActive = (job: JobSummary) => job.status === "queued" || job.status === "running";

async function runJob(path: string, body: unknown, onUpdate?: (job: JobSummary) => void) {
  const job = await request<JobSummary>(path, { method: "POST", body: JSON.stringify(body) });
  return followJob(job, onUpdate);
}

/** Follow a job to its end; a result naming a file updates that file. */
async function followJob(started: JobSummary, onUpdate?: (job: JobSummary) => void) {
  let job = started;
  onUpdate?.(job);
  while (isActive(job)) {
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
    job = await api.getJob(job.id);
    onUpdate?.(job);
  }
  const file = (job.result as { file?: FileSummary } | null | undefined)?.file;
  if (file) useDatasets.getState().replaceFile(file);
  return job;
}

/** Loads each active plugin's frontend. One that fails adds nothing. */
export async function loadPlugins() {
  let plugins: PluginInfo[];
  try {
    plugins = await request<PluginInfo[]>("/api/plugins");
  } catch (e) {
    console.error("Could not list the plugins", e);
    return;
  }
  usePlugins.setState({ plugins });
  for (const plugin of plugins) {
    if (plugin.status !== "active" || !plugin.frontend) continue;
    const processes: PluginProcess[] = [];
    const bottomTabs: PluginBottomTab[] = [];
    const commands: PluginCommand[] = [];
    const projectPanels: PluginsState["projectPanels"] = [];
    const itemViews: PluginsState["itemViews"] = [];
    try {
      const module = await import(/* @vite-ignore */ plugin.frontend);
      await module.activate({
        React,
        addProcess: (p: PluginProcess) => processes.push({ ...p, id: `plugin:${plugin.id}:${p.id}` }),
        addBottomTab: (t: PluginBottomTab) => bottomTabs.push({ ...t, id: `plugin:${plugin.id}:${t.id}` }),
        addCommand: (c: PluginCommand) => commands.push({ ...c, id: `plugin:${plugin.id}:${c.id}` }),
        addProjectPanel: (p: PluginProjectPanel) => projectPanels.push({ ...p, plugin: plugin.id }),
        addItemView: (v: PluginItemView) => itemViews.push({ ...v, plugin: plugin.id }),
        pickPath,
        showFile,
        updateFile: (file) => useDatasets.getState().replaceFile(file),
        request,
        runJob,
        followJob,
        cancelJob: api.cancelJob,
        useLines,
        getLines: (view) => {
          const { lines, selected } = getViewProfiles(view);
          return { lines, selected };
        },
        addLine: (view, geometry, extra) => useProfiles.getState().addLine(view, geometry, extra),
        updateLine: (view, id, patch) => useProfiles.getState().updateLine(view, id, patch),
        selectLine: (view, id) => useProfiles.getState().patch(view, { selected: id }),
        useColormap,
        colormapLut,
      } satisfies PluginHost);
    } catch (e) {
      console.error(`Plugin ${plugin.id} failed to load`, e);
      continue;
    }
    usePlugins.setState((s) => ({
      processes: [...s.processes, ...processes],
      bottomTabs: [...s.bottomTabs, ...bottomTabs],
      commands: [...s.commands, ...commands],
      projectPanels: [...s.projectPanels, ...projectPanels],
      itemViews: [...s.itemViews, ...itemViews],
    }));
  }
}

/** The Process tab's content for the plugin project being viewed, if any. */
export function usePluginProjectPanel(): ReactNode {
  const panels = usePlugins((s) => s.projectPanels);
  const file = useDatasets((s) => s.files.find((f) => f.id === s.active));
  if (!file?.plugin) return null;
  const { id, type } = file.plugin;
  return panels.find((p) => p.plugin === id && p.type === type)?.render(file) ?? null;
}

/** The plugins' processes that apply to ``d``. */
export function usePluginProcesses(d: DatasetSummary) {
  const processes = usePlugins((s) => s.processes);
  return processes
    .filter((p) => p.applies(d))
    .map(({ id, title, tip, section, summary, render }) => ({ id, title, tip, section, summary, body: render(d) }));
}

/** The plugins' bottom tabs for the focused dataset. */
export function usePluginTabs(): PaneTab[] {
  const tabs = usePlugins((s) => s.bottomTabs);
  const files = useDatasets((s) => s.files);
  const focusId = useDatasets((s) => s.focusId);
  const focused = useMemo(
    () => (tabs.length && focusId ? allDatasets(files).find((d) => d.id === focusId) : undefined),
    [tabs.length, files, focusId],
  );
  if (!focused) return [];
  return tabs
    .filter((t) => t.applies(focused))
    .map((t) => ({ id: t.id as BottomTab, label: t.label, content: t.render(focused) }));
}
