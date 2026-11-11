import { useEffect, useRef, useState } from "react";
import { api, type DroppedItem } from "../api/client";
import { useDatasets } from "../state/datasets";
import { useLayers } from "../state/layers";

// Files of a dropped folder sent along to tell it from others of its name.
const FOLDER_SAMPLE = 6;

const hasFiles = (e: DragEvent) => !!e.dataTransfer && Array.from(e.dataTransfer.types).includes("Files");

const fileOf = (entry: FileSystemFileEntry) => new Promise<File>((resolve, reject) => entry.file(resolve, reject));
const entriesOf = (dir: FileSystemDirectoryEntry) =>
  new Promise<FileSystemEntry[]>((resolve, reject) => dir.createReader().readEntries(resolve, reject));

/** What the backend needs to find a dropped file or folder on disk (a browser
 * never gives its path): the name, and the size and time of a file or some of
 * the files inside a folder. */
async function describe(entry: FileSystemEntry | null, file: File | null): Promise<DroppedItem | null> {
  if (entry?.isDirectory) {
    const inside = (await entriesOf(entry as FileSystemDirectoryEntry).catch(() => []))
      .filter((e) => e.isFile && !e.name.startsWith("."))
      .slice(0, FOLDER_SAMPLE);
    const files = await Promise.all(inside.map((e) => fileOf(e as FileSystemFileEntry).catch(() => null)));
    return {
      name: entry.name,
      is_dir: true,
      children: files.filter((f): f is File => !!f).map((f) => ({ name: f.name, size: f.size })),
    };
  }
  const f = file ?? (entry?.isFile ? await fileOf(entry as FileSystemFileEntry).catch(() => null) : null);
  return f ? { name: f.name, is_dir: false, size: f.size, modified: f.lastModified, children: [] } : null;
}

const parentOf = (path: string) => path.replace(/[\\/][^\\/]*$/, "");

/** Files and folders dropped anywhere on the window are opened: a file like
 * one picked in Open…, a folder as an experiment (Open all in folder). */
export function DropZone() {
  const [over, setOver] = useState(false);
  const [finding, setFinding] = useState(false);
  const depth = useRef(0); // dragenter / dragleave fire for every element crossed

  useEffect(() => {
    const onEnter = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth.current += 1;
      setOver(true);
    };
    const onOver = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      e.dataTransfer!.dropEffect = "copy";
    };
    const onLeave = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth.current = Math.max(0, depth.current - 1);
      if (depth.current === 0) setOver(false);
    };
    const onDrop = async (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth.current = 0;
      setOver(false);
      // The entries must be taken before the first await: the items are gone after it.
      const dropped = Array.from(e.dataTransfer!.items)
        .filter((i) => i.kind === "file")
        .map((i) => ({ entry: i.webkitGetAsEntry?.() ?? null, file: i.getAsFile() }));
      const notify = useLayers.getState().notify;
      setFinding(true);
      try {
        const items = (await Promise.all(dropped.map((d) => describe(d.entry, d.file)))).filter((i): i is DroppedItem => !!i);
        if (!items.length) {
          // The browser gave no readable file (e.g. a drag it doesn't pass on).
          notify("Could not read what was dropped. Use Open…");
          return;
        }
        const { files, openFiles, openFolder } = useDatasets.getState();
        const hints = [...new Set(files.flatMap((f) => [parentOf(f.path), ...(f.experiment ? [f.experiment.path] : [])]))];
        const { paths } = await api.locate(items, hints);
        const missing = items.filter((_, i) => !paths[i]).map((i) => i.name);
        if (missing.length) notify(`Not found on disk: ${missing.join(", ")}. Use Open…`);
        setFinding(false);
        const filePaths = paths.filter((p, i): p is string => !!p && !items[i].is_dir);
        if (filePaths.length) await openFiles(filePaths);
        for (const [i, p] of paths.entries()) if (p && items[i].is_dir) await openFolder(p);
      } catch (err) {
        notify(`Could not open what was dropped: ${(err as Error).message}`);
      } finally {
        setFinding(false);
      }
    };
    window.addEventListener("dragenter", onEnter);
    window.addEventListener("dragover", onOver);
    window.addEventListener("dragleave", onLeave);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragenter", onEnter);
      window.removeEventListener("dragover", onOver);
      window.removeEventListener("dragleave", onLeave);
      window.removeEventListener("drop", onDrop);
    };
  }, []);

  if (!over && !finding) return null;
  return (
    <div className="drop-zone" aria-live="polite">
      <div className="drop-zone-box">{finding ? "Finding on disk…" : "Drop to open"}</div>
    </div>
  );
}
