import type { DatasetKind, DatasetSummary } from "./api/client";

export const KIND_LABEL: Record<DatasetKind, string> = {
  image: "Image",
  stack: "Stack",
  spectrum: "Spectrum",
  spectrum_image: "SI",
  other: "Data",
};

export function kindLabel(d: DatasetSummary): string {
  return d.is_element_map ? "Map" : KIND_LABEL[d.kind];
}

/** Shape as HyperSpy prints it: navigation | signal. */
export function shapeLabel(d: DatasetSummary): string {
  const nav = d.navigation_shape.join("×");
  const sig = d.signal_shape.join("×");
  return nav ? `${nav} | ${sig}` : sig;
}

export function bytesLabel(n: number): string {
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}
