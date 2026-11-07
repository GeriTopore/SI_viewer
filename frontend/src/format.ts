import type { DatasetKind, DatasetSummary } from "./api/client";

export const KIND_LABEL: Record<DatasetKind, string> = {
  image: "Image",
  stack: "Stack",
  spectrum: "Spectrum",
  spectrum_image: "SI",
  other: "Data",
};

export function kindLabel(d: DatasetSummary): string {
  if (d.origin === "composition") return "at%";
  return d.is_element_map ? "Map" : KIND_LABEL[d.kind];
}

/** Hover text describing a dataset (what an element map contains). */
export function datasetTooltip(d: DatasetSummary): string {
  const base = `${d.title} — ${shapeLabel(d)}`;
  if (d.origin === "computed") return `${base}\nNet counts of ${d.title}, from the raw spectrum image`;
  if (d.origin === "fitted") return `${base}\nIntensity from fitting every pixel (model fit of the summed spectrum)`;
  if (d.origin === "composition") return `${base}\nComposition (at%) per pixel: Cliff-Lorimer on the fitted maps`;
  if (d.is_element_map) return `${base}\nMap stored in the file (may be processed, e.g. quantified or filtered)`;
  return base;
}

/** Shape as HyperSpy prints it: navigation | signal. */
export function shapeLabel(d: DatasetSummary): string {
  const nav = d.navigation_shape.join("×");
  const sig = d.signal_shape.join("×");
  return nav ? `${nav} | ${sig}` : sig;
}

/** Compact number for readouts: integers as is, others to 4 significant digits. */
export function formatValue(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  if (Number.isInteger(v) && Math.abs(v) < 1e7) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(3);
  return Number(v.toPrecision(4)).toString();
}

const LENGTH_UNITS: Record<string, number> = { pm: 1e-3, "Å": 0.1, nm: 1, "µm": 1e3, um: 1e3, mm: 1e6 };
const LENGTH_LADDER = ["pm", "nm", "µm", "mm"];

/** A 1/2/5 × 10ⁿ scale bar about 100 px long. ``unitsPerPixel`` is per CSS px. */
export function niceScaleBar(unitsPerPixel: number, units: string) {
  if (!(unitsPerPixel > 0)) return null;
  const target = 100 * unitsPerPixel;
  const pow = 10 ** Math.floor(Math.log10(target));
  const nice = [5, 2, 1].map((m) => m * pow).find((v) => v <= target) ?? pow;
  const pixels = nice / unitsPerPixel;

  // Re-express lengths in the most readable unit (e.g. 2000 nm → 2 µm).
  let value = nice;
  let unit = units || "px";
  const factor = LENGTH_UNITS[units];
  if (factor !== undefined) {
    const nm = nice * factor;
    unit = LENGTH_LADDER.reduce((best, u) => (nm >= LENGTH_UNITS[u] ? u : best), "pm");
    value = nm / LENGTH_UNITS[unit];
  }
  return { pixels, label: `${Number(value.toPrecision(3))} ${unit}` };
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
