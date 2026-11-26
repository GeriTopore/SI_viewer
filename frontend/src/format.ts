import type { DatasetKind, DatasetSummary, Filters, InsituInfo } from "./api/client";

export const KIND_LABEL: Record<DatasetKind, string> = {
  image: "Image",
  stack: "Stack",
  spectrum: "Spectrum",
  spectrum_image: "SI",
  "4dstem": "4D-STEM",
  other: "Data",
};

/** The spectrum image of an in-situ series: it holds one frame, or a reduction of a frame range. */
export const isInsituView = (d: DatasetSummary) => !!d.insitu?.view;

/** e.g. "128.4 s". */
export const frameTime = (t: number) => `${t.toFixed(1)} s`;

/** What an in-situ series' spectrum image holds, e.g. "frame 37 · 128.4 s" or
 * "mean of frames 5–20 · 74.0–220.3 s" (frames counted from 1). */
export function insituViewLabel(info: InsituInfo): string {
  const v = info.view;
  if (!v) return "";
  const times = info.frame_times_s ?? [];
  if (v.mode === "frame" || v.start === v.end) {
    return `frame ${v.start + 1}${times[v.start] !== undefined ? ` · ${frameTime(times[v.start])}` : ""}`;
  }
  const span = times[v.start] !== undefined && times[v.end] !== undefined
    ? ` · ${times[v.start].toFixed(1)}–${frameTime(times[v.end])}`
    : "";
  return `${v.mode} of frames ${v.start + 1}–${v.end + 1}${span}`;
}

/** A spectrum image with a single navigation axis (positions × energy). */
export const isLineScan = (d: DatasetSummary) => d.kind === "spectrum_image" && d.navigation_shape.length === 1;

/** Spectra and spectrum images (including line scans), which have a signal type. */
export const hasTechnique = (d: DatasetSummary) =>
  (d.kind === "spectrum" || d.kind === "spectrum_image") && !d.is_element_map && d.origin !== "thickness" && d.origin !== "zlp-shift";

/** A map of an energy window kept from a spectrum image. */
export const isWindowMap = (d: DatasetSummary) => d.origin === "window";

/** A spectrum image made by background removal. */
export const isBackgroundRemoved = (d: DatasetSummary) => d.origin === "background";

/** EELS spectra with plural scattering removed (or sharpened). */
export const isDeconvolved = (d: DatasetSummary) => d.origin === "deconvolved";

/** The thickness of a low loss (a map, or a profile along a line scan). */
export const isThickness = (d: DatasetSummary) => d.origin === "thickness";

/** The energy shift zero-loss alignment applied at every pixel of a low loss (a map, or a profile along a line scan). */
export const isZlpShift = (d: DatasetSummary) => d.origin === "zlp-shift";

const ATOMIC_LABEL: Record<NonNullable<DatasetSummary["atomic"]>["output"], string> = {
  amplitude: "Amp", grating: "Grating", phase: "Phase", "strain-par": "ε∥", "strain-perp": "ε⊥",
  exx: "εxx", eyy: "εyy", exy: "εxy", oxy: "ωxy", columns: "Columns", sublattice: "Sublattice",
  neighbourhood: "Neighbours", bonds: "Bonds", "pld-reference": "PLD ref", "pld-difference": "PLD diff",
  displacements: "PLD",
};

export function kindLabel(d: DatasetSummary): string {
  if (isWindowMap(d)) return "Window";
  if (d.origin === "pca") return "Decomp";
  if (d.origin === "pca-factors") return "Factors";
  if (d.dpc?.output === "colour") return "DPC";
  if (d.dpc?.output === "idpc") return "iDPC";
  if (isBackgroundRemoved(d)) return "BG sub";
  if (isDeconvolved(d)) return "Deconv";
  if (isThickness(d)) return d.thickness?.output === "absolute" ? "nm" : "t/λ";
  if (isZlpShift(d)) return "eV";
  if (d.survey) return "Survey";
  if (d.origin === "curve-model") return "Model";
  if (d.atomic?.kind === "cage") return "Disp";
  if (d.atomic) return ATOMIC_LABEL[d.atomic.output];
  if (d.origin === "mask") return "Mask";
  if (d.origin === "disks") return "Bragg";
  if (d.origin === "bragg-map") return "BVM";
  if (d.origin === "beam-shift") return "Shift";
  if (d.origin === "origin-map") return "Origin";
  if (d.origin === "fitted" || d.origin === "curve-fit") return "Fit";
  const t = hasTechnique(d) ? d.technique : null;
  // EELS spectrum images say whether they are low loss or core loss.
  const role = d.eels ? (d.eels.role === "low_loss" ? "LL" : "HL") : null;
  if (isInsituView(d)) return `IS ${role ?? t ?? "SI"}`;
  if (isLineScan(d)) return "LineScan";
  if (t) return d.kind === "spectrum" ? t : role ? `${t} ${role}` : `${t} SI`;
  return d.is_element_map ? "Map" : KIND_LABEL[d.kind];
}

/** e.g. "EELS spectrum image" (the technique is left out when unknown). */
export function kindName(d: DatasetSummary): string {
  const t = hasTechnique(d) && d.technique ? `${d.technique} ` : "";
  const name = isLineScan(d)
    ? "LineScan"
    : isInsituView(d)
      ? "in-situ spectrum image"
      : ({ image: "image", stack: "stack", spectrum: "spectrum", spectrum_image: "spectrum image", "4dstem": "4D-STEM data", other: "data" } as const)[
          d.kind
        ];
  return t ? t + name : name[0].toUpperCase() + name.slice(1);
}

/** Hover text describing a dataset (what an element map contains). */
export function datasetTooltip(d: DatasetSummary, parentTitle?: string): string {
  const filters = filterLabel(d.filters);
  const t = hasTechnique(d) && d.technique ? d.technique : "";
  const role = d.eels ? ` (${d.eels.role === "low_loss" ? "low loss" : "core loss"})` : "";
  const scan = isLineScan(d)
    ? `\n${t ? `${t} LineScan` : "LineScan"}${role}, ${d.navigation_shape[0]} points × ${d.signal_shape.join("×")} channels`
    : t && !d.is_element_map
      ? `\n${kindName(d)}${role}, ${d.navigation_shape.concat(d.signal_shape).join(" × ")}`
      : "";
  const base = `${d.title} — ${shapeLabel(d)}${scan}${filters ? `\nFiltered: ${filters}` : ""}`;
  if (isWindowMap(d)) {
    const w = d.window;
    const range = w ? `${formatValue(w.lo)}–${formatValue(w.hi)} ${w.units}` : d.title;
    return `${base}\nEnergy window ${range}${parentTitle ? ` of ${parentTitle}` : ""}`;
  }
  if (isBackgroundRemoved(d)) {
    const b = d.background;
    const what = b
      ? `${b.model.replace("_", " ")}${b.model === "polynomial" ? ` (order ${b.order})` : ""}, fitted on ` +
        `${formatValue(b.background[0])}–${formatValue(b.background[1])} ${b.units}`
      : "background removed";
    return `${base}\nBackground removed${parentTitle ? ` from ${parentTitle}` : ""}: ${what}`;
  }
  if (d.origin === "line-scan") {
    const overTime = d.axes.find((a) => a.navigate)?.name === "Time";
    return `${base}\nLineScan ${overTime ? "over the frames" : "along a line"}${parentTitle ? ` of ${parentTitle}` : ""}`;
  }
  if (isDeconvolved(d)) {
    const c = d.deconvolution;
    const how = !c
      ? ""
      : c.method === "fourier_log"
        ? "Fourier-log"
        : c.method === "fourier_ratio"
          ? "Fourier-ratio"
          : `Richardson–Lucy (${c.psf === "low_loss" ? "low loss" : "zero-loss peak"}, ${c.iterations} iterations)`;
    return `${base}
Deconvolved${parentTitle ? ` from ${parentTitle}` : ""}${how ? `: ${how}` : ""}`;
  }
  if (isThickness(d)) {
    const t = d.thickness;
    const threshold = !t ? "" : t.threshold === "auto" ? "threshold estimated" : `threshold ${formatValue(t.threshold)} ${t.units}`;
    const given = !t ? "" : t.mean_free_path != null ? `, λ ${formatValue(t.mean_free_path)} nm` : t.density != null ? `, density ${formatValue(t.density)} g/cm³` : "";
    return `${base}\nThickness of ${parentTitle ?? "its low loss"}${t ? `: ${threshold}${given}` : ""}`;
  }
  if (isZlpShift(d)) return `${base}\nEnergy shift applied by ZLP alignment${parentTitle ? ` to ${parentTitle}` : ""}`;
  if (d.origin === "background-r") return `${base}\nPower-law exponent r of the background fit at each pixel`;
  if (d.origin === "computed") return `${base}\nNet counts of ${d.title}, from the raw spectrum image`;
  if (d.origin === "curve-model") return `${base}\nThe curve fit of every pixel: the model at each pixel, its maps under it`;
  if (d.mask) {
    const m = d.mask;
    const f = (v: number | null) => (v == null ? "–" : formatValue(v));
    const rule = m.keep === "above" ? `≥ ${f(m.lo)}` : m.keep === "below" ? `≤ ${f(m.hi)}` : `${f(m.lo)} – ${f(m.hi)}`;
    const extra = [m.smooth ? `smoothed σ ${m.smooth} px` : "", m.invert ? "inverted" : "", m.clean ? `cleaned ${m.clean} px` : "", m.min_area ? `regions ≥ ${m.min_area} px` : ""].filter(Boolean);
    return `${base}\nPixels of ${parentTitle ?? "its image"} ${rule}${extra.length ? `, ${extra.join(", ")}` : ""}: ${formatValue(m.fraction * 100)} % of the image`;
  }
  if (d.atomic?.output === "columns") {
    const a = d.atomic;
    return `${d.title}\n${a.count} atomic columns fitted with 2D Gaussians (kemstem)${a.failed ? `, ${a.failed} failed` : ""}: window ${a.window} px, blur σ ${a.blur} px; shown over ${parentTitle ?? "its image"}`;
  }
  if (d.atomic?.output === "pld-reference" || d.atomic?.output === "pld-difference") {
    const a = d.atomic;
    return `${d.title}\n${a.output === "pld-reference" ? "The image with" : "Reference less the image, with"} ${a.damped?.length ?? 0} superlattice peaks damped within ${a.radius} FFT px (kemstem)`;
  }
  if (d.atomic?.output === "displacements") {
    const a = d.atomic;
    if (a.kind === "cage") {
      const cage = a.nearest ? "its nearest neighbours" : `${a.cage?.length ?? 0} picked neighbours`;
      return `${d.title}\nEach of ${parentTitle ?? "the neighbourhood"}'s origins less the centre of its cage of ${cage} (within ${a.threshold} px)${a.missing ? `; ${a.missing} without` : ""}`;
    }
    return `${d.title}\nPeriodic lattice displacement of each column: fitted position less its position in the PLD reference${a.missing ? `; ${a.missing} without` : ""} (kemstem)`;
  }
  if (d.atomic?.output === "neighbourhood") {
    const a = d.atomic;
    return `${d.title}\nVectors from each of ${a.count} columns to their ${a.k} nearest${a.clusters ? `; ${a.clusters.length} clusters` : ""} (kemstem)`;
  }
  if (d.atomic?.output === "bonds") {
    const a = d.atomic;
    return `${d.title}\nFrom each origin, the vector to the neighbour nearest (${a.vector?.map((v) => formatValue(v)).join(", ")}) px, within ${a.threshold} px${a.missing ? `; ${a.missing} without one` : ""} (kemstem)`;
  }
  if (d.atomic?.output === "sublattice") {
    return `${d.title}\n${d.atomic.count} of ${parentTitle ?? "the columns"}${d.atomic.mask ? ", those in a mask" : ""}; shown over their image`;
  }
  if (d.atomic) {
    const a = d.atomic;
    const how = a.output === "amplitude" || a.output === "grating" ? "" : a.phase === "gpa" ? ", GPA" : `, lock-in σ ${formatValue(a.lock_sigma)}`;
    return `${base}\nFourier filtered (kemstem) round peak${a.peaks.length > 1 ? "s" : ""} ${a.peaks.join(", ")}: σ ${formatValue(a.sigma)} px${how}; coarsening ${formatValue(a.coarsening)} px`;
  }
  if (d.origin === "curve-fit") {
    return d.title === "Reduced χ²"
      ? `${base}\nReduced χ² of the curve fit at each pixel (the parameters' maps under it)`
      : `${base}\nCurve fit of every pixel`;
  }
  if (d.origin === "fitted") {
    return d.quantity === "at%"
      ? `${base}\nComposition (at%) per pixel from the fitted maps`
      : `${base}\nIntensity from fitting every pixel (model fit of the summed spectrum)`;
  }
  if (d.is_element_map) return `${base}\nMap stored in the file (may be processed, e.g. quantified or filtered)`;
  if (d.origin === "aligned-average") return `${base}\nAverage of the stack's frames after drift correction`;
  if (d.alignment?.active) {
    return `${base}\nDrift-corrected: ${d.alignment.frames.length} of ${d.alignment.loaded_frames} frames, cropped`;
  }
  if (d.insitu?.view) return `${base}\nIn-situ series, showing ${insituViewLabel(d.insitu)}`;
  return base;
}

// Lengths are shown in ångström; these units are converted (others kept as is).
const TO_ANGSTROM: Record<string, number> = {
  m: 1e10, mm: 1e7, µm: 1e4, μm: 1e4, um: 1e4, nm: 10, pm: 0.01, Å: 1, A: 1, angstrom: 1,
};

/** A pixel size in ångström when its units are a length (else as is). */
export function inAngstrom(scale: number, units: string): { size: number; units: string } {
  const f = TO_ANGSTROM[units.trim()];
  return f ? { size: scale * f, units: "Å" } : { size: scale, units };
}

/** e.g. "Gaussian σ 1.5 px; Savitzky–Golay 7 ch, order 2" ("" when unfiltered). */
export function filterLabel(f: Filters | undefined): string {
  if (!f) return "";
  const parts: string[] = [];
  if (f.spatial === "gaussian") parts.push(`Gaussian σ ${f.sigma} px`);
  else if (f.spatial === "median" || f.spatial === "mean")
    parts.push(`${f.spatial === "median" ? "Median" : "Mean"} ${f.size}×${f.size} px`);
  else if (f.spatial === "bandpass") parts.push(`Bandpass ${f.small}–${f.large} px`);
  if (f.spectral === "gaussian") parts.push(`spectral Gaussian σ ${f.spectral_sigma} ch`);
  else if (f.spectral === "savgol") parts.push(`Savitzky–Golay ${f.window} ch, order ${f.order}`);
  return parts.join("; ");
}

/** Shape as HyperSpy prints it: navigation | signal. */
export function shapeLabel(d: DatasetSummary): string {
  // In-situ series: frames × frame (x×y) | channels, short for the dataset list, e.g. "208×100×100|3456".
  if (d.insitu?.view) {
    const [h, w] = d.insitu.frame_shape;
    return `${d.insitu.frames}×${w}×${h}|${d.signal_shape.join("×")}`;
  }
  const nav = d.navigation_shape.join("×");
  const sig = d.signal_shape.join("×");
  // Ragged (e.g. diffraction vectors: a list at each position): navigation only.
  return nav && sig ? `${nav} | ${sig}` : nav || sig;
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
  return { pixels, length: nice, label: scaleBarLabel(nice, units) };
}

/** The label of a scale bar ``length`` long in ``units``, re-expressed in the
 * most readable unit (e.g. 2000 nm → "2 µm"). */
export function scaleBarLabel(length: number, units: string): string {
  let value = length;
  let unit = units || "px";
  const factor = LENGTH_UNITS[units];
  if (factor !== undefined) {
    const nm = length * factor;
    unit = LENGTH_LADDER.reduce((best, u) => (nm >= LENGTH_UNITS[u] ? u : best), "pm");
    value = nm / LENGTH_UNITS[unit];
  }
  return `${Number(value.toPrecision(3))} ${unit}`;
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
