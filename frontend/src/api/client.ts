// Typed client for the FastAPI backend. Mirrors si_viewer/session.py summaries.

export type DatasetKind = "image" | "stack" | "spectrum" | "spectrum_image" | "other";

export interface Axis {
  name: string;
  size: number;
  scale: number;
  offset: number;
  units: string;
  navigate: boolean;
}

/** How a spectrum image's element maps are computed. */
export interface MapOptions {
  background: boolean;
  /** Integration window width in FWHM units. */
  width: number;
  /** n × n box smoothing (1 = none). */
  smoothing: number;
}

export interface EdsInfo {
  elements: string[];
  xray_lines: string[];
  beam_energy: number | null;
  energy_resolution_MnKa: number | null;
  /** Spectrum images only. */
  map_options?: MapOptions;
}

export interface DatasetSummary {
  id: string;
  file_id: string;
  title: string;
  kind: DatasetKind;
  signal_type: string;
  is_element_map: boolean;
  navigation_shape: number[];
  signal_shape: number[];
  dtype: string;
  nbytes: number;
  lazy: boolean;
  axes: Axis[];
  /** Present for EDS spectra and spectrum images. */
  eds: EdsInfo | null;
  /** "computed": an element map made from a spectrum image (window +
   * background); "fitted"/"composition": from fitting every pixel. */
  origin: "file" | "computed" | "fitted" | "composition";
  /** Element symbol of an element map. */
  element: string | null;
  /** Spectrum image a computed map was made from. */
  parent_id: string | null;
  /** Bumped when the data is recomputed or rebinned. */
  revision: number;
  /** Rebinning applied to the loaded data, e.g. ["x×2, y×2"]. */
  rebinned: string[];
  /** Non-destructive filters applied whenever the data is shown. */
  filters: Filters;
  /** Bumped whenever the filters change. */
  filter_rev: number;
  /** Line of a computed map, e.g. "Fe_Ka". */
  xray_line: string | null;
}

export type SpatialFilter = "none" | "gaussian" | "median" | "mean";
export type SpectralFilter = "none" | "gaussian" | "savgol";

/** Pre-processing filters of a dataset (see backend processing/filters.py). */
export interface Filters {
  spatial: SpatialFilter;
  /** Gaussian σ (pixels). */
  sigma: number;
  /** Median / mean window (pixels, odd). */
  size: number;
  spectral: SpectralFilter;
  /** Gaussian σ (channels). */
  spectral_sigma: number;
  /** Savitzky–Golay window (channels, odd) and order. */
  window: number;
  order: number;
}

/** Changes whenever the data shown for a dataset does (recomputed, rebinned
 * or filtered), for cache keys. */
export const dataVersion = (d: { revision: number; filter_rev: number }) => `${d.revision}.${d.filter_rev}`;

/** Options of a spectrum fit (exSpy EDS model). */
export interface FitOptions {
  background_order: number;
  /** Channels below this energy (keV) are ignored. */
  min_energy: number;
  calibrate_resolution: boolean;
  calibrate_energy: boolean;
}

/** Which spectrum of a spectrum image to fit. */
export type FitSource =
  | { type: "all" }
  | { type: "pixel"; x: number; y: number; size: number }
  | { type: "region"; region: object };

export interface FitLine {
  line: string;
  element: string;
  energy: number;
  intensity: number;
  /** exSpy's choice of line for the element. */
  analysis: boolean;
}

export interface FitResult {
  x: number[];
  data: number[];
  model: number[];
  background: number[];
  /** Each element's fitted contribution. */
  elements: Record<string, number[]>;
  lines: FitLine[];
  resolution_MnKa: number | null;
  residual_ratio: number | null;
  /** Pixels summed into the fitted spectrum. */
  pixels: number;
  /** Elements with no line in the fitted range. */
  skipped: string[];
}

/** A dataset change that may also have changed its file's maps. */
export interface DatasetAndFile {
  dataset: DatasetSummary;
  file: FileSummary;
  warnings?: string[];
}

export interface FileSummary {
  id: string;
  path: string;
  name: string;
  datasets: DatasetSummary[];
}

export interface FsEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size: number | null;
  modified: number;
}

export interface FsListing {
  path: string;
  parent: string | null;
  entries: FsEntry[];
}

export interface NavigatorSource {
  id: string;
  label: string;
}

export interface AxisCalibration {
  index: number;
  scale?: number;
  offset?: number;
  units?: string;
}

/** A background job (e.g. fitting every pixel). */
export interface JobSummary {
  id: string;
  kind: string;
  title: string;
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  progress: number;
  message: string;
  error: string | null;
  result: { file: FileSummary; skipped?: string[] } | null;
}

/** Composition maps made alongside per-pixel fitted maps. */
export interface QuantSpec {
  composition: boolean;
  kfactors: Record<string, number>;
  lines: Record<string, string>;
  exclude: string[];
  /** Pixels with fewer fitted element counts get no composition. */
  min_counts: number;
}

/** A bundled example dataset (offline). */
export interface ExampleInfo {
  id: string;
  title: string;
  description: string;
}

export interface Health {
  status: string;
  versions: Record<string, string>;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return (res.status === 204 ? undefined : await res.json()) as T;
}

export const api = {
  health: () => request<Health>("/api/health"),
  listFiles: () => request<FileSummary[]>("/api/files"),
  openFile: (path: string) =>
    request<FileSummary>("/api/files", { method: "POST", body: JSON.stringify({ path }) }),
  closeFile: (id: string) => request<void>(`/api/files/${id}`, { method: "DELETE" }),
  getDataset: (id: string) => request<DatasetSummary>(`/api/datasets/${id}`),
  loadDataset: (id: string) => request<DatasetSummary>(`/api/datasets/${id}/load`, { method: "POST" }),
  navigators: (id: string) => request<NavigatorSource[]>(`/api/datasets/${id}/navigators`),
  calibrate: (id: string, axes: AxisCalibration[]) =>
    request<DatasetSummary>(`/api/datasets/${id}/axes`, { method: "PATCH", body: JSON.stringify(axes) }),
  setElements: (id: string, elements: string[]) =>
    request<DatasetAndFile>(`/api/datasets/${id}/elements`, { method: "PUT", body: JSON.stringify({ elements }) }),
  setMapOptions: (id: string, options: Partial<MapOptions>) =>
    request<DatasetAndFile>(`/api/datasets/${id}/map-options`, { method: "PUT", body: JSON.stringify(options) }),
  deleteDataset: (id: string) => request<{ file: FileSummary }>(`/api/datasets/${id}`, { method: "DELETE" }),
  fit: (id: string, source: FitSource, options: FitOptions) =>
    request<FitResult>(`/api/datasets/${id}/fit`, { method: "POST", body: JSON.stringify({ source, options }) }),
  /** One integer factor per axis, in the summary's axis order. */
  rebin: (id: string, factors: number[]) =>
    request<DatasetAndFile>(`/api/datasets/${id}/rebin`, { method: "POST", body: JSON.stringify({ factors }) }),
  setFilters: (id: string, filters: Filters) =>
    request<DatasetAndFile & { notes: string[] }>(`/api/datasets/${id}/filters`, {
      method: "PUT",
      body: JSON.stringify(filters),
    }),
  startFitMaps: (id: string, options: FitOptions, quant: QuantSpec) =>
    request<JobSummary>(`/api/datasets/${id}/fit-maps`, { method: "POST", body: JSON.stringify({ options, quant }) }),
  getJob: (id: string) => request<JobSummary>(`/api/jobs/${id}`),
  cancelJob: (id: string) => request<JobSummary>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  listExamples: () => request<ExampleInfo[]>("/api/examples"),
  openExample: (id: string) => request<FileSummary>(`/api/examples/${id}`, { method: "POST" }),
  listDir: (path?: string) =>
    request<FsListing>(`/api/fs/list${path ? `?path=${encodeURIComponent(path)}` : ""}`),
};

/** Thumbnail URL; the version makes recomputed or filtered data bypass the browser cache. */
export const thumbnailUrl = (d: { id: string; revision: number; filter_rev: number }, size = 256) =>
  `/api/datasets/${d.id}/thumbnail?size=${size}&v=${dataVersion(d)}`;
