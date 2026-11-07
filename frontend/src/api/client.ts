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
  /** "computed": an element map made from a spectrum image. */
  origin: "file" | "computed";
  /** Element symbol of an element map. */
  element: string | null;
  /** Spectrum image a computed map was made from. */
  parent_id: string | null;
  /** Bumped when the data is recomputed or rebinned. */
  revision: number;
  /** Rebinning applied to the loaded data, e.g. ["x×2, y×2"]. */
  rebinned: string[];
  /** Line of a computed map, e.g. "Fe_Ka". */
  xray_line: string | null;
}

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
  listExamples: () => request<ExampleInfo[]>("/api/examples"),
  openExample: (id: string) => request<FileSummary>(`/api/examples/${id}`, { method: "POST" }),
  listDir: (path?: string) =>
    request<FsListing>(`/api/fs/list${path ? `?path=${encodeURIComponent(path)}` : ""}`),
};

/** Thumbnail URL; the revision makes recomputed maps bypass the browser cache. */
export const thumbnailUrl = (d: { id: string; revision: number }, size = 256) =>
  `/api/datasets/${d.id}/thumbnail?size=${size}&v=${d.revision}`;
