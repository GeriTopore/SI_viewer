// Typed client for the FastAPI backend. Mirrors si_viewer/session.py summaries.

export type Technique = "EDS" | "EELS";
export type TechniqueSource = "metadata" | "dm_tag" | "energy_axis" | "user";

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

export type EelsRole = "low_loss" | "core_loss";

/** Where a role came from: the file's metadata, the user, or a default (core loss). */
export type EelsRoleSource = "metadata" | "user" | "default";

/** EELS spectrum images (incl. line scans): low or core loss (and where that
 * came from), and the others in their link group (possibly in other files):
 * the low loss first, then core losses by energy offset. */
export interface EelsInfo {
  role: EelsRole;
  role_source: EelsRoleSource;
  linked: string[];
  beam_energy: number | null;
  convergence_angle: number | null;
  collection_angle: number | null;
}

/** A dataset a spectrum image could be linked with (same scan, any role). */
export interface EelsPartner {
  id: string;
  title: string;
  file_id: string;
  role: EelsRole;
  /** Its link group, if any. */
  group: string | null;
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
  /** Spectra and spectrum images only; null when unknown (or not a spectrum). */
  technique: Technique | null;
  technique_source: TechniqueSource | null;
  /** EELS spectrum images only. */
  eels: EelsInfo | null;
  /** "computed": an element map made from a spectrum image (window +
   * background); "fitted"/"composition": from fitting every pixel. */
  origin:
    | "file" | "computed" | "fitted" | "composition" | "aligned-average" | "window" | "background" | "background-r";
  /** Window maps: the energy window (in the spectrum's units) it sums. */
  window: { lo: number; hi: number; units: string } | null;
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
  /** Stacks: an applied drift correction, shown when active (the frames are
   * then the aligned ones, cropped). */
  alignment: AppliedAlignment | null;
  /** A dataset made from another, e.g. a stack's aligned average. */
  derived_from: string | null;
  /** EELS spectrum images made by background removal: the settings used
   * (re-run with PUT /background); else null. */
  background: BackgroundInfo | null;
}

export type BackgroundModel = "power_law" | "exponential" | "polynomial" | "offset";

/** Options of an EELS background removal; windows are [lo, hi] in energy units. */
export interface BackgroundOptions {
  model: BackgroundModel;
  /** Polynomial only (1…6). */
  order: number;
  /** Closed-form estimate (else a least-squares fit at every pixel). */
  fast: boolean;
  /** Fit range, before the edge. */
  background: [number, number];
  /** Range kept; it starts where the background window ends or later. */
  signal: [number, number];
}

/** The settings a background-removed spectrum image was made with. */
export interface BackgroundInfo extends BackgroundOptions {
  source_id: string;
  units: string;
}

/** Fitted backgrounds of some spectra: per spectrum the values on channels
 * start … start + n - 1 (to the end of the signal window), and its power-law
 * exponent (null for other models). */
export interface BackgroundPreview {
  start: number;
  background: number[][];
  r: (number | null)[];
}

/** A drift correction stored with a stack (Apply in the Result step). */
export interface AppliedAlignment {
  /** Frames kept (loaded numbering, 0-based); bad frames are left out. */
  frames: number[];
  first: number;
  last: number;
  rect: [number, number, number, number];
  square: [number, number, number, number];
  drift: Drift;
  settings: {
    mask: MaskSettings;
    peaks: PeakSettings;
    outliers: OutlierSettings;
    bad_frames: number[];
    /** Absent in alignments applied before manual edits existed. */
    edits?: MaskEdits;
  };
  /** The aligned frames are shown (else the frames as loaded). */
  active: boolean;
  loaded_frames: number;
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

export type MaskType = "bandpass" | "hann" | "gaussian" | "hamming" | "blackman" | "elliptical" | "none";

/** Fourier mask of a registration: features smaller than ~n px are ignored. */
export interface MaskSettings {
  type: MaskType;
  n: number;
  /** Elliptical Gaussian: n is the cutoff along the first axis, n2 along the
   * second, the first axis at theta degrees. */
  n2: number;
  theta: number;
}

export type CorrelationType = "cc" | "mc" | "pc";
export type PeakMethod = "pixel" | "gf" | "com";

/** How each pair's shift is read from its correlation. */
export interface PeakSettings {
  correlation: CorrelationType;
  method: PeakMethod;
  /** Gaussian fit: candidate maxima, starting σ, cut-out radius (px). */
  num_peaks: number;
  sigma_guess: number;
  window_radius: number;
}

/** Which pair shifts are masked as outliers before correcting. */
export interface OutlierSettings {
  /** Transitivity (shifts add up along paths) or neighbours (agree with the
   * neighbouring matrix entries). */
  method: "transitivity" | "neighbours" | "none";
  /** Transitivity: allowed mismatch (px) and how many paths are averaged. */
  threshold: number;
  max_paths: number;
  /** Neighbours: allowed difference (px). */
  max_shift: number;
  /** Use only pairs at most this many frames apart (null: all). */
  off_diagonal: number | null;
}

/** Pairs of frames (stack numbering) whose outlier mask is set by hand; each
 * applies to both (i, j) and (j, i). Keeps override the automatic tests. */
export interface MaskEdits {
  keep: [number, number][];
  mask: [number, number][];
}

/** A candidate maximum of a pair's correlation (Gaussian fit), with its
 * cut-out and fitted model. Positions are in the centred correlation. */
export interface PeakCandidate {
  x: number;
  y: number;
  fitted: boolean;
  chosen?: boolean;
  success?: boolean;
  fit_x?: number;
  fit_y?: number;
  amplitude?: number;
  sigma?: [number, number];
  offset?: number;
  data?: number[][];
  model?: number[][];
}

/** How a pair's shift is found: the shift and where its peak is. */
export interface PeakResult {
  dx: number;
  dy: number;
  x: number;
  y: number;
  candidates: PeakCandidate[];
}

/** Each registered frame's measured position relative to the mean (px;
 * image x right, y down), from the corrected shift matrices. */
export interface Drift {
  frames: number[];
  x: number[];
  y: number[];
}

/** The aligned stack: the frames kept (bad ones left out), the area inside
 * every frame and the square cropped from it ([x, y, width, height] in
 * aligned pixels; a frame's pixel p lands at p - drift), and the drift. */
export interface RegistrationResult {
  frames: number[];
  rect: [number, number, number, number];
  square: [number, number, number, number];
  drift: Drift;
}

/** State of a stack's registration in progress (backend session). */
export interface RegistrationSummary {
  /** Frames first … last - 1 are registered (stack numbering). */
  first: number;
  last: number;
  n_total: number;
  /** Frame size (rows, columns). */
  shape: [number, number];
  bad_frames: number[];
  mask: MaskSettings;
  peaks: PeakSettings;
  outliers: OutlierSettings;
  edits: MaskEdits;
  /** The shift matrices' settings once computed; stale if changed since. */
  shifts: { mask: MaskSettings; peaks: PeakSettings; stale: boolean } | null;
  /** Memory taken by the FFTs of the frames. */
  fft_bytes: number;
  /** pyfftw is used (faster FFTs). */
  fftw: boolean;
}

/** A file format a dataset can be saved in. */
export interface ExportFormat {
  format: string;
  extension: string;
  label: string;
}

/** Download URL of a dataset saved as ``format`` (the data as shown). */
export const exportUrl = (id: string, format: string) =>
  `/api/datasets/${id}/export?format=${encodeURIComponent(format)}`;

/** Download URLs of a drift correction: the current registration's shifts
    (CSV) and report (PDF), and the applied alignment's shifts (CSV). */
export const registrationShiftsUrl = (id: string) => `/api/datasets/${id}/registration/shifts.csv`;
export const registrationReportUrl = (id: string) => `/api/datasets/${id}/registration/report.pdf`;
export const alignmentShiftsUrl = (id: string) => `/api/datasets/${id}/alignment/shifts.csv`;

/** Download URL of a zip of every element map of a file. */
export const mapsZipUrl = (fileId: string) => `/api/files/${fileId}/maps`;

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
  result: { file: FileSummary; skipped?: string[]; dataset_id?: string } | null;
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
  /** Switching away from EDS may remove computed maps, hence the file. */
  setTechnique: (id: string, technique: Technique | null) =>
    request<DatasetAndFile>(`/api/datasets/${id}/technique`, { method: "PUT", body: JSON.stringify({ technique }) }),
  setMapOptions: (id: string, options: Partial<MapOptions>) =>
    request<DatasetAndFile>(`/api/datasets/${id}/map-options`, { method: "PUT", body: JSON.stringify(options) }),
  eelsPartners: (id: string) => request<EelsPartner[]>(`/api/datasets/${id}/eels-partners`),
  /** Put a spectrum image in a partner's link group (null: leave its group); every changed file comes back. */
  setEelsPartner: (id: string, partnerId: string | null) =>
    request<{ files: FileSummary[] }>(`/api/datasets/${id}/eels-partner`, {
      method: "PUT",
      body: JSON.stringify({ partner_id: partnerId }),
    }),
  /** Say by hand that a spectrum image is a low or core loss; every changed file comes back. */
  setEelsRole: (id: string, role: EelsRole) =>
    request<{ files: FileSummary[] }>(`/api/datasets/${id}/eels-role`, {
      method: "PUT",
      body: JSON.stringify({ role }),
    }),
  /** Keep an energy window of a spectrum image as a map (label null: named after the window). */
  addWindowMap: (siId: string, lo: number, hi: number, label: string | null) =>
    request<DatasetAndFile>(`/api/datasets/${siId}/window-maps`, {
      method: "POST",
      body: JSON.stringify({ lo, hi, label }),
    }),
  /** Move a window map to [lo, hi]: recomputed in place (same id, new revision). */
  setWindowMap: (id: string, lo: number, hi: number) =>
    request<DatasetAndFile>(`/api/datasets/${id}/window`, { method: "PATCH", body: JSON.stringify({ lo, hi }) }),
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
  /** Fitted background of each given spectrum (on the dataset's full energy axis). */
  backgroundPreview: (id: string, options: BackgroundOptions, spectra: number[][], signal?: AbortSignal) =>
    request<BackgroundPreview>(`/api/datasets/${id}/background/preview`, {
      method: "POST",
      body: JSON.stringify({ options, spectra }),
      signal,
    }),
  /** Job: a new spectrum image of the signal window with the background removed. */
  removeBackground: (id: string, options: BackgroundOptions) =>
    request<JobSummary>(`/api/datasets/${id}/background`, { method: "POST", body: JSON.stringify({ options }) }),
  /** Job: redo a background-removed spectrum image in place with new options. */
  rerunBackground: (id: string, options: BackgroundOptions) =>
    request<JobSummary>(`/api/datasets/${id}/background`, { method: "PUT", body: JSON.stringify({ options }) }),
  startFitMaps: (id: string, options: FitOptions, quant: QuantSpec) =>
    request<JobSummary>(`/api/datasets/${id}/fit-maps`, { method: "POST", body: JSON.stringify({ options, quant }) }),
  getJob: (id: string) => request<JobSummary>(`/api/jobs/${id}`),
  cancelJob: (id: string) => request<JobSummary>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  startRegistration: (id: string, first: number, last: number) =>
    request<RegistrationSummary>(`/api/datasets/${id}/registration`, {
      method: "POST",
      body: JSON.stringify({ first, last }),
    }),
  getRegistration: (id: string) => request<RegistrationSummary>(`/api/datasets/${id}/registration`),
  putRegistration: (
    id: string,
    settings: {
      bad_frames?: number[]; mask?: MaskSettings; peaks?: PeakSettings; outliers?: OutlierSettings; edits?: MaskEdits;
    },
  ) =>
    request<RegistrationSummary>(`/api/datasets/${id}/registration`, { method: "PUT", body: JSON.stringify(settings) }),
  endRegistration: (id: string) => request<void>(`/api/datasets/${id}/registration`, { method: "DELETE" }),
  peakPreview: (id: string, i: number, j: number, mask: MaskSettings, peaks: PeakSettings) =>
    request<PeakResult>(`/api/datasets/${id}/registration/peak`, {
      method: "POST",
      body: JSON.stringify({ i, j, mask, peaks }),
    }),
  correctShifts: (id: string, outliers: OutlierSettings, badFrames: number[], edits: MaskEdits) =>
    request<Drift>(`/api/datasets/${id}/registration/correct`, {
      method: "POST",
      body: JSON.stringify({ outliers, bad_frames: badFrames, edits }),
    }),
  startShifts: (id: string) => request<JobSummary>(`/api/datasets/${id}/registration/shifts`, { method: "POST" }),
  registrationResult: (id: string) => request<RegistrationResult>(`/api/datasets/${id}/registration/result`),
  /** Store the result with the stack (shown aligned) and add the average. */
  applyAlignment: (id: string) =>
    request<DatasetAndFile & { average: DatasetSummary }>(`/api/datasets/${id}/registration/apply`, { method: "POST" }),
  showAlignment: (id: string, active: boolean) =>
    request<DatasetAndFile>(`/api/datasets/${id}/alignment`, { method: "PUT", body: JSON.stringify({ active }) }),
  removeAlignment: (id: string) => request<DatasetAndFile>(`/api/datasets/${id}/alignment`, { method: "DELETE" }),
  exportFormats: () => request<Record<string, ExportFormat[]>>("/api/export-formats"),
  listExamples: () => request<ExampleInfo[]>("/api/examples"),
  openExample: (id: string) => request<FileSummary>(`/api/examples/${id}`, { method: "POST" }),
  listDir: (path?: string) =>
    request<FsListing>(`/api/fs/list${path ? `?path=${encodeURIComponent(path)}` : ""}`),
};

/** Thumbnail URL; the version makes recomputed or filtered data bypass the browser cache. */
export const thumbnailUrl = (d: { id: string; revision: number; filter_rev: number }, size = 256) =>
  `/api/datasets/${d.id}/thumbnail?size=${size}&v=${dataVersion(d)}`;
