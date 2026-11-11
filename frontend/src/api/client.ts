// Typed client for the FastAPI backend. Mirrors si_viewer/session.py summaries.

export type Technique = "EDS" | "EELS";
export type TechniqueSource = "metadata" | "dm_tag" | "energy_axis" | "user";

export type DatasetKind = "image" | "stack" | "spectrum" | "spectrum_image" | "other";

/** What an in-situ series dataset holds: one frame, or the mean/sum of frames
 * start…end (0-based, inclusive). */
export interface InsituView {
  mode: "frame" | "mean" | "sum";
  start: number;
  end: number;
}

/** A Gatan in-situ series (time, y, x[, energy]), read one frame at a time.
 * The spectrum image of such a series (``view`` is set) holds one frame, or a
 * reduction of a range, as an ordinary 2D spectrum image. */
export interface InsituInfo {
  /** Complete frames present in the .raw files. */
  frames: number;
  /** Frames the .dm4 says were recorded (null if it doesn't say). */
  recorded_frames: number | null;
  /** Recorded frames summed in each frame (1: not binned along time). */
  time_binning?: number;
  /** One frame as in the array: [y, x, energy] or [y, x]. */
  frame_shape: number[];
  first_time_s: number;
  last_time_s: number;
  time_source: "recorded" | "duration" | "index";
  duration_s: number | null;
  /** Set when fewer frames than recorded are present. */
  note: string | null;
  /** Spectrum image only: the time of every frame (s), the median step, the
   * experiment start, the size of a frame and of the .raw files (bytes). */
  frame_times_s?: number[];
  frame_step_s?: number | null;
  start_time?: string | null;
  frame_bytes?: number | null;
  raw_bytes?: number | null;
  raw_files?: string[];
  /** Spectrum image only: what the dataset holds now. */
  view?: InsituView;
}

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

/** An ionisation edge of an element inside a spectrum's energy range. */
export interface EelsSubshell {
  /** e.g. "Mn_L3". */
  name: string;
  element: string;
  edge: string;
  /** eV. */
  onset: number;
  major: boolean;
}

/** EELS spectra and spectrum images: the elements chosen for a model fit and
 * their edges inside the energy range (by onset). */
export interface EdgesInfo {
  elements: string[];
  subshells: EelsSubshell[];
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
  /** EELS spectra and spectrum images only. */
  edges: EdgesInfo | null;
  /** In-situ series (spectrum image or image stack); null otherwise. */
  insitu: InsituInfo | null;
  /** "computed": an element map made from a spectrum image (window +
   * background); "fitted": from fitting every pixel. */
  origin:
    | "file" | "computed" | "fitted" | "aligned-average" | "window" | "background" | "background-r"
    | "thickness" | "zlp-shift" | "deconvolved" | "line-scan" | "aligned-sum" | "rebinned" | "zlp-aligned" | "aligned";
  /** Window maps: the energy window (in the spectrum's units) it sums. */
  window: { lo: number; hi: number; units: string } | null;
  /** Element symbol of an element map. */
  element: string | null;
  /** Fitted maps: what they can show (counts, and at% when composition was
   * made) and what they show; empty / null for other datasets. */
  quantities: FitQuantity[];
  quantity: FitQuantity | null;
  /** EELS spectrum images: every pixel was fitted (the fit at the cursor can be shown). */
  pixel_fit: boolean;
  /** Spectrum image a computed map was made from. */
  parent_id: string | null;
  /** Bumped when the data is recomputed or rebinned. */
  revision: number;
  /** Bumped when the data is replaced by other data of the same kind (the next
   * frame of an in-situ series): refetch, but keep contrast and zoom. */
  data_rev: number;
  /** Background removal of an in-situ frame: which one, e.g. "frame 37". */
  source_frame: string | null;
  /** Rebinning applied to the loaded data, e.g. ["x×2, y×2"]. */
  rebinned: string[];
  /** Zero-loss alignment applied to the loaded data, e.g. ["sub-pixel, cropped"]. */
  zlp_aligned: string[];
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
  /** The survey image of a spectrum imaging run: the rectangles GMS drew on it. */
  survey: { regions: SurveyRegion[] } | null;
  /** EELS spectrum images made by background removal: the settings used
   * (re-run with PUT /background); else null. */
  background: BackgroundInfo | null;
  /** The thickness of a low loss (a map, or a profile for a line scan): the
   * settings it was made with (re-run with POST /thickness); else null. */
  thickness: ThicknessInfo | null;
  /** A deconvolved spectrum image: the settings it was made with (re-run
   * with PUT /deconvolution); else null. */
  deconvolution: DeconvolutionInfo | null;
}

/** Options of a thickness estimate. ``threshold`` is "auto" (estimated at
 * every pixel) or one energy (the energy axis' units) for all pixels. */
export interface ThicknessOptions {
  threshold: "auto" | number;
  /** Auto only: how far from the zero-loss peak its end is looked for. */
  window: number;
  output: "relative" | "absolute";
  /** Absolute thickness: from a mean free path (nm) or a density (g/cm³). */
  mean_free_path: number | null;
  density: number | null;
}

/** Options of a zero-loss alignment (the peak is always put at 0 eV). */
export interface ZlpOptions {
  /** Refine the alignment by cross-correlation (else the peak maxima are lined up). */
  subpixel: boolean;
  /** Crop the channels the shifts leave empty. */
  crop: boolean;
}

/** The settings a thickness was made with. */
export interface ThicknessInfo extends ThicknessOptions {
  source_id: string;
  /** Energy units of the threshold. */
  units: string;
}

export type BackgroundModel = "power_law" | "exponential" | "polynomial" | "offset";

/** Options of an EELS background removal; windows are [lo, hi] in energy units. */
export interface BackgroundOptions {
  model: BackgroundModel;
  /** Polynomial only (1…6). */
  order: number;
  /** Closed-form estimate (else a least-squares fit at every pixel). */
  fast: boolean;
  /** Power law and exponential: a linear fit of the logarithm (overrides fast). */
  log: boolean;
  /** LBA: FWHM in pixels of the Gaussian the background window is averaged with; 0 for none. */
  lba: number;
  /** Fit range, before the edge. */
  background: [number, number];
  /** Range kept; it starts where the background window ends or later. */
  signal: [number, number];
}

export type DeconvolutionMethod = "fourier_log" | "fourier_ratio" | "richardson_lucy";

/** Options of a plural-scattering deconvolution (energies in the low loss' units). */
export interface DeconvolutionOptions {
  method: DeconvolutionMethod;
  /** End of the zero-loss peak: estimated at every pixel, or one energy. */
  threshold: "auto" | number;
  /** Auto threshold: how far above the peak the search goes. */
  window: number;
  /** Fourier-log: add the zero-loss peak back. */
  add_zlp: boolean;
  /** Fourier-ratio: width of the Gaussian the result is convolved with (null: the zero-loss peak's). */
  fwhm: number | null;
  /** Fourier-ratio: extrapolate both spectra with a power law. */
  extrapolate: boolean;
  /** Richardson–Lucy. */
  iterations: number;
  psf: "low_loss" | "zlp";
}

/** The settings a deconvolved spectrum image was made with. */
export interface DeconvolutionInfo extends DeconvolutionOptions {
  source_id: string;
  low_loss_id: string | null;
  units: string;
}

/** Pixels whose summed spectrum is previewed. */
export type DeconvolutionSelection =
  | { kind: "whole" }
  | { kind: "pixel"; x: number; y: number; size: number }
  | { kind: "region"; region: object };

/** Deconvolved summed spectra (null for a selection with no pixels), and
 * the threshold and FWHM used for the first. */
export interface DeconvolutionPreview {
  spectra: (number[] | null)[];
  pixels: number[];
  threshold: number | null;
  fwhm: number | null;
}

/** The settings a background-removed spectrum image was made with. */
export interface BackgroundInfo extends BackgroundOptions {
  source_id: string;
  units: string;
}

/** Fitted backgrounds of some spectra: per spectrum the values on channels
 * start … start + n - 1 (to the end of the signal window), and its power-law
 * exponent (null for other models). */
/** What a time scan averages in every frame: a pixel block or a region. */
export type TimeScanSource = { pixel: { x: number; y: number; size: number } } | { region: object };

/** The block of pixels a previewed spectrum sums (for LBA). */
export interface BackgroundPixel {
  x: number;
  y: number;
  size: number;
}

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

export type SpatialFilter = "none" | "gaussian" | "median" | "mean" | "bandpass";
export type SpectralFilter = "none" | "gaussian" | "savgol";

/** Pre-processing filters of a dataset (see backend processing/filters.py). */
export interface Filters {
  spatial: SpatialFilter;
  /** Gaussian σ (pixels). */
  sigma: number;
  /** Median / mean window (pixels, odd). */
  size: number;
  /** Bandpass: structures smaller / larger than these (pixels) are removed. */
  small: number;
  large: number;
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
/** The applied alignment's shifts as a NumPy array (frames, 2): (x, y) that align each frame; NaN: left out. */
export const alignmentNpyUrl = (id: string) => `/api/datasets/${id}/alignment/shifts.npy`;

/** What the aligned sum of an in-situ series can use. */
export interface AlignmentSource {
  /** Recorded frames of the series. */
  frames: number;
  /** The simultaneous ADF stack (k: its pixels per scan pixel), if any. */
  adf: { id: string; title: string; k: number; aligned: boolean; frames: number } | null;
}

/** Where the aligned sum's shifts come from: the ADF's alignment, or a .npy file (base64). */
export interface ShiftsChoice {
  source: "adf" | "npy";
  npy?: string;
  units?: "adf" | "scan";
}

/** Shifts checked against the series: frames summed and left out, the largest shift (scan px), the area kept. */
export interface ShiftsCheck {
  frames: number;
  left_out: number;
  largest: number;
  width: number;
  height: number;
  source: string;
}


type Versioned = { revision: number; filter_rev: number; data_rev?: number; insitu?: { view?: { mode: string } | null } | null };

/** Changes whenever the data shown for a dataset does (recomputed, rebinned,
 * filtered or replaced by the next frame), for cache keys. */
export const dataVersion = (d: Versioned) => `${d.revision}.${d.filter_rev}.${d.data_rev ?? 0}`;

export type FitQuantity = "counts" | "at%";

/** Like ``dataVersion`` but unchanged by a new frame: contrast is redone when
 * it changes. An in-situ sum is on another scale than a frame or a mean, so
 * switching to or from it changes it too. */
export const contrastVersion = (d: Versioned) =>
  `${d.revision}.${d.filter_rev}${d.insitu?.view?.mode === "sum" ? ".sum" : ""}`;

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

/** Cross-sections of an EELS model: exSpy's GOS tables, or the hydrogenic model (K and L edges). */
export type EelsGos = "dft" | "dirac" | "hydrogenic";

export interface EelsFitOptions {
  gos: EelsGos;
  /** Fit the fine structure after each onset with a spline over ``fine_structure_width`` (eV). */
  fine_structure: boolean;
  fine_structure_width: number;
  /** Element → onset (eV) of its edge, where the database's is off. */
  onsets: Record<string, number>;
  /** Fitted range (eV); null: to the end of the spectrum. */
  lo: number | null;
  hi: number | null;
  /** Convolve the model with the group's low loss, when it has one. */
  convolve: boolean;
}

export interface EelsFitRow {
  element: string;
  /** The edge carrying the element's intensity, e.g. "Mn_L3". */
  edge: string;
  onset: number;
  intensity: number;
  std: number | null;
  atomic_percent: number | null;
  fine_structure_width: number | null;
}

/** An EELS model fit of one spectrum: curves over the fitted range (as ``FitResult``) and a row per element. */
export interface EelsFitResult {
  x: number[];
  data: number[];
  model: number[];
  background: number[];
  elements: Record<string, number[]>;
  rows: EelsFitRow[];
  /** Exponent of the power-law background. */
  background_r: number | null;
  residual_ratio: number | null;
  convolved: boolean;
  low_loss_id: string | null;
  gos: EelsGos;
  pixels: number;
  /** Elements with no edge in the fitted range. */
  skipped: string[];
}

/** A dropped file or folder, as a browser describes it (it gives no path). */
export interface DroppedItem {
  name: string;
  is_dir: boolean;
  size?: number;
  /** ms since the epoch. */
  modified?: number;
  /** Folders: some of the files directly inside. */
  children: { name: string; size: number }[];
}

/** A dataset change that may also have changed its file's maps. */
export interface DatasetAndFile {
  dataset: DatasetSummary;
  file: FileSummary;
  warnings?: string[];
}

/** A rectangle on a survey image, in pixels: where the spectrum image was
 * taken ("si"), the spatial drift reference ("drift"), or another one. */
export interface SurveyRegion {
  label: string;
  kind: "si" | "drift" | "other";
  x: number;
  y: number;
  width: number;
  height: number;
}

/** The files of one folder, opened together. */
export interface ExperimentSummary {
  id: string;
  name: string;
  path: string;
}

export interface FileSummary {
  id: string;
  path: string;
  name: string;
  experiment: ExperimentSummary | null;
  /** What the file is in its experiment ("Survey", "ADF", "EELS LL", …). */
  role: string | null;
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
  /** Pixels of the image per pixel of the spectrum image's scan, along each
   * axis (1: the scan's own grid; k > 1: a sub-scan, shown at its resolution). */
  factor: number;
  /** The image dataset (its axes are the navigator's calibration). */
  dataset_id: string;
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
  result: {
    file: FileSummary;
    skipped?: string[];
    dataset_id?: string;
    /** Zero-loss alignment: the other files changed, the ids aligned, and how far their axes moved (eV). */
    partner_files?: FileSummary[];
    aligned?: string[];
    shift?: number;
  } | null;
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

/** A curve of an exported plot (matplotlib); null values are gaps. */
export interface PlotSeriesBody {
  label: string;
  x: number[];
  y: (number | null)[];
  /** Hex colour. */
  color: string;
  dash?: number[];
  /** Relative to the live plot's line width (1.25). */
  width?: number;
  axis: "left" | "right";
}

export interface PlotMarkerBody {
  x: number;
  label?: string;
  color: string;
  weight: number;
}

export interface PlotWindowBody {
  lo: number;
  hi: number;
  color: string;
  label: string;
}

export type PlotLimits = [number | null, number | null];

export interface PlotOptionsBody {
  style: "overlap" | "cascade";
  padding: number;
  normalise: boolean;
  reverse: boolean;
  log: boolean;
  title: string;
  x_label: string;
  y_label: string;
  y2_label: string;
  xlim: PlotLimits;
  ylim: PlotLimits;
  y2lim: PlotLimits;
  title_size: number;
  label_size: number;
  tick_size: number;
  legend_size: number;
  legend: boolean;
  legend_loc: string;
  /** Base line width (pt). */
  line_width: number;
  /** Figure size (inches). */
  width: number;
  height: number;
  dpi: number;
  format: "png" | "svg" | "pdf";
}

export interface PlotBody {
  series: PlotSeriesBody[];
  markers: PlotMarkerBody[];
  windows: PlotWindowBody[];
  options: PlotOptionsBody;
}

/** The plot drawn by the server as an image (PNG, SVG or PDF). */
async function renderPlot(body: PlotBody, signal?: AbortSignal): Promise<Blob> {
  const res = await fetch("/api/plot/render", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    let detail: string = res.statusText;
    try {
      const d = (await res.json()).detail;
      // A malformed request comes back as a list of problems.
      if (typeof d === "string") detail = d;
      else if (Array.isArray(d)) detail = d.map((e: { msg?: string }) => e.msg ?? "").filter(Boolean).join("; ") || detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.blob();
}

export const api = {
  health: () => request<Health>("/api/health"),
  renderPlot,
  listFiles: () => request<FileSummary[]>("/api/files"),
  /** Where dropped files and folders are on disk (null: not found). */
  locate: (items: DroppedItem[], hints: string[]) =>
    request<{ paths: (string | null)[] }>("/api/fs/locate", { method: "POST", body: JSON.stringify({ items, hints }) }),
  openFile: (path: string, experimentId?: string) =>
    request<FileSummary>("/api/files", {
      method: "POST",
      body: JSON.stringify({ path, experiment_id: experimentId ?? null }),
    }),
  /** Start an experiment of a folder: the files to open into it, in order. */
  openFolder: (path: string) =>
    request<{ experiment: ExperimentSummary; paths: string[] }>("/api/files/folder", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),
  closeFile: (id: string) => request<void>(`/api/files/${id}`, { method: "DELETE" }),
  getDataset: (id: string) => request<DatasetSummary>(`/api/datasets/${id}`),
  loadDataset: (id: string) => request<DatasetSummary>(`/api/datasets/${id}/load`, { method: "POST" }),
  navigators: (id: string) => request<NavigatorSource[]>(`/api/datasets/${id}/navigators`),
  calibrate: (id: string, axes: AxisCalibration[]) =>
    request<DatasetSummary>(`/api/datasets/${id}/axes`, { method: "PATCH", body: JSON.stringify(axes) }),
  setElements: (id: string, elements: string[]) =>
    request<DatasetAndFile>(`/api/datasets/${id}/elements`, { method: "PUT", body: JSON.stringify({ elements }) }),
  setEelsElements: (id: string, elements: string[]) =>
    request<DatasetAndFile>(`/api/datasets/${id}/eels-elements`, { method: "PUT", body: JSON.stringify({ elements }) }),
  /** The per-pixel fit at the cursor: pixel (x, y), or the ``size`` × ``size`` block there added up. */
  eelsPixelFit: (id: string, x: number, y: number, size: number, signal?: AbortSignal) =>
    request<EelsFitResult>(`/api/datasets/${id}/eels-pixel-fit?x=${x}&y=${y}&size=${size}`, { signal }),
  eelsFit: (id: string, source: FitSource, options: EelsFitOptions) =>
    request<EelsFitResult>(`/api/datasets/${id}/eels-fit`, { method: "POST", body: JSON.stringify({ source, options }) }),
  /** Job: fit every pixel; its result's file has a fitted map per element (and at% maps). */
  startEelsFitMaps: (id: string, options: EelsFitOptions, composition: boolean) =>
    request<JobSummary>(`/api/datasets/${id}/eels-fit-maps`, {
      method: "POST",
      body: JSON.stringify({ options, composition }),
    }),
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
  /** Give a dataset a name of the user's (kept when its data is recomputed). */
  rename: (id: string, title: string) =>
    request<DatasetSummary>(`/api/datasets/${id}/title`, { method: "PUT", body: JSON.stringify({ title }) }),
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
  /** Delete a map or result; ``undo`` is the token that brings it back. */
  deleteDataset: (id: string) =>
    request<{ file: FileSummary; undo: string }>(`/api/datasets/${id}`, { method: "DELETE" }),
  /** Undo a deletion: the files that changed. */
  undoDelete: (token: string) => request<{ files: FileSummary[] }>(`/api/undo/${token}`, { method: "POST" }),
  fit: (id: string, source: FitSource, options: FitOptions) =>
    request<FitResult>(`/api/datasets/${id}/fit`, { method: "POST", body: JSON.stringify({ source, options }) }),
  /** One integer factor per axis, in the summary's axis order. */
  /** Cut a line scan along ``line`` (scan pixels; ``width`` averaged across),
   * and from the spectrum images linked to it (``partners``, linked in turn). */
  cutLineScan: (id: string, line: { x0: number; y0: number; x1: number; y1: number; width: number }, name: string) =>
    request<DatasetAndFile & { partners: DatasetSummary[]; partner_files: FileSummary[] }>(
      `/api/datasets/${id}/line-scan`,
      { method: "POST", body: JSON.stringify({ line, name }) },
    ),
  /** Show ``quantity`` on all the maps of a spectrum image's per-pixel fit. */
  setFitQuantity: (siId: string, quantity: FitQuantity) =>
    request<{ file: FileSummary }>(`/api/datasets/${siId}/fit-quantity`, {
      method: "PUT",
      body: JSON.stringify({ quantity }),
    }),
  /** ``time``: frames per binned frame of an in-situ series. */
  rebin: (id: string, factors: number[], time = 1, inPlace = true) =>
    request<DatasetAndFile>(`/api/datasets/${id}/rebin`, {
      method: "POST",
      body: JSON.stringify({ factors, time, in_place: inPlace }),
    }),
  setFilters: (id: string, filters: Filters) =>
    request<DatasetAndFile & { notes: string[] }>(`/api/datasets/${id}/filters`, {
      method: "PUT",
      body: JSON.stringify(filters),
    }),
  /** Fitted background of each given spectrum (on the dataset's full energy axis). */
  backgroundPreview: (
    id: string, options: BackgroundOptions, spectra: number[][], pixels?: (BackgroundPixel | null)[], signal?: AbortSignal,
  ) =>
    request<BackgroundPreview>(`/api/datasets/${id}/background/preview`, {
      method: "POST",
      body: JSON.stringify({ options, spectra, pixels }),
      signal,
    }),
  /** Job: a new spectrum image of the signal window with the background removed. */
  removeBackground: (id: string, options: BackgroundOptions) =>
    request<JobSummary>(`/api/datasets/${id}/background`, { method: "POST", body: JSON.stringify({ options }) }),
  /** Job: redo a background-removed spectrum image in place with new options. */
  rerunBackground: (id: string, options: BackgroundOptions) =>
    request<JobSummary>(`/api/datasets/${id}/background`, { method: "PUT", body: JSON.stringify({ options }) }),
  /** Deconvolved summed spectra of some selections of pixels. */
  deconvolutionPreview: (
    id: string, options: DeconvolutionOptions, selections: DeconvolutionSelection[], signal?: AbortSignal,
  ) =>
    request<DeconvolutionPreview>(`/api/datasets/${id}/deconvolution/preview`, {
      method: "POST",
      body: JSON.stringify({ options, selections }),
      signal,
    }),
  /** Job: a new spectrum image, deconvolved with the group's low loss. */
  deconvolve: (id: string, options: DeconvolutionOptions) =>
    request<JobSummary>(`/api/datasets/${id}/deconvolution`, { method: "POST", body: JSON.stringify({ options }) }),
  /** Job: redo a deconvolved spectrum image in place with new options. */
  rerunDeconvolution: (id: string, options: DeconvolutionOptions) =>
    request<JobSummary>(`/api/datasets/${id}/deconvolution`, { method: "PUT", body: JSON.stringify({ options }) }),
  /** The automatic zero-loss threshold of a low loss's summed spectrum (null: none found). */
  thicknessThreshold: (id: string, window: number, signal?: AbortSignal) =>
    request<{ low_loss_id: string; threshold: number | null }>(`/api/datasets/${id}/thickness/threshold`, {
      method: "POST",
      body: JSON.stringify({ window }),
      signal,
    }),
  /** Job: the thickness of a low loss (or of the low loss a linked dataset belongs to);
   * replaces the one it has. */
  estimateThickness: (id: string, options: ThicknessOptions) =>
    request<JobSummary>(`/api/datasets/${id}/thickness`, { method: "POST", body: JSON.stringify({ options }) }),
  /** Job: align the zero-loss peak of a low loss (or of the low loss a linked dataset belongs to)
   * and of its linked core losses, replacing their loaded data. */
  alignZeroLoss: (id: string, options: ZlpOptions, inPlace = true) =>
    request<JobSummary>(`/api/datasets/${id}/zlp-alignment`, {
      method: "POST",
      body: JSON.stringify({ options, in_place: inPlace }),
    }),
  startFitMaps: (id: string, options: FitOptions, quant: QuantSpec) =>
    request<JobSummary>(`/api/datasets/${id}/fit-maps`, { method: "POST", body: JSON.stringify({ options, quant }) }),
  /** In-situ series: hold frame ``frame`` (0-based). Reads one frame; the file's window maps follow it. */
  setInsituFrame: (id: string, frame: number) =>
    request<DatasetAndFile>(`/api/datasets/${id}/insitu/frame`, { method: "PUT", body: JSON.stringify({ frame }) }),
  /** Job: hold the mean or sum of frames start…end (0-based, inclusive); reads every frame of the range. */
  reduceInsitu: (id: string, mode: "mean" | "sum", start: number, end: number) =>
    request<JobSummary>(`/api/datasets/${id}/insitu/reduce`, {
      method: "POST",
      body: JSON.stringify({ mode, start, end }),
    }),
  /** Job: a line scan over the frames of an in-situ series (row k: the mean
   * spectrum of the pixel block or region in frame k), in scan pixels. */
  timeScan: (id: string, source: TimeScanSource, name: string) =>
    request<JobSummary>(`/api/datasets/${id}/insitu/time-scan`, {
      method: "POST",
      body: JSON.stringify({ ...source, name }),
    }),
  insituAlignment: (id: string) => request<AlignmentSource>(`/api/datasets/${id}/insitu/alignment`),
  checkShifts: (id: string, choice: ShiftsChoice) =>
    request<ShiftsCheck>(`/api/datasets/${id}/insitu/shifts-check`, { method: "POST", body: JSON.stringify(choice) }),
  /** Job: every frame registered with the shifts and summed: a new spectrum image under the series. */
  alignedSum: (id: string, choice: ShiftsChoice) =>
    request<JobSummary>(`/api/datasets/${id}/insitu/aligned-sum`, { method: "POST", body: JSON.stringify(choice) }),
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
  /** ``inPlace`` false: the aligned frames become a new stack (``aligned_id``) under it. */
  applyAlignment: (id: string, inPlace = true) =>
    request<DatasetAndFile & { average: DatasetSummary; aligned_id: string }>(`/api/datasets/${id}/registration/apply`, {
      method: "POST",
      body: JSON.stringify({ in_place: inPlace }),
    }),
  showAlignment: (id: string, active: boolean) =>
    request<DatasetAndFile>(`/api/datasets/${id}/alignment`, { method: "PUT", body: JSON.stringify({ active }) }),
  removeAlignment: (id: string) => request<DatasetAndFile>(`/api/datasets/${id}/alignment`, { method: "DELETE" }),
  exportFormats: () => request<Record<string, ExportFormat[]>>("/api/export-formats"),
  listExamples: () => request<ExampleInfo[]>("/api/examples"),
  openExample: (id: string) => request<FileSummary>(`/api/examples/${id}`, { method: "POST" }),
  /** A folder's subfolders and openable files, or with ``ext`` its files with that extension. */
  listDir: (path?: string, ext?: string) => {
    const q = new URLSearchParams();
    if (path) q.set("path", path);
    if (ext) q.set("ext", ext);
    const query = q.toString();
    return request<FsListing>(`/api/fs/list${query ? `?${query}` : ""}`);
  },
  /** Write a dataset as ``format`` to ``path`` (ApiError 409 if it exists and not ``overwrite``). */
  saveDataset: (id: string, format: string, path: string, overwrite = false) =>
    request<{ path: string }>(`/api/datasets/${id}/save`, {
      method: "POST",
      body: JSON.stringify({ format, path, overwrite }),
    }),
  /** Write datasets as ``format`` into ``folder``, named after their titles (409 if any exist). */
  batchExport: (ids: string[], format: string, folder: string, overwrite = false) =>
    request<{ paths: string[] }>("/api/export/batch", {
      method: "POST",
      body: JSON.stringify({ ids, format, folder, overwrite }),
    }),
};

/** Thumbnail URL; the version makes recomputed or filtered data bypass the browser cache. */
export const thumbnailUrl = (d: Versioned & { id: string }, size = 256) =>
  `/api/datasets/${d.id}/thumbnail?size=${size}&v=${dataVersion(d)}`;
