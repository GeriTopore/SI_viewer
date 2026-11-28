// Typed client for the FastAPI backend. Mirrors si_viewer/session.py summaries.

export type Technique = "EDS" | "EELS";
export type TechniqueSource = "metadata" | "dm_tag" | "energy_axis" | "user";

/** A crystal structure read from a CIF (as orix's Phase.from_cif). */
export interface StructureInfo {
  name: string;
  formula: string;
  spacegroup: { number: number; symbol: string } | null;
  point_group: string | null;
  crystal_system: string | null;
  /** a, b, c (Å), α, β, γ (degrees). */
  cell: number[];
  /** Atoms per cell, by occupancy. */
  composition: Record<string, number>;
  sites: { element: string; xyz: number[]; occupancy: number }[];
}

/** An inverse pole figure's sector as orix draws it (stereographic x, y; y up). */
export interface IpfSector {
  outline: { x: number[]; y: number[] };
  corners: { x: number; y: number; label: string }[];
  /** orix's colour key (PNG data URL) over ``extent`` [x0, x1, y0, y1], first row at y1. */
  key: string;
  extent: number[];
  point_group: string;
}

/** A Templates object's inverse pole figure: each template's beam direction. */
export interface TemplatesIpf {
  sector: IpfSector;
  points: { x: number[]; y: number[]; rgb: number[][] };
}

/** One template: its reflections (1/Å; intensity: the brightest 1). */
export interface TemplatePattern {
  index: number;
  x: number[];
  y: number[];
  intensity: number[];
  hkl: number[][];
  /** The crystal direction along the beam (largest 1). */
  uvw: number[];
  /** Bunge Euler angles (°). */
  euler: number[];
  reciprocal_radius: number;
}

/** pyxem's template matching at one scan position (``orientation.match``). */
export interface MatchResult {
  x: number;
  y: number;
  best: {
    index: number;
    correlation: number;
    /** In-plane rotation (°). */
    rotation: number;
    mirror: boolean;
    /** Bunge Euler angles (°). */
    euler: number[];
    /** The template's beam direction (largest 1). */
    uvw: number[];
    /** How the crystal is seen: towards the viewer and the screen's up, [uvw]. */
    view?: number[];
    up?: number[];
  };
  /** The best template's reflections as pyxem draws them: 1/Å on the pattern's axes. */
  vectors: { x: number[]; y: number[]; intensity: number[]; hkl: number[][] };
  /** The same on the polar transform: angle (rad), radius (1/Å). */
  polar: { theta: number[]; r: number[] };
  /** Every template's correlation (null: not matched): of the phase matched. */
  correlations: (number | null)[];
  /** Several phases (a phase map): the phase matched best, its Templates
   * object, and every phase's best. */
  phase?: string;
  phase_index?: number;
  templates?: string;
  phases?: { name: string; templates: string; correlation: number; uvw?: number[] }[];
  /** Every template matched, best first (all phases'; ``phase``: its index
   * in the templates matched). Not in an orientation map's. */
  ranked?: RankedMatch[];
}

/** One template of a match's ranked list: at what in-plane rotation it matched. */
export interface RankedMatch {
  index: number;
  correlation: number;
  rotation: number;
  mirror: boolean;
  uvw: number[];
  phase: number;
}

/** orix's plots of a Structure. */
export interface StructurePlots {
  symmetry: { name: string; upper: { x: number[]; y: number[] }; lower: { x: number[]; y: number[] } };
  sector: IpfSector;
}

/** A zone axis the templates are kept near (within ``angle``, degrees). */
export interface ZoneAxis {
  uvw: number[];
  angle: number;
}

/** A Structure's templates: diffsims and orix settings (the backend's ``TemplateOptions``). */
export interface TemplateOptions {
  kv: number;
  resolution: number;
  zone_axes: ZoneAxis[];
  reciprocal_radius: number;
  max_excitation_error: number;
  precession_angle: number;
  minimum_intensity: number;
}

/** A Structure's templates (origin "templates", under the Structure). */
export interface TemplatesInfo {
  /** The phase (the Structure's title). */
  name: string;
  templates: number;
  options: TemplateOptions;
}

export type DatasetKind = "image" | "stack" | "spectrum" | "spectrum_image" | "4dstem" | "other";

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

/** EELS microscope parameters: beam energy (kV), convergence and collection angles (mrad). */
export interface MicroscopeInfo {
  beam_energy: number | null;
  convergence_angle: number | null;
  collection_angle: number | null;
}

/** Curve fitting (M21): a parameter of a HyperSpy component. */
export interface CurveParam {
  name: string;
  value: number;
  free: boolean;
  bmin: number | null;
  bmax: number | null;
  /** Its error after a fit. */
  std?: number | null;
}

export interface CurveComponentKind {
  kind: string;
  group: string;
  summary: string;
  /** It has ``estimate_parameters``. */
  estimate: boolean;
  params: CurveParam[];
}

export interface CurveCatalogue {
  components: CurveComponentKind[];
  max_order: number;
}

/** One component of a model, as sent to the backend. */
export interface CurveComponent {
  name: string;
  kind: string;
  active: boolean;
  /** Polynomial only. */
  order?: number;
  params: Record<string, CurveParamSpec>;
}

/** A parameter following another (HyperSpy's twin): ``expr`` of its value (``x``). */
export interface CurveTwin {
  component: string;
  param: string;
  expr: string;
}

export interface CurveParamSpec {
  value: number;
  free: boolean;
  bmin: number | null;
  bmax: number | null;
  twin?: CurveTwin | null;
}

export interface CurveModel {
  components: CurveComponent[];
  lo: number | null;
  hi: number | null;
  /** Background window: the background components are fitted there alone, then fixed. */
  bg_lo: number | null;
  bg_hi: number | null;
}

export interface CurveFitResult {
  x: number[];
  model: number[];
  components: { name: string; curve: number[] | null; params: CurveParam[] }[];
  range: [number, number];
  /** Plotted counts per channel = a component's function × this. */
  factor: number;
  residual_ratio: number | null;
  fitted: boolean;
  pixels: number;
  chisq?: number | null;
  red_chisq?: number | null;
  dof?: number;
  /** The noise variance is Poisson's (counts): the errors and χ² mean something. */
  poisson?: boolean;
  success?: boolean;
  message?: string;
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
/** How DPC images are made from the four quadrant images. */
/** How FFT peaks are refined (kemstem): 2D Gaussian fits or centre of mass,
 * in a window (odd, FFT px) of the blurred log pattern. */
export interface FftPeakOptions {
  method: "gauss" | "com";
  window: number;
  blur: number;
  edge: "none" | "hann" | "pplus";
}

/** What Fourier filtering makes (per peak), and how (kemstem). */
/** How a lattice was fitted at every position of 4D-STEM data. */
export interface StrainFitMeta {
  lattice: { ox: number; oy: number; g1x: number; g1y: number; g2x: number; g2y: number };
  options: { tolerance: number; min_disks: number; weighted: boolean };
  frame: { centred: boolean; ellipse: boolean; rotation: { theta: number; mirror: boolean } | null };
}
export type StrainMapOutput = "exx" | "eyy" | "exy" | "oxy";
/** The strain maps' x axis: the scan's, along g1, or at ``angle``° (counter-clockwise on screen). */
export interface StrainAxes {
  along: "scan" | "g1" | "angle";
  angle: number;
}
/** The unstrained g1 and g2 (calibrated frame, pattern px), the positions
 * they are the median of, the maps' x axis (degrees, counter-clockwise). */
export interface StrainReference {
  g1: [number, number];
  g2: [number, number];
  positions: number;
  x_angle: number;
}

export type AtomicOutput = "amplitude" | "grating" | "phase" | "strain";

export interface FourierSettings {
  sigma: number;
  phase: "lockin" | "gpa";
  /** The lock-in's low-pass σ; null: twice the filter's. */
  lock_sigma: number | null;
  edge: "none" | "hann" | "pplus";
  /** |ε| beyond this is replaced by its median-filtered surroundings; null: kept. */
  mask_threshold: number | null;
}

/** Finding and fitting atomic columns (kemstem). */
export interface ColumnSettings {
  /** Gaussian blur (σ, px) of the normalised image first. */
  blur: number;
  /** Least distance between columns (px). */
  distance: number;
  /** Intensity threshold (0–1). */
  threshold: number;
  /** Fit window (odd, px). */
  window: number;
  /** Finding only: a broad Gaussian (σ, px) subtracted first (0: none). */
  background: number;
  /** How columns are found: kemstem's local maxima ("maxima", threshold
   * 0–1 of the image) or the maxima of the correlation with a Gaussian
   * template of σ ``template`` px ("template", threshold a correlation). */
  method?: "maxima" | "template";
  template?: number;
}

/** One column fitted, to choose the window: patches and its row of the table. */
export interface ColumnTest {
  data: number[][];
  fit: number[][];
  row: (number | null)[];
  fields: string[];
}

export interface AtomicMapInfo extends Omit<FourierSettings, "lock_sigma"> {
  output: "amplitude" | "grating" | "phase" | "strain-par" | "strain-perp" | "exx" | "eyy" | "exy" | "oxy" | "columns" | "sublattice"
    | "neighbourhood" | "bonds" | "pld-reference" | "pld-difference" | "displacements";
  /** Columns: how many, how many fits failed, and how they were found and fitted. */
  count?: number;
  failed?: number;
  window?: number;
  /** A sublattice (a dataset under its Columns, holding its columns' rows): its order. */
  index?: number;
  /** A neighbourhood (under the Columns): its origins' and neighbours'
   * datasets (Columns or sublattices; origins: else its parent), how many
   * neighbours each, and its k-means clusters ([dy, dx] px). */
  origins?: string;
  neighbours?: string;
  k?: number;
  clusters?: [number, number][];
  /** Bonds (under their neighbourhood): name, the guess vector ([dy, dx] px),
   * how far from it a neighbour may be (px), how many origins have none, and
   * their mean length (px). */
  name?: string;
  vector?: [number, number];
  missing?: number;
  mean_length?: number | null;
  /** PLD reference: the damping radius (FFT px) and the peaks damped (fy, fx). */
  radius?: number;
  damped?: [number, number][];
  /** Displacements: their median and largest magnitude (px). */
  median?: number | null;
  max?: number | null;
  /** A sublattice made from a mask: the mask's id. */
  mask?: string;
  /** A displacement from a cage of neighbours: "cage", and the cage's picked clusters ([dy, dx] px). */
  kind?: "cage";
  cage?: [number, number][];
  /** A cage of all the k nearest neighbours (none picked). */
  nearest?: boolean;
  /** What was picked or set by hand for it in the page (lattice vectors,
   * cage, cell sites …), kept with it. */
  picks?: Record<string, unknown>;
  blur?: number;
  distance?: number;
  threshold?: number;
  peaks: number[];
  freqs: [number, number][];
  lock_sigma: number;
  /** Real-space coarsening radius (image px). */
  coarsening: number;
}

/** An FFT peak found for PLD mapping: where (cycles/px), how far above the
 * background, its (h, k) in the lattice found, its label and family (0
 * Bragg, n: family n of off-lattice peaks, null: none). */
export interface PldDetected {
  fy: number;
  fx: number;
  height: number;
  hk?: [number, number];
  label: "bragg" | "superlattice";
  family: number | null;
}
/** Sites in a reference sublattice's cell, to split columns by (backend
 * ``CellSitesBody``): each column to the nearest site within ``radius`` px,
 * or (``mode`` "clusters") each site the dense groups of columns (within
 * ``link`` px of each other, ``least`` or more) that come within ``radius``
 * of it, however far they reach. */
export interface CellSitesRequest {
  reference: string;
  vectors: [[number, number], [number, number]];
  sites: [number, number][];
  names: string[];
  radius: number;
  mode?: "radius" | "clusters";
  link?: number;
  least?: number;
}

/** A site's cluster in the cell: its members' mean place (fractions), how
 * many columns it took, and whether no group came near it. */
export interface CellSiteFit {
  u: number;
  v: number;
  count: number;
  empty: boolean;
}

/** How a mask was made from an image (backend processing/masks.py). */
export interface MaskOptions {
  /** Gaussian blur (px) before thresholding. */
  smooth: number;
  keep: "above" | "below" | "between";
  lo: number | null;
  hi: number | null;
  invert: boolean;
  /** Opening then closing radius (px). */
  clean: number;
  /** Regions and holes smaller than this (px) go. */
  min_area: number;
}

export interface MaskInfo extends MaskOptions {
  source_id: string;
  /** The share of the pixels in it. */
  fraction: number;
}

export interface PldFamily {
  id: number;
  /** Offset from the lattice (h, k fractions): its peaks are at G ± q. */
  q: [number, number];
  count: number;
  strength: number;
}

export interface RefinedPeak {
  fy: number;
  fx: number;
  failed: boolean;
  amplitude?: number;
  background?: number;
  sigma?: [number, number];
}

export interface DpcSettings {
  /** The image of each quadrant (A–D); none for DPC made from 4D-STEM data. */
  inputs: Record<string, string>;
  normalise: boolean;
  /** "4d": made from 4D-STEM data's centre of mass (``cube``: its id). */
  source?: "4d";
  cube?: string;
  /** Rotation of the detector's axes into the image's (degrees). */
  theta: number;
  mirror: boolean;
}

/** A PCA of a spectrum image: its source, energy window and components. */
export interface PcaInfo {
  source_id: string;
  /** First and last energy of the window. */
  window: [number, number];
  channels: [number, number];
  poisson: boolean;
  components: number;
  /** Fraction of the sum of squares of each component (empty for the
   * algorithms that don't rank them: NMF, sparse PCA, the online ones). */
  explained: number[];
  /** HyperSpy's algorithm, and the name it goes by here (absent: "SVD", "PCA"). */
  algorithm?: DecompositionAlgorithm;
  name?: string;
  /** Each spectrum normalised before decomposing (null: as they are). */
  normalise?: { mode: "range" | "value" | "mean"; lo?: number | null; hi?: number | null } | null;
  /** A denoised spectrum image: the components it was rebuilt from. */
  n?: number;
}

/** What a PCA is made with. */
export type DecompositionAlgorithm =
  | "SVD" | "MLPCA" | "sklearn_pca" | "NMF" | "sparse_pca" | "mini_batch_sparse_pca" | "RPCA" | "ORPCA" | "ORNMF";

export interface PcaOptions {
  lo: number | null;
  hi: number | null;
  components: number;
  poisson: boolean;
  algorithm: DecompositionAlgorithm;
}

/** A segment image a DPC can be made from. */
export interface DpcCandidate {
  id: string;
  title: string;
  segments: number[];
  quadrant: string;
  origin: string;
}

export interface DetectorInfo {
  name: string;
  /** Hardware segment numbers. */
  segments: number[];
  /** Each segment's sign in the sum ("Positive" / "Negative"). */
  polarities: string[];
  /** The reader's name for it, e.g. "DFS(0,4)". */
  label: string | null;
}

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
  /** What the values are (HyperSpy's Signal.quantity, e.g. "Shift (px)"). */
  signal_quantity?: string | null;
  /** Present for EDS spectra and spectrum images. */
  eds: EdsInfo | null;
  /** Spectra and spectrum images only; null when unknown (or not a spectrum). */
  technique: Technique | null;
  technique_source: TechniqueSource | null;
  /** EELS spectrum images only. */
  eels: EelsInfo | null;
  /** EELS spectra and spectrum images only. */
  edges: EdgesInfo | null;
  /** EELS: the microscope parameters its quantification needs (null: not in the metadata). */
  microscope: MicroscopeInfo | null;
  /** Images of a segmented detector: the segments summed into it, as
   * recorded at acquisition (Velox), e.g. DFS 0 and 4; null otherwise. */
  detector: DetectorInfo | null;
  /** In-situ series (spectrum image or image stack); null otherwise. */
  insitu: InsituInfo | null;
  /** "computed": an element map made from a spectrum image (window +
   * background); "fitted": from fitting every pixel; "curve-fit": maps of the curve fit of every pixel. */
  origin:
    | "file" | "computed" | "fitted" | "aligned-average" | "window" | "background" | "background-r"
    | "thickness" | "zlp-shift" | "deconvolved" | "line-scan" | "aligned-sum" | "rebinned" | "zlp-aligned" | "aligned" | "dpc" | "atomic"
    | "pca" | "pca-factors" | "pca-denoised" | "cropped" | "curve-fit" | "curve-model" | "mask" | "virtual" | "disks" | "bragg-map" | "beam-shift" | "centred" | "origin-map" | "strain-fit" | "templates" | "polar" | "vector-masked" | "orientation-map";
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
  /** Spectrum images: the model of the curve fit of every pixel (null: none). */
  curve_pixel_fit?: CurveModel | null;
  /** A curve model (origin "curve-model"): the spectrum image it was fitted
   * to and the model per pixel. */
  curve_model?: { source_id: string; model: CurveModel } | null;
  /** Atomic-resolution maps (kemstem): how each was made. */
  atomic?: AtomicMapInfo | null;
  /** A mask made from an image (origin "mask"): how. */
  mask?: MaskInfo | null;
  /** A virtual image of 4D-STEM data: the detector (pattern pixels) and response it was made with. */
  virtual?: { source_id: string; detector: Record<string, number | string>; response: string } | null;
  /** Bragg disks found in 4D-STEM data (the "Bragg vectors" table, origin
   * "disks", and its "Bragg vector map", origin "bragg-map"). */
  /** 4D-STEM data's calibration (py4DSTEM-style): the pattern's pixel size
   * (scale, units, how), the ring's ellipse, each scan position's origin;
   * which are applied to the Bragg vectors (``enabled``), and whether the
   * Bragg vector map shows them (``bvm``). Each part may be missing. */
  calibration?: {
    scale?: number;
    units?: string;
    how?: { standard?: string; hkl?: string; d?: number; camera_length?: number; pixel?: number; kv?: number; entered?: boolean };
    ring?: { cx: number; cy: number; a: number; b: number; angle: number; radius: number; ellipticity: number; rms?: number } | null;
    affine?: number[][] | null;
    origin?: { source: "vectors" | "beam"; plane: { mode: "none" | "plane" | "corners"; corners: number }; mean: [number, number]; rms: number };
    /** The QR rotation: the pattern's axes into the scan's (degrees; y flipped first if mirrored). */
    rotation?: { theta: number; mirror: boolean; how: "curl" | "com" | "entered"; curl_ratio?: number };
    enabled?: { origin?: boolean; ellipse?: boolean; scale?: boolean; rotation?: boolean };
    bvm?: "raw" | "calibrated";
  } | null;
  /** Strain from 4D-STEM Bragg vectors (origin "strain-fit"): the 4D data and
   * which output. The fit's ("Lattice fit" stack, "Indexed disks", "Fit
   * residual"): the lattice picked, the fit settings, the calibration it was
   * fitted in. The strain maps' (εxx, εyy, εxy, ωxy): the reference, the axes
   * and the fit they are from. */
  /** A crystal structure (a CIF, read as an orix phase). */
  structure?: StructureInfo | null;
  /** 4D-STEM data: orientation and phase mapping (pyxem). */
  /** Orientation mapping results: an orientation map ("map") and its
   * images ("correlation", "ipf" along ``direction``), with what was matched. */
  orientation?: {
    output: "map" | "correlation" | "phase" | "ipf";
    direction?: "x" | "y" | "z";
    /** The Templates matched (several, one per phase, joined by commas). */
    templates: string;
    polar: string;
    phase: string;
    /** A phase map: each phase, its Templates and colour. */
    phases?: { name: string; templates: string; colour: string }[];
    options: { n_best: number; frac_keep: number };
    map_id?: string;
  } | null;
  /** A Structure's templates (origin "templates"). */
  templates?: TemplatesInfo | null;
  /** Polar patterns: what they were made from (the source's pattern axes
   * then, ky and kx; source null when made in place). */
  polar?: { source: string | null; axes: { scale: number; offset: number; units: string }[]; options: Record<string, unknown> } | null;
  /** 4D-STEM data: the accelerating voltage the file records (kV). */
  beam_energy?: number | null;
  strain?: Partial<StrainFitMeta> & {
    source_id: string;
    output: "lattice" | "count" | "residual" | StrainMapOutput;
    components?: string[];
    reference?: StrainReference & { region: unknown };
    axes?: StrainAxes;
    fit?: StrainFitMeta;
  } | null;
  /** An origin map of 4D-STEM data ("Origin x / y", "(fit)"; origin "origin-map"). */
  origin_map?: { axis: "x" | "y"; kind: "measured" | "fitted"; source_id: string } | null;
  /** The direct beam found in 4D-STEM data ("Beam shift x" / "Beam shift y", origin "beam-shift"). */
  beam?: {
    source_id: string;
    axis: "x" | "y";
    options: Record<string, unknown>;
    range: { x: [number, number]; y: [number, number] };
    /** The data's revision it was found on (centring it makes it stale). */
    revision?: number;
  } | null;
  disks?: {
    source_id: string;
    options: Record<string, unknown>;
    template: { kind: string; cx: number; cy: number; r: number };
    count: number;
    positions: number;
  } | null;
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
  /** DPC images: how they were made. */
  dpc: (DpcSettings & { output: "colour" | "x" | "y" | "idpc" }) | null;
  /** PCA (origin "pca", "pca-factors", "pca-denoised"): how it was made. */
  pca: PcaInfo | null;
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


type Versioned = { revision: number; filter_rev: number; data_rev?: number; insitu?: { view?: { mode: string; start: number; end: number } | null } | null };

/** Changes whenever the data shown for a dataset does (recomputed, rebinned,
 * filtered or replaced by the next frame), for cache keys. */
export const dataVersion = (d: Versioned) => `${d.revision}.${d.filter_rev}.${d.data_rev ?? 0}`;

export type FitQuantity = "counts" | "at%";

/** Like ``dataVersion`` but unchanged by a new frame: contrast is redone when
 * it changes. An in-situ sum is on another scale than a frame or a mean, and
 * grows with the frames summed, so switching to or from it, or changing its
 * range, changes it too. */
export const contrastVersion = (d: Versioned) => {
  const v = d.insitu?.view;
  return `${d.revision}.${d.filter_rev}${v?.mode === "sum" ? `.sum${v.start}-${v.end}` : ""}`;
};

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
  /** Background window (eV): the background is fitted there alone, then
   * fixed while the edges are fitted. null: exSpy's smart fit. */
  bg_lo: number | null;
  bg_hi: number | null;
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
  /** Opened from a saved analysis without its raw file (its stored data). */
  raw_missing?: boolean;
  /** Opened from a copy of its raw file embedded in a saved analysis. */
  embedded?: boolean;
  datasets: DatasetSummary[];
}

/** How much a saved analysis holds: raw data referenced, or the raw datasets' data stored too. */
export type SaveLevel = "link" | "results" | "raw";

/** A raw file of a saved analysis that can't be found. */
export interface MissingRaw {
  id: string;
  name: string;
  path: string;
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
  /** When it was submitted, started and finished (seconds since the epoch). */
  created?: number;
  started?: number | null;
  finished?: number | null;
  /** Left out of the list of jobs (``listJobs``). */
  result?: {
    file: FileSummary;
    skipped?: string[];
    /** Curve fit of every pixel: the pixels whose fit failed. */
    failed?: number;
    dataset_id?: string;
    /** Zero-loss alignment: the other files changed, the ids aligned, and how far their axes moved (eV). */
    partner_files?: FileSummary[];
    aligned?: string[];
    shift?: number;
    /** PCA: the PCA stack made. */
    pca_id?: string;
    /** Opening an analysis: the files its drift corrections changed. */
    files?: FileSummary[];
    /** Disk detection: the Bragg vectors table's id and how many disks. */
    id?: string;
    count?: number;
    /** Finding the beam: the shift images' ids and each shift's range (px). */
    ids?: string[];
    range?: { x: [number, number]; y: [number, number] };
    /** A Structure's templates made: how many. */
    templates?: number;
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

/** An example dataset: bundled, simulated, or pyxem's (downloaded on first use). */
export interface ExampleInfo {
  id: string;
  title: string;
  description: string;
  /** What it is for: the menu's group. */
  group: string;
  /** Not downloaded yet: its size (MB). */
  download_mb?: number;
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
  /** ``fileIds``: the files saved (and what they link to). */
  saveProject: (
    path: string, views: Record<string, unknown>, overwrite: boolean, level: SaveLevel, fileIds: string[],
  ) =>
    request<{ path: string }>("/api/project/save", {
      method: "POST",
      body: JSON.stringify({ path, views, overwrite, level, file_ids: fileIds }),
    }),
  /** Bytes a save of ``fileIds`` would store at each level (before
   * compression), and the files it would hold (those and what they link to). */
  projectEstimate: (fileIds: string[]) =>
    request<Record<SaveLevel, number> & { file_ids: string[] }>("/api/project/estimate", {
      method: "POST",
      body: JSON.stringify({ file_ids: fileIds }),
    }),
  /** ``locations``: where raw files that moved are (file id -> path). */
  openProject: (path: string, locations: Record<string, string> = {}) =>
    request<{
      files: FileSummary[];
      file_ids: string[];
      views: Record<string, unknown>;
      warnings: string[];
      /** Drift corrections redone after opening. */
      job: JobSummary | null;
    }>("/api/project/open", {
      method: "POST",
      body: JSON.stringify({ path, locations }),
    }),
  /** The raw files that have to be located before ``path`` opens. */
  checkProject: (path: string, locations: Record<string, string> = {}) =>
    request<{ missing: MissingRaw[] }>("/api/project/check", {
      method: "POST",
      body: JSON.stringify({ path, locations }),
    }),
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
  /** Curve fitting (M21): HyperSpy's 1D components offered. */
  curveComponents: () => request<CurveCatalogue>("/api/curve-fit/components"),
  /** Starting values of one component (its ``estimate_parameters`` over the fit range). */
  curveEstimate: (id: string, source: FitSource, model: CurveModel, component: string) =>
    request<{ params: CurveParam[]; pixels: number }>(`/api/datasets/${id}/curve-fit/estimate`, {
      method: "POST",
      body: JSON.stringify({ source, model, component }),
    }),
  /** Fit the model to one spectrum (``fit`` false: draw it with its current values). */
  curveFit: (id: string, source: FitSource, model: CurveModel, fit = true, signal?: AbortSignal) =>
    request<CurveFitResult>(`/api/datasets/${id}/curve-fit`, {
      method: "POST",
      body: JSON.stringify({ source, model, fit }),
      signal,
    }),
  /** Job: fit the model at every pixel (``source``: the spectrum it was made
   * on); its result's file has a reduced χ² map with the parameters' maps under it. */
  startCurveFitMaps: (id: string, source: FitSource, model: CurveModel) =>
    request<JobSummary>(`/api/datasets/${id}/curve-fit/maps`, {
      method: "POST",
      body: JSON.stringify({ source, model }),
    }),
  /** A curve model at the cursor's block, from its maps (with the data there). */
  curveModelPixel: (id: string, x: number, y: number, size: number, signal?: AbortSignal) =>
    request<CurveFitResult & { data: number[] }>(`/api/datasets/${id}/curve-model/pixel?x=${x}&y=${y}&size=${size}`, { signal }),
  /** The curve fit of every pixel at the cursor's block (404: none). */
  curvePixelFit: (id: string, x: number, y: number, size: number, signal?: AbortSignal) =>
    request<CurveFitResult>(`/api/datasets/${id}/curve-fit/pixel?x=${x}&y=${y}&size=${size}`, { signal }),
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
  setMicroscope: (id: string, params: Partial<MicroscopeInfo>) =>
    request<DatasetSummary>(`/api/datasets/${id}/microscope`, { method: "PATCH", body: JSON.stringify(params) }),
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
  /** Not ``inPlace``: copies are rebinned (the linked ones' too). */
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
  /** ZLP alignment by hand: move the energy axis of an EELS spectrum image by
   * ``shift`` (with what was made from it); not ``inPlace``: a shifted copy. */
  shiftEnergy: (id: string, shift: number, inPlace = true) =>
    request<DatasetAndFile>(`/api/datasets/${id}/energy-shift`, {
      method: "POST",
      body: JSON.stringify({ shift, in_place: inPlace }),
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
  /** Find the direct beam in every pattern of 4D-STEM data (a job). */
  /** The polar transform of every pattern (a job): a copy under the data, or in place. */
  polarTransform: (id: string, options: object, inPlace: boolean) =>
    request<JobSummary>(`/api/datasets/${id}/4d/polar`, { method: "POST", body: JSON.stringify({ options, in_place: inPlace }) }),
  /** Simulate a Structure's templates (a job): a new Templates object under it. */
  makeTemplates: (id: string, options: TemplateOptions) =>
    request<JobSummary>(`/api/datasets/${id}/templates/make`, { method: "POST", body: JSON.stringify({ options }) }),
  /** A Templates object's inverse pole figure (orix). */
  templatesIpf: (id: string) => request<TemplatesIpf>(`/api/datasets/${id}/templates`),
  /** Template ``index`` of a Templates object. */
  template: (id: string, index: number) => request<TemplatePattern>(`/api/datasets/${id}/templates/${index}`),
  /** The template nearest the zone axis [u v w]. */
  nearestTemplate: (id: string, uvw: number[]) =>
    request<{ index: number }>(`/api/datasets/${id}/templates/nearest?u=${uvw[0]}&v=${uvw[1]}&w=${uvw[2]}`),
  /** The template nearest the beam direction [u v w], its reflections on that zone. */
  templateToward: (id: string, uvw: number[]) =>
    request<{ index: number; angle: number; hkl: number[][]; intensity: number[] }>(
      `/api/datasets/${id}/templates/toward?u=${uvw[0]}&v=${uvw[1]}&w=${uvw[2]}`),
  /** pyxem's template matching at scan ``positions`` of the data's polar transform. */
  /** An orientation map at scan positions: its best template there, described as a match is. */
  mapAt: (omapId: string, positions: number[][]) =>
    request<{ matches: MatchResult[] }>(`/api/datasets/${omapId}/orientation/at`, {
      method: "POST", body: JSON.stringify({ positions }),
    }),
  /** A template at an in-plane rotation described as a match (one of a match's ranked). */
  describeTemplate: (id: string, row: Pick<RankedMatch, "index" | "rotation" | "mirror" | "correlation">) =>
    request<MatchResult>(`/api/datasets/${id}/templates/describe`, {
      method: "POST", body: JSON.stringify({ index: row.index, rotation: row.rotation, mirror: row.mirror, correlation: row.correlation }),
    }),
  matchAt: (id: string, templates: string, positions: number[][], polar?: string | null) =>
    request<{ polar: string; matches: MatchResult[] }>(`/api/datasets/${id}/4d/orientation/match`, {
      method: "POST", body: JSON.stringify({ templates, positions, polar: polar ?? null }),
    }),
  /** orix's plots of a Structure. */
  structurePlots: (id: string) => request<StructurePlots>(`/api/datasets/${id}/structure/plots`),
  findBeam: (id: string, options: object) =>
    request<JobSummary>(`/api/datasets/${id}/4d/beam/find`, { method: "POST", body: JSON.stringify({ options }) }),
  /** Shift every pattern so the beam is at the centre (a job; not ``inPlace``: a copy),
   * or (``move`` "axes") only put the pattern axes' 0 on the beam. */
  centreBeam: (id: string, plane: object, inPlace: boolean, move: "patterns" | "axes" = "patterns") =>
    request<JobSummary>(`/api/datasets/${id}/4d/beam/centre`, {
      method: "POST",
      body: JSON.stringify({ plane, in_place: inPlace, move }),
    }),
  /** Subtract the background of every pattern of 4D-STEM data (a job), in place or as a copy. */
  subtractBackground: (id: string, options: object, inPlace: boolean) =>
    request<JobSummary>(`/api/datasets/${id}/4d/background/run`, {
      method: "POST",
      body: JSON.stringify({ options, in_place: inPlace }),
    }),
  /** Find the Bragg disks of every pattern of 4D-STEM data (a job). */
  findDisks: (id: string, options: object) =>
    request<JobSummary>(`/api/datasets/${id}/4d/disks/run`, { method: "POST", body: JSON.stringify({ options }) }),
  /** Every queued or running job, then the ``recent`` latest to finish (no results). */
  listJobs: (recent = 3) => request<JobSummary[]>(`/api/jobs?recent=${recent}`),
  /** A new spectrum image under ``id``, cropped to a rectangle (navigation
   * pixels, as a region's spectrum) and/or an energy window. */
  cropDataset: (id: string, crop: { region?: object; lo?: number; hi?: number }) =>
    request<{ id: string; file: FileSummary }>(`/api/datasets/${id}/crop`, { method: "POST", body: JSON.stringify(crop) }),
  /** Job: decompose a spectrum image; its result's ``pca_id`` is the PCA made (or remade). */
  startPca: (id: string, options: PcaOptions, normalise: PcaInfo["normalise"] = null) =>
    request<JobSummary>(`/api/datasets/${id}/pca`, { method: "POST", body: JSON.stringify({ ...options, normalise }) }),
  /** The spectrum image rebuilt from a PCA's first ``n`` components (a new SI, ``id``). */
  denoisePca: (id: string, n: number) =>
    request<{ id: string; file: FileSummary }>(`/api/datasets/${id}/pca/denoise`, {
      method: "POST",
      body: JSON.stringify({ n }),
    }),
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
  /** ``also``: other stacks (see ``registrationPartners``) given the same shifts. */
  applyAlignment: (id: string, inPlace = true, also: string[] = []) =>
    request<
      DatasetAndFile & {
        average: DatasetSummary;
        aligned_id: string;
        also: { id: string; aligned_id: string }[];
        /** Every file changed. */
        files: FileSummary[];
      }
    >(`/api/datasets/${id}/registration/apply`, {
      method: "POST",
      body: JSON.stringify({ in_place: inPlace, also }),
    }),
  /** What a DPC of image ``id`` can be made from, the quadrants chosen, and
   * the settings of DPC images made from them (if any). */
  dpcInfo: (id: string) =>
    request<{
      candidates: DpcCandidate[];
      choice: Record<string, string>;
      settings: DpcSettings | null;
      /** The detector rotation recorded in the file (degrees), if any. */
      recorded: number | null;
    }>(
      `/api/datasets/${id}/dpc`,
    ),
  /** The detector's rotation found from the data (least curl). */
  dpcRotation: (inputs: Record<string, string>, normalise: boolean) =>
    request<{ theta: number; mirror: boolean; curl_ratio: number }>("/api/dpc/rotation", {
      method: "POST",
      body: JSON.stringify({ inputs, normalise }),
    }),
  /** Make (or make again) DPC x, DPC y and iDPC images. */
  refineFftPeaks: (id: string, peaks: [number, number][], options: FftPeakOptions) =>
    request<{ peaks: RefinedPeak[]; shape: [number, number] }>(`/api/datasets/${id}/atomic/fft-peaks`, {
      method: "POST",
      body: JSON.stringify({ peaks, ...options }),
    }),
  atomicFourier: (
    id: string,
    body: FourierSettings & {
      peaks: Record<string, [number, number]>;
      per_peak: number[];
      outputs: AtomicOutput[];
      tensor: [number, number] | null;
    },
  ) => request<JobSummary>(`/api/datasets/${id}/atomic/fourier`, { method: "POST", body: JSON.stringify(body) }),
  findColumns: (id: string, s: ColumnSettings) =>
    request<{ columns: [number, number][] }>(`/api/datasets/${id}/atomic/columns/find`, { method: "POST", body: JSON.stringify(s) }),
  testColumn: (id: string, s: ColumnSettings, at: [number, number]) =>
    request<ColumnTest>(`/api/datasets/${id}/atomic/columns/test`, { method: "POST", body: JSON.stringify({ ...s, at }) }),
  refineColumns: (id: string, s: ColumnSettings, columns: [number, number][]) =>
    request<JobSummary>(`/api/datasets/${id}/atomic/columns/refine`, { method: "POST", body: JSON.stringify({ ...s, columns }) }),
  appendColumns: (columns: string, s: ColumnSettings, add: [number, number][]) =>
    request<JobSummary>(`/api/datasets/${columns}/atomic/columns/append`, { method: "POST", body: JSON.stringify({ ...s, columns: add }) }),
  setPicks: (id: string, picks: Record<string, unknown>) =>
    request<void>(`/api/datasets/${id}/atomic/picks`, { method: "PUT", body: JSON.stringify({ picks }) }),
  cellPositions: (columns: string, reference: string, vectors: [[number, number], [number, number]]) =>
    request<{ u: (number | null)[]; v: (number | null)[] }>(`/api/datasets/${columns}/atomic/columns/cell`, {
      method: "POST",
      body: JSON.stringify({ reference, vectors }),
    }),
  cellFit: (columns: string, body: CellSitesRequest) =>
    request<{ labels: (number | null)[]; sites: CellSiteFit[]; unpicked: { u: number; v: number; count: number }[]; left_out: number }>(
      `/api/datasets/${columns}/atomic/columns/cell/fit`,
      { method: "POST", body: JSON.stringify(body) },
    ),
  sublatticesByCell: (columns: string, body: CellSitesRequest) =>
    request<{ file: FileSummary; ids: string[] }>(`/api/datasets/${columns}/atomic/columns/sublattices/cell`, {
      method: "PUT",
      body: JSON.stringify(body),
    }),
  cageCentres: (nb: string, vectors: [number, number][], threshold: number, nearest = false) =>
    request<{ columns: [number, number][]; added: number; id: string; name: string }>(
      `/api/datasets/${nb}/atomic/neighbourhood/centres`,
      { method: "POST", body: JSON.stringify({ vectors, threshold, nearest }) },
    ),
  cageDisplacement: (nb: string, vectors: [number, number][], threshold: number, nearest = false) =>
    request<{ file: FileSummary; id: string }>(`/api/datasets/${nb}/atomic/neighbourhood/displacement`, {
      method: "POST",
      body: JSON.stringify({ vectors, threshold, nearest }),
    }),
  columnsTable: (id: string) =>
    request<{ fields: string[]; rows: (number | null)[][] }>(`/api/datasets/${id}/atomic/columns`),
  setSublattices: (id: string, labels: (number | null)[], names: string[]) =>
    request<{ file: FileSummary; ids: string[] }>(`/api/datasets/${id}/atomic/columns/sublattices`, {
      method: "PUT",
      body: JSON.stringify({ labels, names }),
    }),
  kmeans: (values: (number | null)[], k: number) =>
    request<{ centres: number[] }>("/api/atomic/kmeans", { method: "POST", body: JSON.stringify({ values, k }) }),
  makeNeighbourhood: (origins: string, neighbours: string, k: number, replace?: string) =>
    request<{ file: FileSummary; id: string }>(`/api/datasets/${origins}/atomic/neighbourhood`, {
      method: "POST",
      body: JSON.stringify({ neighbours, k, replace: replace ?? null }),
    }),
  clusterNeighbourhood: (id: string, n: number) =>
    request<{ file: FileSummary }>(`/api/datasets/${id}/atomic/neighbourhood/clusters`, {
      method: "POST",
      body: JSON.stringify({ n }),
    }),
  neighbourhood: (id: string) =>
    request<{ dy: number[]; dx: number[]; clusters: [number, number][] }>(`/api/datasets/${id}/atomic/neighbourhood`),
  makeBonds: (id: string, name: string, vector: [number, number], threshold: number) =>
    request<{ file: FileSummary; id: string }>(`/api/datasets/${id}/atomic/bonds`, {
      method: "POST",
      body: JSON.stringify({ name, vector, threshold }),
    }),
  bondsTable: (id: string) =>
    request<{ fields: string[]; rows: (number | null)[][] }>(`/api/datasets/${id}/atomic/bonds`),
  pldDetect: (id: string, body: { count: number; within: number; noise?: number }) =>
    request<{ peaks: PldDetected[]; basis: [[number, number], [number, number]] | null; families: PldFamily[] }>(
      `/api/datasets/${id}/atomic/pld/detect`,
      { method: "POST", body: JSON.stringify(body) },
    ),
  pldReference: (id: string, peaks: [number, number][], radius: number) =>
    request<{ file: FileSummary; ids: string[]; peaks: RefinedPeak[] }>(`/api/datasets/${id}/atomic/pld/reference`, {
      method: "POST",
      body: JSON.stringify({ peaks, radius }),
    }),
  pldDisplacements: (columns: string, threshold: number) =>
    request<JobSummary>(`/api/datasets/${columns}/atomic/pld/displacements`, {
      method: "POST",
      body: JSON.stringify({ threshold }),
    }),
  maskValues: (id: string, smooth: number) =>
    request<{ values: number[]; otsu: number; min: number; max: number }>(`/api/datasets/${id}/mask/values`, {
      method: "POST",
      body: JSON.stringify({ smooth }),
    }),
  makeMask: (id: string, options: MaskOptions, replace?: string) =>
    request<{ file: FileSummary; id: string }>(`/api/datasets/${id}/mask`, {
      method: "POST",
      body: JSON.stringify({ ...options, replace: replace ?? null }),
    }),
  sublatticeFromMask: (columns: string, mask: string) =>
    request<{ file: FileSummary; id: string }>(`/api/datasets/${columns}/atomic/columns/sublattices/mask`, {
      method: "POST",
      body: JSON.stringify({ mask }),
    }),
  displacementsTable: (id: string) =>
    request<{ fields: string[]; rows: (number | null)[][] }>(`/api/datasets/${id}/atomic/displacements`),
  makeDpc: (settings: DpcSettings) =>
    request<{ ids: Record<"colour" | "x" | "y" | "idpc", string>; files: FileSummary[] }>("/api/dpc", {
      method: "POST",
      body: JSON.stringify(settings),
    }),
  /** The other stacks a drift measured on ``id`` applies to (same frames and
   * frame size, same file or experiment). */
  registrationPartners: (id: string) =>
    request<{ id: string; title: string }[]>(`/api/datasets/${id}/registration/partners`),
  /** The drift-corrected stacks whose measured shifts ``id`` can be aligned with (a HAADF first). */
  alignmentSources: (id: string) => request<{ id: string; title: string }[]>(`/api/datasets/${id}/alignment/sources`),
  /** Align a stack with the shifts measured on ``source`` (shifted and cropped as it is). */
  alignFrom: (id: string, source: string) =>
    request<DatasetAndFile>(`/api/datasets/${id}/alignment/from`, { method: "POST", body: JSON.stringify({ source }) }),
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
