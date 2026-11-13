import { ApiError, type MaskEdits, type MaskSettings } from "./client";

export interface ArrayResult {
  data: Float32Array;
  shape: number[];
  meta: Record<string, unknown>;
}

/** Fetch a float32 array from an endpoint that returns raw bytes. */
export async function fetchArray(url: string, signal?: AbortSignal, init?: RequestInit): Promise<ArrayResult> {
  const res = await fetch(url, { ...init, signal });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  const shape = (res.headers.get("X-Array-Shape") ?? "").split(",").map(Number);
  const meta = JSON.parse(res.headers.get("X-Array-Meta") ?? "{}");
  return { data: new Float32Array(await res.arrayBuffer()), shape, meta };
}

/** A 2D frame as displayed: ``bin`` is the number of pixels of the full
 * image per pixel of the frame (a binned preview; or a map drawn over a finer
 * navigator, see ``views/si/scan.ts``). */
export interface Frame {
  data: Float32Array;
  width: number;
  height: number;
  bin: number;
  /** Brightness of each pixel (see ``LayerData.shade``): a DPC image's magnitude. */
  shade?: Float32Array;
  shadeHi?: number;
}

export async function fetchFrame(
  datasetId: string,
  params: { index?: number; projection?: string; bin?: number } = {},
  signal?: AbortSignal,
): Promise<Frame> {
  const q = new URLSearchParams();
  if (params.index !== undefined) q.set("index", String(params.index));
  if (params.projection) q.set("projection", params.projection);
  if (params.bin && params.bin > 1) q.set("bin", String(params.bin));
  const { data, shape } = await fetchArray(`/api/datasets/${datasetId}/frame?${q}`, signal);
  return { data, height: shape[0], width: shape[1], bin: params.bin ?? 1 };
}

/** A DPC image (or a frame of a DPC stack, or its projection): the direction
 * of the deflection, with its magnitude as the brightness (up to the 99.5th
 * percentile at full). */
export async function fetchDpcFrame(
  datasetId: string,
  params: { index?: number; projection?: string; bin?: number } = {},
  signal?: AbortSignal,
): Promise<Frame> {
  const q = new URLSearchParams();
  if (params.index !== undefined) q.set("index", String(params.index));
  if (params.projection) q.set("projection", params.projection);
  if (params.bin && params.bin > 1) q.set("bin", String(params.bin));
  const { data, shape } = await fetchArray(`/api/datasets/${datasetId}/dpc/frame?${q}`, signal);
  const [, height, width] = shape;
  const n = height * width;
  const shade = data.subarray(n, 2 * n);
  const sorted = Float32Array.from(shade).sort();
  const shadeHi = sorted[Math.min(n - 1, Math.floor(n * 0.995))] || 1;
  return { data: data.subarray(0, n), height, width, bin: params.bin ?? 1, shade, shadeHi };
}

/** The spectrum of component ``index`` (0-based) of a PCA, with its energy axis. */
export async function fetchPcaFactor(
  pcaId: string, index: number, signal?: AbortSignal,
): Promise<{ data: Float32Array; offset: number; scale: number; units: string }> {
  const { data, meta } = await fetchArray(`/api/datasets/${pcaId}/pca/factor?index=${index}`, signal);
  return { data, offset: Number(meta.offset), scale: Number(meta.scale), units: String(meta.units ?? "") };
}

/** A stack's frame (or projection) as shown: a DPC stack's in colour. */
export const fetchStackFrame = (
  d: { id: string; dpc?: { output: string } | null },
  params: { index?: number; projection?: string; bin?: number },
) => (d.dpc?.output === "colour" ? fetchDpcFrame(d.id, params) : fetchFrame(d.id, params));

/** The ADF image of the frame (or range) an in-situ series holds, at the ADF's
 * own resolution (finer than the scan for a sub-scan). */
export async function fetchInsituAdf(datasetId: string, signal?: AbortSignal): Promise<Frame> {
  const { data, shape } = await fetchArray(`/api/datasets/${datasetId}/insitu/adf`, signal);
  return { data, height: shape[0], width: shape[1], bin: 1 };
}

export async function fetchSpectrum(datasetId: string, signal?: AbortSignal): Promise<Float32Array> {
  return (await fetchArray(`/api/datasets/${datasetId}/spectrum`, signal)).data;
}

/** A summed spectrum and the number of pixels it covers. */
export interface PixelsSpectrum {
  data: Float32Array;
  pixels: number;
}

const asPixels = ({ data, meta }: ArrayResult): PixelsSpectrum => ({ data, pixels: Number(meta.pixels ?? 1) });

/** Summed spectrum of the size × size block around pixel (x, y). */
export async function fetchPixelSpectrum(si: string, x: number, y: number, size: number) {
  return asPixels(await fetchArray(`/api/datasets/${si}/pixel-spectrum?x=${x}&y=${y}&size=${size}`));
}

/** Summed spectrum over a region (rect/ellipse bounds or polygon points). */
export async function fetchRegionSpectrum(si: string, region: object) {
  return asPixels(
    await fetchArray(`/api/datasets/${si}/region-spectrum`, undefined, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(region),
    }),
  );
}

/** Map of the counts between energies lo and hi, as a frame. */
export async function fetchEnergyMap(si: string, lo: number, hi: number): Promise<Frame> {
  const { data, shape } = await fetchArray(`/api/datasets/${si}/energy-map?lo=${lo}&hi=${hi}`);
  return { data, height: shape[0], width: shape[1], bin: 1 };
}

/** Square frames of a (k, rows, cols) array, e.g. a registration's panels. */
export function splitFrames({ data, shape }: ArrayResult): Frame[] {
  const [k, height, width] = shape;
  return Array.from({ length: k }, (_, i) => ({
    data: data.subarray(i * height * width, (i + 1) * height * width),
    width,
    height,
    bin: 1,
  }));
}

/** Frame i of a stack being registered, as loaded. */
export async function fetchRegistrationFrame(stack: string, i: number): Promise<Frame> {
  const { data, shape } = await fetchArray(`/api/datasets/${stack}/registration/frame?i=${i}`);
  return { data, height: shape[0], width: shape[1], bin: 1 };
}

/** Frame i shifted into register, cropped to the area every frame covers. */
export async function fetchAlignedFrame(stack: string, i: number): Promise<Frame> {
  const { data, shape } = await fetchArray(`/api/datasets/${stack}/registration/aligned?i=${i}`);
  return { data, height: shape[0], width: shape[1], bin: 1 };
}

/** The average of the aligned frames, cropped like them. */
export async function fetchAlignedAverage(stack: string): Promise<Frame> {
  const { data, shape } = await fetchArray(`/api/datasets/${stack}/registration/average`);
  return { data, height: shape[0], width: shape[1], bin: 1 };
}

/** Correlation of frames i, j with a Fourier mask (zero shift at the centre);
 * cross (cc), mutual (mc) or phase (pc). */
export async function fetchCorrelation(
  stack: string, i: number, j: number, mask: MaskSettings, corr = "cc",
): Promise<Frame> {
  const { type, n, n2, theta } = mask;
  const { data, shape } = await fetchArray(
    `/api/datasets/${stack}/registration/correlation?i=${i}&j=${j}&type=${type}&n=${n}&n2=${n2}&theta=${theta}&corr=${corr}`,
  );
  return { data, height: shape[0], width: shape[1], bin: 1 };
}

/** Which pair shifts the outlier settings keep (1) or mask (0). */
export async function fetchOutlierMask(stack: string, outliers: object, badFrames: number[], edits: MaskEdits) {
  const r = await fetchArray(`/api/datasets/${stack}/registration/outlier-mask`, undefined, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ outliers, bad_frames: badFrames, edits }),
  });
  const [n] = r.shape;
  return { kept: r.data, n, masked: Number(r.meta.masked), pairs: Number(r.meta.pairs) };
}

/** The corrected shift matrices: Δx, Δy and which entries are known (1). */
export async function fetchCorrectedMatrices(stack: string) {
  const [dx, dy, known] = splitFrames(await fetchArray(`/api/datasets/${stack}/registration/corrected`));
  return { dx, dy, known: known.data };
}

/** The shift matrices of a registration: Δx and Δy (px) of frame j relative to i. */
export async function fetchShiftMatrices(stack: string) {
  const r = await fetchArray(`/api/datasets/${stack}/registration/matrices`);
  const [dx, dy] = splitFrames(r);
  return { dx, dy, first: Number(r.meta.first) };
}

/** Cross-correlations of frames i, j for several cutoffs n (an elliptical
 * mask keeps its shape). */
export async function fetchCompareN(stack: string, i: number, j: number, mask: MaskSettings, ns: number[]) {
  const q = ns.map((n) => `n=${n}`).join("&");
  const { type, n, n2, theta } = mask;
  return splitFrames(
    await fetchArray(
      `/api/datasets/${stack}/registration/compare-n?i=${i}&j=${j}&type=${type}&n1=${n}&n2=${n2}&theta=${theta}&${q}`,
    ),
  );
}
