import { ApiError } from "./client";

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

/** A 2D frame as displayed: possibly binned relative to the full image. */
export interface Frame {
  data: Float32Array;
  width: number;
  height: number;
  bin: number;
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
