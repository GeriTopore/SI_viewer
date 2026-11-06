// Contrast statistics from a strided sample of the data: fast and accurate
// enough for percentiles and a display histogram, even for 4k × 4k images.

const MAX_SAMPLES = 65536;

/** Sorted finite values sampled evenly from ``data``. */
export function sortedSample(data: Float32Array): Float32Array {
  const step = Math.max(1, Math.floor(data.length / MAX_SAMPLES));
  const out = new Float32Array(Math.ceil(data.length / step));
  let n = 0;
  for (let i = 0; i < data.length; i += step) {
    const v = data[i];
    if (Number.isFinite(v)) out[n++] = v;
  }
  return out.subarray(0, n).sort();
}

export function percentile(sorted: Float32Array, p: number): number {
  if (!sorted.length) return 0;
  const i = Math.round((p / 100) * (sorted.length - 1));
  return sorted[Math.min(sorted.length - 1, Math.max(0, i))];
}

/** Auto contrast: clip 0.5 % at both ends (never a zero-width range). */
export function autoRange(sorted: Float32Array, clip = 0.5): [number, number] {
  let lo = percentile(sorted, clip);
  let hi = percentile(sorted, 100 - clip);
  if (hi <= lo) {
    lo = sorted.length ? sorted[0] : 0;
    hi = sorted.length ? sorted[sorted.length - 1] : 1;
  }
  if (hi <= lo) hi = lo + 1;
  return [lo, hi];
}

export function fullRange(sorted: Float32Array): [number, number] {
  return autoRange(sorted, 0);
}

export function histogram(sorted: Float32Array, lo: number, hi: number, bins: number): Float32Array {
  const counts = new Float32Array(bins);
  const width = (hi - lo) / bins || 1;
  for (const v of sorted) {
    const b = Math.floor((v - lo) / width);
    if (b >= 0 && b < bins) counts[b]++;
  }
  return counts;
}
