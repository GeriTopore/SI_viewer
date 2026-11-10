/// Spectrum of a square image patch, off the main thread.
///
/// The patch (size × size) has its mean removed and, if asked, a Hann window
/// applied (so the patch edges don't add a bright cross), is zero-padded to n × n (n a
/// power of two) and transformed with a radix-2 FFT. The result is the
/// amplitude |F|, shifted so zero frequency is at (n/2, n/2); the display
/// scale (linear or log) is applied by the caller.

export interface FftRequest {
  id: number;
  patch: Float32Array;
  size: number;
  n: number;
  /** Apply a Hann window before transforming. */
  hann: boolean;
}

export interface FftResult {
  id: number;
  n: number;
  /** Amplitude |F|, n × n, zero frequency at the centre. */
  amp: Float32Array;
}

const twiddleCache = new Map<number, { cos: Float32Array; sin: Float32Array; rev: Uint32Array }>();

function tables(n: number) {
  let t = twiddleCache.get(n);
  if (!t) {
    const cos = new Float32Array(n / 2);
    const sin = new Float32Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      cos[i] = Math.cos((-2 * Math.PI * i) / n);
      sin[i] = Math.sin((-2 * Math.PI * i) / n);
    }
    const bits = Math.log2(n);
    const rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      rev[i] = r;
    }
    t = { cos, sin, rev };
    twiddleCache.set(n, t);
  }
  return t;
}

/** In-place FFT of one row (re, im of length n). */
function fft1d(re: Float32Array, im: Float32Array, n: number) {
  const { cos, sin, rev } = tables(n);
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) {
      let t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const step = n / len;
    for (let start = 0; start < n; start += len) {
      for (let k = 0; k < half; k++) {
        const wr = cos[k * step];
        const wi = sin[k * step];
        const a = start + k;
        const b = a + half;
        const xr = re[b] * wr - im[b] * wi;
        const xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr;
        im[b] = im[a] - xi;
        re[a] += xr;
        im[a] += xi;
      }
    }
  }
}

function amplitudeSpectrum({ patch, size, n, hann: windowed }: FftRequest): Float32Array {
  // Mean of the finite pixels; NaN (masked) pixels count as the mean.
  let sum = 0;
  let count = 0;
  for (let i = 0; i < patch.length; i++) {
    const v = patch[i];
    if (Number.isFinite(v)) {
      sum += v;
      count++;
    }
  }
  const mean = count ? sum / count : 0;
  const hann = new Float32Array(size);
  for (let i = 0; i < size; i++) hann[i] = windowed ? 0.5 - 0.5 * Math.cos((2 * Math.PI * (i + 0.5)) / size) : 1;

  const re = new Float32Array(n * n);
  const im = new Float32Array(n * n);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const v = patch[y * size + x];
      re[y * n + x] = Number.isFinite(v) ? (v - mean) * hann[x] * hann[y] : 0;
    }
  }

  const rowRe = new Float32Array(n);
  const rowIm = new Float32Array(n);
  // Rows (only the first ``size`` have data; the padding stays zero).
  for (let y = 0; y < size; y++) {
    rowRe.set(re.subarray(y * n, y * n + n));
    rowIm.fill(0);
    fft1d(rowRe, rowIm, n);
    re.set(rowRe, y * n);
    im.set(rowIm, y * n);
  }
  // Columns
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) {
      rowRe[y] = re[y * n + x];
      rowIm[y] = im[y * n + x];
    }
    fft1d(rowRe, rowIm, n);
    for (let y = 0; y < n; y++) {
      re[y * n + x] = rowRe[y];
      im[y * n + x] = rowIm[y];
    }
  }

  // Amplitude |F|, shifted so (0, 0) frequency is at the centre.
  const amp = new Float32Array(n * n);
  const h = n / 2;
  for (let y = 0; y < n; y++) {
    const sy = (y + h) % n;
    for (let x = 0; x < n; x++) {
      const i = y * n + x;
      amp[sy * n + ((x + h) % n)] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
    }
  }
  amp[h * n + h] = 0; // what is left of the mean
  return amp;
}

self.onmessage = (e: MessageEvent<FftRequest>) => {
  const amp = amplitudeSpectrum(e.data);
  const result: FftResult = { id: e.data.id, n: e.data.n, amp };
  (self as unknown as Worker).postMessage(result, [amp.buffer]);
};
