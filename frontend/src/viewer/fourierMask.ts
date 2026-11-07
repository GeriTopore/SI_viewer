import type { MaskSettings } from "../api/client";

/**
 * rigidregistration's Fourier masks (``imstack.makeFourierMask``) as a
 * function of spatial frequency f in cycles per frame pixel. The package
 * defines them on FFT pixels k of an N-pixel FFT, with the cutoff at
 * k_max = N / n / 2; with f = k / N the cutoff is f_c = 1 / (2n), whatever
 * the FFT size, so the same mask can be drawn on any FFT of the frame.
 */
export function maskValue(mask: MaskSettings, f: number): number {
  const { type, n } = mask;
  const fc = 1 / (2 * n);
  switch (type) {
    case "bandpass":
      return f < fc ? Math.sin(2 * n * Math.PI * f) ** 2 : 0;
    case "hann":
      return f < fc ? Math.cos(n * Math.PI * f) ** 2 : 0;
    case "hamming":
      return f < fc ? 27 / 50 + (23 / 50) * Math.cos((Math.PI * f) / fc) : 0;
    case "blackman":
      return f < fc ? 21 / 50 + 0.5 * Math.cos((Math.PI * f) / fc) + (2 / 25) * Math.cos((2 * Math.PI * f) / fc) : 0;
    case "gaussian":
      return Math.exp(-((f / (fc / 3)) ** 2));
    default:
      return 1;
  }
}

/** The cutoff frequency (cycles per pixel) of a mask; null for "none". */
export const maskCutoff = (mask: MaskSettings): number | null => (mask.type === "none" ? null : 1 / (2 * mask.n));
