import type { MaskSettings } from "../api/client";
import { formatValue } from "../format";

/**
 * rigidregistration's Fourier masks (``imstack.makeFourierMask`` and
 * ``makeFourierMask_eg``) as a function of spatial frequency, in cycles per
 * frame pixel along x (columns) and y (rows). The package defines them on FFT
 * pixels k of an N-pixel FFT, with the cutoff at k_max = N / n / 2; with
 * f = k / N the cutoff is f_c = 1 / (2n), whatever the FFT size, so the same
 * mask can be drawn on any FFT of the frame.
 */
export function maskValue(mask: MaskSettings, fx: number, fy: number): number {
  const { type, n } = mask;
  if (type === "elliptical") {
    // σ = cutoff / 3 along each axis; the package's kx is the row axis.
    const s1 = 1 / (2 * n) / 3;
    const s2 = 1 / (2 * mask.n2) / 3;
    const t = (mask.theta * Math.PI) / 180;
    const [c, s] = [Math.cos(t), Math.sin(t)];
    const a = (c * c) / (2 * s1 * s1) + (s * s) / (2 * s2 * s2);
    const b = -Math.sin(2 * t) / (4 * s1 * s1) + Math.sin(2 * t) / (4 * s2 * s2);
    const cc = (s * s) / (2 * s1 * s1) + (c * c) / (2 * s2 * s2);
    return Math.exp(-(a * fy * fy + 2 * b * fy * fx + cc * fx * fx));
  }
  const f = Math.hypot(fx, fy);
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

/** The mask's cutoff (cycles per pixel) as an ellipse: semi-axes along its
 * first and second axes and the first axis's direction on screen (degrees,
 * from +x towards +y down). A circle for the round masks; null for "none". */
export function maskCutoff(mask: MaskSettings): { r1: number; r2: number; angle: number } | null {
  if (mask.type === "none") return null;
  if (mask.type !== "elliptical") return { r1: 1 / (2 * mask.n), r2: 1 / (2 * mask.n), angle: 0 };
  // The first axis is (cos θ, −sin θ) in the package's (row, column) frequencies.
  const t = (mask.theta * Math.PI) / 180;
  const angle = (Math.atan2(Math.cos(t), -Math.sin(t)) * 180) / Math.PI;
  return { r1: 1 / (2 * mask.n), r2: 1 / (2 * mask.n2), angle };
}

/** e.g. "hann, n = 6" or "elliptical, n1 = 12, n2 = 9.106, θ = 81.4°". */
export function maskLabel(mask: MaskSettings): string {
  if (mask.type === "elliptical") {
    return `elliptical, n1 = ${formatValue(mask.n)}, n2 = ${formatValue(mask.n2)}, θ = ${formatValue(mask.theta)}°`;
  }
  return mask.type === "none" ? "no mask" : `${mask.type}, n = ${formatValue(mask.n)}`;
}
