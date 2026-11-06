// Colormaps as evenly spaced control points, expanded to 256-entry LUTs.
// The perceptual maps are sampled from matplotlib's definitions.

const STOPS: Record<string, string[]> = {
  gray: ["#000000", "#ffffff"],
  inferno: [
    "#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60",
    "#cf4446", "#ed6925", "#fb9b06", "#f7d13d", "#fcffa4",
  ],
  magma: [
    "#000004", "#180f3d", "#440f76", "#721f81", "#9e2f7f",
    "#cd4071", "#f1605d", "#fd9668", "#feca8d", "#fcfdbf",
  ],
  viridis: [
    "#440154", "#472d7b", "#3b528b", "#2c728e", "#21918c",
    "#28ae80", "#5ec962", "#addc30", "#fde725",
  ],
  cividis: [
    "#00224e", "#123570", "#3b496c", "#575d6d", "#707173",
    "#8a8678", "#a59c74", "#c3b369", "#e1cc55", "#fee838",
  ],
  hot: ["#000000", "#800000", "#ff0000", "#ff8000", "#ffff00", "#ffffff"],
};

export const COLORMAPS = Object.keys(STOPS);

const hexToRgb = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

const lutCache = new Map<string, Uint8Array>();

/** 256 × RGBA8 lookup table. */
export function colormapLut(name: string): Uint8Array {
  const cached = lutCache.get(name);
  if (cached) return cached;
  const stops = (STOPS[name] ?? STOPS.gray).map(hexToRgb);
  const lut = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const pos = (i / 255) * (stops.length - 1);
    const k = Math.min(Math.floor(pos), stops.length - 2);
    const f = pos - k;
    for (let c = 0; c < 3; c++) {
      lut[i * 4 + c] = Math.round(stops[k][c] * (1 - f) + stops[k + 1][c] * f);
    }
    lut[i * 4 + 3] = 255;
  }
  lutCache.set(name, lut);
  return lut;
}

/** CSS gradient for previews in the colormap picker. */
export const colormapGradient = (name: string) =>
  `linear-gradient(to right, ${(STOPS[name] ?? STOPS.gray).join(", ")})`;
