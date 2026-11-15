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
  // Circular (hue round the wheel, back to red): directions, e.g. of a DPC field.
  hsv: ["#ff0000", "#ffff00", "#00ff00", "#00ffff", "#0000ff", "#ff00ff", "#ff0000"],
  // cmocean's "phase" (circular, as kemstem's plot_phase): phases over 0–2π.
  phase: [
    "#a8780d", "#b3701b", "#be6828", "#c75f35", "#cf5643", "#d54b53", "#db4066", "#de357b", "#df2a93",
    "#dc25ad", "#d529c4", "#cc34d7", "#c041e5", "#b24fef", "#a25cf3", "#9168f4", "#7d73f0", "#687ce8",
    "#5285dc", "#3d8bcd", "#2c90bc", "#2093ac", "#19959c", "#12978c", "#0c987c", "#119a69", "#249a52",
    "#409839", "#5e9420", "#778d12", "#8b860d", "#9b7f0d", "#a8780d",
  ],
  // Diverging (matplotlib RdBu): negative red, zero white, positive blue.
  rdbu: [
    "#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#f7f7f7",
    "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061",
  ],
};

export const COLORMAPS = Object.keys(STOPS);

// Default colours for element maps, in the order they're handed out.
export const OVERLAY_COLOURS = [
  "#ff3b30", "#34c759", "#2f7bff", "#00d1e6", "#ff2dd4", "#ffd60a",
  "#ff9500", "#af52de", "#a8ff3e", "#ff8fa3", "#7fd4ff", "#ffffff",
];

/** A colormap name may also be a "#rrggbb" colour: black → that colour. */
export const isColour = (name: string) => /^#[0-9a-f]{6}$/i.test(name);

const stopsFor = (name: string) => (isColour(name) ? ["#000000", name] : (STOPS[name] ?? STOPS.gray));

const hexToRgb = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

const lutCache = new Map<string, Uint8Array>();

/** 256 × RGBA8 lookup table. */
export function colormapLut(name: string): Uint8Array {
  const cached = lutCache.get(name);
  if (cached) return cached;
  const stops = stopsFor(name).map(hexToRgb);
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

/** CSS background for a small swatch: the colour itself, or the gradient. */
export const swatchBackground = (name: string) => (isColour(name) ? name : colormapGradient(name));

/** CSS gradient for previews in the colormap picker. */
export const colormapGradient = (name: string) =>
  `linear-gradient(to right, ${stopsFor(name).join(", ")})`;
