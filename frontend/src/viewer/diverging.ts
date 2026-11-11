/** Colour maps for colouring a sequence of lines (first to last), mostly
 * diverging, low end first. Stops are evenly spaced. */
export interface LineMap {
  id: string;
  label: string;
  stops: [number, number, number][];
}

const hex = (h: string): [number, number, number] => [
  parseInt(h.slice(1, 3), 16),
  parseInt(h.slice(3, 5), 16),
  parseInt(h.slice(5, 7), 16),
];

export const LINE_MAPS: LineMap[] = [
  { id: "blue-red", label: "Blue–red", stops: [[49, 84, 199], [214, 39, 40]] },
  { id: "coolwarm", label: "Coolwarm", stops: ["#3b4cc0", "#7b9ff9", "#c0d4f5", "#f2cbb7", "#ee8468", "#b40426"].map(hex) },
  { id: "rdbu", label: "Blue–white–red", stops: ["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7", "#fddbc7", "#ef8a62", "#b2182b"].map(hex) },
  { id: "spectral", label: "Spectral", stops: ["#3288bd", "#99d594", "#e6f598", "#ffffbf", "#fee08b", "#fc8d59", "#d53e4f"].map(hex) },
  { id: "puor", label: "Purple–orange", stops: ["#542788", "#998ec3", "#d8daeb", "#f7f7f7", "#fee0b6", "#f1a340", "#b35806"].map(hex) },
  { id: "piyg", label: "Green–pink", stops: ["#4d9221", "#a1d76a", "#e6f5d0", "#f7f7f7", "#fde0ef", "#e9a3c9", "#c51b7d"].map(hex) },
  { id: "brbg", label: "Teal–brown", stops: ["#01665e", "#5ab4ac", "#c7eae5", "#f5f5f5", "#f6e8c3", "#d8b365", "#8c510a"].map(hex) },
];

export const lineMap = (id: string): LineMap => LINE_MAPS.find((m) => m.id === id) ?? LINE_MAPS[0];

/** The colour of ``map`` at ``t`` (0: first, 1: last), interpolated between its stops. */
export function lineColor(map: LineMap, t: number): string {
  const s = map.stops;
  const f = Math.min(1, Math.max(0, t)) * (s.length - 1);
  const i = Math.min(s.length - 2, Math.floor(f));
  const k = f - i;
  const c = s[i].map((v, j) => Math.round(v + (s[i + 1][j] - v) * k));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

/** A CSS gradient of ``map``, left to right. */
export const lineGradient = (map: LineMap) =>
  `linear-gradient(to right, ${map.stops.map((c) => `rgb(${c[0]}, ${c[1]}, ${c[2]})`).join(", ")})`;

/** Its stops as a Plotly colour scale. */
export const plotlyScale = (map: LineMap): [number, string][] =>
  map.stops.map((c, i) => [i / (map.stops.length - 1), `rgb(${c[0]}, ${c[1]}, ${c[2]})`]);
