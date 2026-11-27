// What is drawn of a structure: repeated cells and bonds.

export const MAX_REPEAT = 20;
export const MAX_ATOMS = 50000;
const MAX_BONDS = 200000;
const MIN_BOND = 0.4;
const ON_FAR_FACE = 1 - 1e-4;
const MIN_COLOUR_DISTANCE = 0.55;
const HUE_STEP = 0.17;
const HUE_TURNS = 5;

export type Vec3 = [number, number, number];
export type Rgb = [number, number, number];

export interface ElementInfo {
  /** r, g, b in 0-1. */
  color: Rgb;
  radius: number;
  covalent: number;
}

/** The structure as the backend's atoms endpoint returns it. */
export interface StructureData {
  /** a, b, c as Cartesian vectors (Å). */
  cell: Vec3[];
  atoms: { symbol: string[]; frac: Vec3[]; xyz: Vec3[]; site: number[] };
  /** Occupancy per element, for each site. */
  sites: Record<string, number>[];
  elements: Record<string, ElementInfo>;
}

/** The atoms of some repeated cells. */
export interface DrawnAtoms {
  symbol: string[];
  site: number[];
  frac: Vec3[];
  xyz: Vec3[];
}

const NONMETALS = new Set([
  "H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "As", "Se", "Br", "Kr", "Te", "I", "Xe", "At", "Rn",
]);

/** The cell shifts of a drawn atom along one axis: an atom on the cell's far
 * face is only drawn on the far face of the last cell, as the near face of
 * the next cell already holds it. */
const shiftsOf = (f: number, n: number): number[] =>
  f > ON_FAR_FACE ? [n - 1] : Array.from({ length: n }, (_, i) => i);

/** How many atoms ``repeats`` cells draw. */
export function countAtoms(data: StructureData, repeats: number[]): number {
  return data.atoms.frac.reduce(
    (total, f) => total + f.reduce((k, x, axis) => k * shiftsOf(x, repeats[axis]).length, 1),
    0,
  );
}

/** The atoms of ``repeats`` (na, nb, nc) cells: symbol, site, frac (in unit
 * cells) and xyz (Å) per atom. */
export function expandAtoms(data: StructureData, repeats: number[]): DrawnAtoms {
  const [a, b, c] = data.cell;
  const out: DrawnAtoms = { symbol: [], site: [], frac: [], xyz: [] };
  data.atoms.frac.forEach((f, atom) => {
    for (const i of shiftsOf(f[0], repeats[0])) {
      for (const j of shiftsOf(f[1], repeats[1])) {
        for (const k of shiftsOf(f[2], repeats[2])) {
          const p: Vec3 = [f[0] + i, f[1] + j, f[2] + k];
          out.symbol.push(data.atoms.symbol[atom]);
          out.site.push(data.atoms.site[atom]);
          out.frac.push(p);
          out.xyz.push([0, 1, 2].map((m) => p[0] * a[m] + p[1] * b[m] + p[2] * c[m]) as Vec3);
        }
      }
    }
  });
  return out;
}

const rgbToHsl = ([r, g, b]: Rgb): Rgb => {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  const d = max - min;
  if (!d) return [0, 0, l];
  const s = d / (1 - Math.abs(2 * l - 1));
  const h = max === r ? ((g - b) / d + 6) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [h / 6, s, l];
};

const hslToRgb = ([h, s, l]: Rgb): Rgb => {
  const k = (n: number) => (n + h * 12) % 12;
  const a = s * Math.min(l, 1 - l);
  return [0, 8, 4].map((n) => l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1))) as Rgb;
};

/** ``elements`` with colours told apart on a shared site: an element too
 * close in colour to another on its site has its hue turned. */
export function distinctColors(data: StructureData): Record<string, ElementInfo> {
  const elements: Record<string, ElementInfo> = Object.fromEntries(
    Object.entries(data.elements).map(([el, e]) => [el, { ...e }]),
  );
  for (const site of data.sites) {
    const names = Object.keys(site);
    names.forEach((el, k) => {
      for (let turn = 0; turn < HUE_TURNS; turn++) {
        const colour = elements[el].color;
        const clash = names.slice(0, k).some(
          (other) => Math.hypot(...colour.map((x, i) => x - elements[other].color[i])) < MIN_COLOUR_DISTANCE,
        );
        if (!clash) break;
        const [h, s, l] = rgbToHsl(colour);
        elements[el].color = hslToRgb([(h + HUE_STEP) % 1, Math.max(s, 0.6), Math.min(Math.max(l, 0.4), 0.6)]);
      }
    });
  }
  return elements;
}

/** A site's covalent radius, weighted by occupancy. */
const siteCovalent = (data: StructureData, site: number): number => {
  const entries = Object.entries(data.sites[site]);
  const total = entries.reduce((s, [, x]) => s + x, 0);
  return entries.reduce((s, [el, x]) => s + data.elements[el].covalent * x, 0) / total;
};

/** Bonded pairs [i, j] of ``atoms``: closer than ``tolerance`` × the sum of
 * their covalent radii. With any non-metal present, two metals aren't bonded. */
export function findBonds(atoms: DrawnAtoms, data: StructureData, tolerance: number): [number, number][] {
  const n = atoms.xyz.length;
  const siteRadius = data.sites.map((_, s) => siteCovalent(data, s));
  const radii = atoms.site.map((s) => siteRadius[s]);
  const nonmetal = atoms.symbol.map((s) => NONMETALS.has(s));
  const anyNonmetal = nonmetal.some(Boolean);
  const cutoff = 2 * Math.max(...siteRadius) * tolerance;
  const OFFSET = 1024;
  const SIDE = 2048;
  const keyOf = (x: number, y: number, z: number) => x + OFFSET + (y + OFFSET) * SIDE + (z + OFFSET) * SIDE * SIDE;
  const binOf = (v: number) => Math.floor(v / cutoff);

  const grid = new Map<number, number[]>();
  atoms.xyz.forEach((p, i) => {
    const key = keyOf(binOf(p[0]), binOf(p[1]), binOf(p[2]));
    const list = grid.get(key);
    if (list) list.push(i);
    else grid.set(key, [i]);
  });

  const bonds: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const p = atoms.xyz[i];
    const [bx, by, bz] = p.map(binOf);
    for (let x = bx - 1; x <= bx + 1; x++) {
      for (let y = by - 1; y <= by + 1; y++) {
        for (let z = bz - 1; z <= bz + 1; z++) {
          for (const j of grid.get(keyOf(x, y, z)) ?? []) {
            if (j <= i || (anyNonmetal && !nonmetal[i] && !nonmetal[j])) continue;
            const q = atoms.xyz[j];
            const d = Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
            if (d > MIN_BOND && d < (radii[i] + radii[j]) * tolerance) {
              bonds.push([i, j]);
              if (bonds.length >= MAX_BONDS) return bonds;
            }
          }
        }
      }
    }
  }
  return bonds;
}
