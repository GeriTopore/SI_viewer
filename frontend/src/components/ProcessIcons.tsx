import type { ReactNode } from "react";

/** Pictures of the Process tab's processes, drawn in the text colour with
 * the accent for what the process makes. */
const Icon = ({ size, children }: { size: number; children: ReactNode }) => (
  <svg
    className="process-icon"
    width={size}
    height={size}
    viewBox="0 0 48 48"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden
  >
    {children}
  </svg>
);

const ACCENT = "var(--accent)";

/** An arrow from (x1, y) to (x2, y), its head at x2. */
const HArrow = ({ x1, x2, y }: { x1: number; x2: number; y: number }) => {
  const d = x2 > x1 ? -4 : 4;
  return <path d={`M${x1} ${y}H${x2}M${x2 + d} ${y - 4}L${x2} ${y}L${x2 + d} ${y + 4}`} />;
};

/** Square cells of a grid, ``n`` × ``n`` of side ``cell`` from (x, y). */
const Cells = ({ x, y, n, cell, stroke }: { x: number; y: number; n: number; cell: number; stroke?: string }) => (
  <g stroke={stroke} strokeWidth="1.5">
    {Array.from({ length: n * n }, (_, i) => (
      <rect key={i} x={x + (i % n) * cell} y={y + Math.floor(i / n) * cell} width={cell} height={cell} />
    ))}
  </g>
);

const ICONS: Record<string, () => ReactNode> = {
  // Two drifted frames brought onto one.
  align: () => (
    <>
      <rect x="6" y="6" width="22" height="22" rx="2" strokeDasharray="3 3" />
      <rect x="20" y="20" width="22" height="22" rx="2" stroke={ACCENT} />
      <path d="M13 13L27 27M27 20V27H20" />
    </>
  ),
  // Drifted frames brought onto one and added up.
  "aligned-sum": () => (
    <>
      <rect x="4" y="4" width="18" height="18" rx="2" strokeDasharray="3 3" />
      <rect x="9" y="9" width="18" height="18" rx="2" strokeDasharray="3 3" />
      <rect x="22" y="22" width="20" height="20" rx="2" stroke={ACCENT} />
      <path d="M37 27H28L33 32L28 37H37" stroke={ACCENT} />
    </>
  ),
  // A fine grid summed into a coarser one.
  rebin: () => (
    <>
      <Cells x={3} y={14} n={4} cell={5} />
      <HArrow x1={26} x2={32} y={24} />
      <Cells x={35} y={17} n={2} cell={6} stroke={ACCENT} />
    </>
  ),
  // A zero-loss peak, with the spectra shifted either way onto 0 eV.
  zlp: () => (
    <>
      <path d="M4 40H44" />
      <path d="M24 9V40" strokeDasharray="2 3" strokeWidth="1.5" />
      <path d="M8 40C16 40 19 9 24 9S32 40 40 40" stroke={ACCENT} />
      <HArrow x1={17} x2={4} y={20} />
      <HArrow x1={31} x2={44} y={20} />
    </>
  ),
  // A noisy line smoothed.
  filters: () => (
    <>
      <path d="M4 18L8 11L11 20L15 9L18 17L22 8L25 16L29 7L32 15L36 9L39 14L44 8" strokeWidth="1.5" />
      <path d="M24 22V28M21 25L24 28L27 25" />
      <path d="M4 42C12 42 14 33 24 33S36 40 44 36" stroke={ACCENT} />
    </>
  ),
  // An edge on a decaying background, the background fitted under it.
  background: () => (
    <>
      <path d="M4 42H44" />
      <path d="M6 8C10 24 14 30 20 32L22 18C28 22 36 28 44 30" />
      <path d="M20 32C28 35 36 37 44 38" stroke={ACCENT} strokeDasharray="3 3" />
      <path d="M22 18C28 22 36 28 44 30V38C36 37 28 35 20 32Z" fill={ACCENT} fillOpacity="0.18" stroke="none" />
    </>
  ),
  // A slab and its thickness.
  thickness: () => (
    <>
      <path d="M4 16L14 10H44L34 16Z" />
      <path d="M4 16V32L34 32V16M34 32L44 26V10" />
      <path d="M19 16V32" stroke={ACCENT} />
      <path d="M15 20L19 16L23 20M15 28L19 32L23 28" stroke={ACCENT} />
    </>
  ),
  // A four-segment detector with the direct beam shifted off its centre.
  dpc: () => (
    <>
      <circle cx="24" cy="24" r="19" />
      <circle cx="24" cy="24" r="9" />
      <path d="M24 5V15M24 33V43M5 24H15M33 24H43" />
      <circle cx="29" cy="20" r="7.5" fill={ACCENT} fillOpacity="0.35" stroke={ACCENT} />
    </>
  ),
  // Two peaks (dashed) and the model through them.
  curvefit: () => (
    <>
      <path d="M4 42H44" />
      <path d="M6 41C13 41 14 16 18 16S23 41 30 41" strokeWidth="1.5" strokeDasharray="3 2.5" />
      <path d="M18 41C25 41 26 24 30 24S35 41 42 41" strokeWidth="1.5" strokeDasharray="3 2.5" />
      <path d="M5 41C12 41 14 15 18 15S22 30 25 30S28 23 30 23S35 41 43 41" stroke={ACCENT} />
    </>
  ),
  // Two spectra of different heights, scaled to meet between 0 and 1.
  normalise: () => (
    <>
      <path d="M6 6V42H44" />
      <path d="M10 8H14M10 38H14" stroke={ACCENT} />
      <path d="M12 38C17 38 19 14 23 14S29 30 33 30S38 26 43 26" strokeWidth="1.5" />
      <path d="M12 38C17 38 19 8 23 8S29 28 33 28S38 22 43 22" stroke={ACCENT} />
    </>
  ),
  // A scree plot: the components' share falling, the first few kept.
  pca: () => (
    <>
      <path d="M6 6V42H44" />
      <path d="M12 42V10M19 42V22M26 42V31" stroke={ACCENT} strokeWidth="4" strokeLinecap="butt" />
      <path d="M33 42V36M40 42V38" strokeWidth="4" strokeLinecap="butt" />
    </>
  ),
  // An FFT: its centre and a pair of spots, one ringed by the filter.
  fourier: () => (
    <>
      <rect x="5" y="5" width="38" height="38" rx="2" />
      <circle cx="24" cy="24" r="2.5" fill="currentColor" />
      <circle cx="14" cy="17" r="2" fill="currentColor" />
      <circle cx="34" cy="31" r="2" fill="currentColor" />
      <circle cx="34" cy="31" r="7" stroke={ACCENT} fill={ACCENT} fillOpacity="0.25" />
    </>
  ),
  // A lattice of columns, one fitted (a Gaussian's contours).
  columns: () => (
    <>
      {[10, 24, 38].flatMap((y) => [10, 24, 38].map((x) => <circle key={`${x}-${y}`} cx={x} cy={y} r="3" fill="currentColor" />))}
      <circle cx="24" cy="24" r="7" stroke={ACCENT} />
      <circle cx="24" cy="24" r="11" stroke={ACCENT} strokeWidth="1.2" strokeDasharray="2.5 2" />
    </>
  ),
  // Columns displaced in a wave: triangles pointing their way.
  pld: () => (
    <>
      {[10, 24, 38].map((x, i) => (
        <g key={x}>
          <path d={i % 2 ? `M${x} 14L${x - 3} 8L${x + 3} 8Z` : `M${x} 8L${x - 3} 14L${x + 3} 14Z`} fill={i % 2 ? ACCENT : "currentColor"} stroke="none" />
          <path d={i % 2 ? `M${x} 40L${x - 3} 34L${x + 3} 34Z` : `M${x} 34L${x - 3} 40L${x + 3} 40Z`} fill={i % 2 ? ACCENT : "currentColor"} stroke="none" />
          <circle cx={x} cy={24} r="3" fill="currentColor" />
        </g>
      ))}
      <path d="M4 24C10 18 14 18 17 24S24 30 31 24S38 18 44 24" stroke={ACCENT} strokeWidth="1.2" strokeDasharray="2 2" />
    </>
  ),
  // A column's neighbours: vectors from the centre, one picked.
  neighbourhoods: () => (
    <>
      {[[10, 10], [24, 8], [38, 10], [8, 24], [40, 24], [10, 38], [24, 40], [38, 38]].map(([x, y]) => (
        <circle key={`${x}-${y}`} cx={x} cy={y} r="3" fill="currentColor" />
      ))}
      <circle cx="24" cy="24" r="3.5" fill="currentColor" />
      <path d="M24 24L37 24" stroke={ACCENT} strokeWidth="2" />
      <path d="M33 20.5L37.5 24L33 27.5" stroke={ACCENT} strokeWidth="2" />
      <path d="M24 24L24 11" stroke={ACCENT} strokeWidth="1.2" strokeDasharray="2 2" />
    </>
  ),
  // An image's bright region cut out as a mask.
  mask: () => (
    <>
      <rect x="6" y="6" width="36" height="36" rx="2" />
      <path d="M12 30C10 20 18 12 26 14S40 22 34 32S16 40 12 30Z" stroke={ACCENT} fill={ACCENT} fillOpacity="0.35" />
    </>
  ),
  // A lattice of two kinds of columns (two sublattices).
  sublattices: () => (
    <>
      {[10, 24, 38].flatMap((y, j) =>
        [10, 24, 38].map((x, i) =>
          (i + j) % 2 ? (
            <circle key={`${x}-${y}`} cx={x} cy={y} r="3.5" stroke={ACCENT} fill={ACCENT} fillOpacity="0.35" />
          ) : (
            <circle key={`${x}-${y}`} cx={x} cy={y} r="4.5" fill="currentColor" />
          ),
        ),
      )}
    </>
  ),
  // ---------- 4D-STEM ----------
  // A diffraction pattern and an annular detector over it.
  "virtual-image": () => (
    <>
      <circle cx="24" cy="24" r="19" />
      <path d="M11 24a13 13 0 1 0 26 0a13 13 0 1 0 -26 0ZM16 24a8 8 0 1 0 16 0a8 8 0 1 0 -16 0Z"
        fillRule="evenodd" fill={ACCENT} fillOpacity="0.3" stroke={ACCENT} strokeWidth="1.5" />
      <circle cx="24" cy="24" r="3" fill="currentColor" />
      {[[24, 8], [24, 40], [8, 24], [40, 24]].map(([x, y]) => <circle key={`${x}-${y}`} cx={x} cy={y} r="1.8" fill="currentColor" />)}
    </>
  ),
  // A detector's pixels, a hot one crossed out.
  "bad-pixels": () => (
    <>
      <Cells x={6} y={6} n={4} cell={9} />
      <rect x="24" y="15" width="9" height="9" fill="currentColor" />
      <path d="M21 12L36 27M36 12L21 27" stroke={ACCENT} strokeWidth="2.5" />
    </>
  ),
  // The direct beam off the centre (descan) brought back onto it.
  centring: () => (
    <>
      <path d="M24 4V44M4 24H44" strokeWidth="1.2" strokeDasharray="2 3" />
      <circle cx="35" cy="13" r="6" strokeDasharray="3 2" />
      <path d="M31 17L28 20M28 16V20H32" />
      <circle cx="24" cy="24" r="6" stroke={ACCENT} fill={ACCENT} fillOpacity="0.3" />
    </>
  ),
  // A pattern's rings unrolled into lines (angle across, radius down).
  polar: () => (
    <>
      <circle cx="13" cy="24" r="10" />
      <circle cx="13" cy="24" r="5" strokeWidth="1.5" />
      <circle cx="13" cy="24" r="1.5" fill="currentColor" />
      <HArrow x1={25} x2={31} y={24} />
      <rect x="33" y="12" width="11" height="24" rx="1" />
      <path d="M33 18H44M33 29H44" stroke={ACCENT} />
    </>
  ),
  // Diffraction disks, found ones ringed.
  disks: () => (
    <>
      {[10, 24, 38].flatMap((y) => [10, 24, 38].map((x) => (
        <circle key={`${x}-${y}`} cx={x} cy={y} r={x === 24 && y === 24 ? 3.5 : 2.5} fill="currentColor" />
      )))}
      {[[24, 10], [10, 24], [38, 24], [24, 38]].map(([x, y]) => (
        <circle key={`r${x}-${y}`} cx={x} cy={y} r="6" stroke={ACCENT} />
      ))}
    </>
  ),
  // A crystal (a cell) and the pattern simulated from it.
  templates: () => (
    <>
      <path d="M5 11H17V23H5ZM11 5H23V17H11ZM5 11L11 5M17 11L23 5M17 23L23 17M5 23L11 17" strokeWidth="1.5" />
      <circle cx="32" cy="32" r="2.2" fill="currentColor" />
      {[0, 60, 120, 180, 240, 300].map((a) => {
        const r = (a * Math.PI) / 180;
        return <circle key={a} cx={32 + 9 * Math.cos(r)} cy={32 + 9 * Math.sin(r)} r="2.4" fill={ACCENT} stroke="none" />;
      })}
    </>
  ),
  // Grains, each with its crystal turned its own way: one picked out.
  orientation: () => (
    <>
      <path d="M5 22L20 18L28 30L22 43H5Z" fill={ACCENT} fillOpacity="0.3" stroke="none" />
      <rect x="5" y="5" width="38" height="38" rx="2" />
      <path d="M5 22L20 18L24 5M20 18L28 30L43 26M28 30L22 43" strokeWidth="1.5" />
      {([[12, 12, 20], [33, 14, -15], [14, 32, 40], [35, 35, 5]] as const).map(([x, y, a], i) => (
        <rect key={i} x="-3.5" y="-3.5" width="7" height="7" transform={`translate(${x} ${y}) rotate(${a})`}
          strokeWidth="1.5" stroke={i === 2 ? ACCENT : "currentColor"} />
      ))}
    </>
  ),
  // Spots, their spacing measured against a ruler.
  calibration: () => (
    <>
      {[10, 24, 38].map((x) => <circle key={x} cx={x} cy="14" r="3" fill="currentColor" />)}
      <path d="M10 24H38M10 20V28M38 20V28" stroke={ACCENT} />
      <rect x="4" y="34" width="40" height="8" rx="1" />
      <path d="M10 34V38M17 34V37M24 34V38M31 34V37M38 34V38" strokeWidth="1.5" />
    </>
  ),
  // The vectors' lengths as a histogram: its peaks the reflections' rings.
  "vector-magnitudes": () => (
    <>
      <path d="M6 6V42H44" />
      {[[11, 4], [16, 22], [21, 8], [26, 4], [31, 28], [36, 9], [41, 3]].map(([x, h]) => (
        <path key={x} d={`M${x} 42V${42 - h}`} strokeWidth="3.5" strokeLinecap="butt" stroke={h > 20 ? ACCENT : "currentColor"} />
      ))}
    </>
  ),
  // A pattern with its Bragg spots masked.
  "vector-mask": () => (
    <>
      <circle cx="24" cy="24" r="19" />
      <circle cx="24" cy="24" r="3" fill="currentColor" />
      {[[24, 12], [24, 36], [12, 24], [36, 24]].map(([x, y]) => (
        <g key={`${x}-${y}`}>
          <circle cx={x} cy={y} r="5" stroke={ACCENT} fill={ACCENT} fillOpacity="0.35" />
          <circle cx={x} cy={y} r="1.6" fill="currentColor" />
        </g>
      ))}
    </>
  ),
  // A cell of the lattice (dashed) stretched and sheared.
  strain: () => (
    <>
      <rect x="7" y="12" width="26" height="26" strokeDasharray="3 3" />
      <path d="M7 38L13 10H43L37 38Z" stroke={ACCENT} />
      <path d="M33 25H41M38 22L41 25L38 28" strokeWidth="1.5" />
    </>
  ),
  // A peak with plural-scattering bumps made single.
  deconvolution: () => (
    <>
      <path d="M4 42H44" />
      <path d="M3 41C7 41 8 22 10 22S12 34 14 34S16 28 18 28S20 38 22 41" strokeWidth="1.5" />
      <HArrow x1={23} x2={29} y={24} />
      <path d="M30 41C34 41 35 12 37 12S40 41 44 41" stroke={ACCENT} />
    </>
  ),
};

/** The picture of process ``id`` (none for one without). */
export function ProcessIcon({ id, size = 40 }: { id: string; size?: number }) {
  const draw = ICONS[id];
  return draw ? <Icon size={size}>{draw()}</Icon> : null;
}
