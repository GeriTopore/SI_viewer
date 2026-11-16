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
