import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { formatValue } from "../format";
import { addSite, removeSite, useCellState } from "../state/cell";
import { SUBLATTICE_COLOURS, SUBLATTICE_NAMES } from "../state/columns";
import { useUi } from "../state/ui";
import { ProcessLink } from "./ProcessLink";

const MARGIN = { left: 44, right: 12, top: 10, bottom: 30 };
/** How far beyond the cell (fractions) the plot repeats it. */
const EDGE = 0.25;

/** The Cell tab under an image: every column placed in the cell of the
 * reference sublattice's lattice (x right, y down, px, equal scales; the
 * cell outlined, the reference at its corners), its edges repeated so sites
 * across them show whole. A click adds a site of the chosen sublattice; a
 * click on a site removes it. */
export function CellPlot({ image }: { image: string }) {
  const st = useCellState(image);
  if (!st.data)
    return (
      <p className="muted small pane-empty">
        <ProcessLink to="sublattices" image={image}>Show the cell</ProcessLink> (Sublattices, by position in cell).
      </p>
    );
  return <Plot image={image} />;
}

function Plot({ image }: { image: string }) {
  const st = useCellState(image);
  const { u, v, vectors } = st.data!;
  // The sites' clusters, when found for the sites as they are: each column coloured by its site.
  const found = st.mode === "clusters" && st.fits && st.fits.sites.length === st.sites.length && st.fits.labels.length === u.length ? st.fits : null;
  const siteColours = st.sites.map((q) => SUBLATTICE_COLOURS[SUBLATTICE_NAMES.indexOf(q.name)] ?? "#fff");
  const [[ay, ax], [by, bx]] = vectors;
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const theme = useUi((s) => s.theme);
  useLayoutEffect(() => {
    const el = wrapRef.current!;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Fractions → px (y down) and back.
  const toPx = (s: number, t: number): [number, number] => [s * ay + t * by, s * ax + t * bx];
  const det = ay * bx - by * ax;
  const toCell = (y: number, x: number): [number, number] => [(y * bx - by * x) / det, (ay * x - y * ax) / det];

  // Points, with copies across the edges (up to EDGE beyond).
  const pts = useMemo(() => {
    const ys: number[] = [];
    const xs: number[] = [];
    const site: number[] = [];
    for (let i = 0; i < u.length; i++) {
      if (!Number.isFinite(u[i])) continue;
      for (const du of [-1, 0, 1])
        for (const dv of [-1, 0, 1]) {
          const s = u[i] + du;
          const t = v[i] + dv;
          if (Math.abs(s) > 0.5 + EDGE || Math.abs(t) > 0.5 + EDGE) continue;
          const [y, x] = toPx(s, t);
          ys.push(y);
          xs.push(x);
          site.push(found ? (found.labels[i] ?? -1) : -2);
        }
    }
    return { ys, xs, site };
  }, [u, v, ay, ax, by, bx, found]); // eslint-disable-line react-hooks/exhaustive-deps

  // The view: the repeated cell's corners, equal scales.
  const corners = [-1, 1].flatMap((s) => [-1, 1].map((t) => toPx(s * (0.5 + EDGE), t * (0.5 + EDGE))));
  const ext = {
    y0: Math.min(...corners.map((c) => c[0])), y1: Math.max(...corners.map((c) => c[0])),
    x0: Math.min(...corners.map((c) => c[1])), x1: Math.max(...corners.map((c) => c[1])),
  };
  const aw = Math.max(1, size.w - MARGIN.left - MARGIN.right);
  const ah = Math.max(1, size.h - MARGIN.top - MARGIN.bottom);
  const k = Math.min(aw / (ext.x1 - ext.x0), ah / (ext.y1 - ext.y0));
  const ox = MARGIN.left + (aw - k * (ext.x1 - ext.x0)) / 2;
  const oy = MARGIN.top + (ah - k * (ext.y1 - ext.y0)) / 2;
  const sx = (x: number) => ox + (x - ext.x0) * k;
  const sy = (y: number) => oy + (y - ext.y0) * k;

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !size.w) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = size.w * dpr;
    c.height = size.h * dpr;
    const g = c.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, size.w, size.h);
    const css = getComputedStyle(document.documentElement);
    const accent = css.getPropertyValue("--accent").trim() || "#4da3ff";
    const none = css.getPropertyValue("--muted").trim() || "#888";
    // Coloured by site once clusters are found (left out: grey, drawn bolder).
    g.globalAlpha = found ? 0.6 : pts.xs.length > 20000 ? 0.15 : 0.35;
    for (let i = 0; i < pts.xs.length; i++) {
      const k2 = pts.site[i];
      g.fillStyle = k2 === -2 ? accent : k2 === -1 ? none : siteColours[k2];
      const r = k2 === -1 ? 1.25 : 0.75;
      g.fillRect(sx(pts.xs[i]) - r, sy(pts.ys[i]) - r, 2 * r, 2 * r);
    }
  }, [pts, size.w, size.h, k, theme, siteColours.join()]); // eslint-disable-line react-hooks/exhaustive-deps

  const outline = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]].map(([s, t]) => toPx(s, t)).map(([y, x]) => `${sx(x)},${sy(y)}`).join(" ");
  const click = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - r.left - ox) / k + ext.x0;
    const y = (e.clientY - r.top - oy) / k + ext.y0;
    const [s, t] = toCell(y, x);
    // On a site (or a copy of it): remove it; else add one.
    const hit = st.sites.findIndex((q) => {
      const ds = s - q.u - Math.round(s - q.u);
      const dt = t - q.v - Math.round(t - q.v);
      const [py, px] = toPx(ds, dt);
      return Math.hypot(py, px) * k < 8;
    });
    if (hit >= 0) removeSite(image, hit);
    else addSite(image, s, t);
  };
  // Each site drawn where it falls in the repeated view.
  const marks = st.sites.flatMap((q, i) =>
    [-1, 0, 1].flatMap((du) =>
      [-1, 0, 1]
        .filter((dv) => Math.abs(q.u + du) <= 0.5 + EDGE && Math.abs(q.v + dv) <= 0.5 + EDGE)
        .map((dv) => ({ i, q, at: toPx(q.u + du, q.v + dv) })),
    ),
  );
  const refs = [-1, 0, 1].flatMap((s) => [-1, 0, 1].map((t) => toPx(s, t))).filter(([y, x]) => y >= ext.y0 && y <= ext.y1 && x >= ext.x0 && x <= ext.x1);

  return (
    <div className="nb-plot-wrap">
      <div className="nb-plot-title small">
        <span className="muted">
          {u.filter((x) => Number.isFinite(x)).length} columns in the cell · click: add a site of {st.name} · click a site: remove it
        </span>
      </div>
      <div ref={wrapRef} className="nb-plot">
        <canvas ref={canvasRef} style={{ width: size.w, height: size.h }} />
        {size.w > 0 && (
          <svg width={size.w} height={size.h} onClick={click} style={{ cursor: "crosshair" }}>
            <polygon className="cell-outline" points={outline} />
            {refs.map(([y, x], i) => (
              <g key={i} className="nb-cage-centre">
                <line x1={sx(x) - 5} x2={sx(x) + 5} y1={sy(y)} y2={sy(y)} />
                <line x1={sx(x)} x2={sx(x)} y1={sy(y) - 5} y2={sy(y) + 5} />
              </g>
            ))}
            {marks.map(({ i, q, at: [y, x] }, j) => {
              const colour = SUBLATTICE_COLOURS[SUBLATTICE_NAMES.indexOf(q.name)] ?? "#fff";
              return (
                <g key={`${i}-${j}`} className="cell-site">
                  <circle cx={sx(x)} cy={sy(y)} r={Math.max(3, st.radius * k)} stroke={colour} className={st.mode === "clusters" ? "reach" : undefined} />
                  <text x={sx(x) + 6} y={sy(y) - 6} fill={colour}>{q.name}</text>
                </g>
              );
            })}
            {/* Groups no site took: where they are, and how many. */}
            {found &&
              found.unpicked.flatMap((g, gi) =>
                [-1, 0, 1].flatMap((du) =>
                  [-1, 0, 1]
                    .filter((dv) => Math.abs(g.u + du) <= 0.5 + EDGE && Math.abs(g.v + dv) <= 0.5 + EDGE)
                    .map((dv) => {
                      const [y, x] = toPx(g.u + du, g.v + dv);
                      return (
                        <g key={`un-${gi}-${du}-${dv}`} className="cell-unpicked">
                          <circle cx={sx(x)} cy={sy(y)} r={Math.max(6, 4 * k)} />
                          <text x={sx(x) + 8} y={sy(y) + 4}>{g.count}?</text>
                        </g>
                      );
                    }),
                ),
              )}
            <text className="nb-tick" x={ox} y={size.h - 8}>
              cell {formatValue(Math.hypot(ay, ax))} × {formatValue(Math.hypot(by, bx))} px
            </text>
          </svg>
        )}
      </div>
    </div>
  );
}
