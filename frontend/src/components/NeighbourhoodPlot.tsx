import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { DatasetSummary } from "../api/client";
import { formatValue } from "../format";
import { allDatasets, useDatasets } from "../state/datasets";
import {
  ensureNeighbourhoodData, neighbourhoodsOf, originsOf, pickVector, selectNeighbourhood, shownNeighbourhood, useAnalysis,
  useNeighbourhoods, useNeighbourhoodState,
} from "../state/neighbourhoods";
import { useUi } from "../state/ui";

const MARGIN = { left: 44, right: 12, top: 10, bottom: 30 };

/** A round step for about ``n`` ticks over ``range``. */
function niceStep(range: number, n = 3): number {
  const raw = range / n;
  const pow = 10 ** Math.floor(Math.log10(raw));
  return ([1, 2, 5, 10].find((m) => m * pow >= raw) ?? 10) * pow;
}

/** The Neighbourhood tab under an image: the vectors from each origin
 * column to its neighbours (kemstem's pair correlation), as in the image (x
 * right, y down, px, equal scales), with the k-means clusters and the
 * lattice vectors picked from them. A click on a cluster sets the vector
 * being picked; the wheel zooms. */
export function NeighbourhoodPlot({ image }: { image: string }) {
  const files = useDatasets((s) => s.files);
  const focusId = useDatasets((s) => s.focusId);
  const st = useNeighbourhoodState(image);
  const list = useMemo(() => neighbourhoodsOf(files, image), [files, image]);
  const nb = shownNeighbourhood(files, image, st.selected);
  // A neighbourhood (or its bonds) clicked in the dataset pane is the one shown.
  useEffect(() => {
    const all = allDatasets(files);
    const focused = all.find((d) => d.id === focusId);
    const x = focused?.atomic?.output === "bonds" ? all.find((d) => d.id === focused.parent_id) : focused;
    if (x && list.some((y) => y.id === x.id) && x.id !== st.selected) selectNeighbourhood(image, x);
  }, [focusId]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (nb) ensureNeighbourhoodData(nb);
  }, [nb]);
  const data = useNeighbourhoods((s) => (nb ? s.data[nb.id] : undefined));
  const a = useAnalysis(nb?.id);
  if (!nb) return <p className="muted small pane-empty">Compute a neighbourhood in the Neighbourhoods process.</p>;
  const origins = allDatasets(files).find((x) => x.id === originsOf(nb));
  return (
    <div className="nb-plot-wrap">
      <div className="nb-plot-title small">
        <select
          className="nb-plot-choice"
          value={nb.id}
          title="The neighbourhood shown (and worked on in its process)"
          onChange={(e) => {
            const x = list.find((y) => y.id === e.target.value);
            if (x) selectNeighbourhood(image, x);
          }}
        >
          {list.map((x) => (
            <option key={x.id} value={x.id}>
              {x.title}
            </option>
          ))}
        </select>
        <span className="muted">
          {origins?.atomic?.output === "columns" ? "all" : origins?.title} · {nb.atomic?.count} columns × {nb.atomic?.k}
          {nb.atomic?.clusters ? ` · ${nb.atomic.clusters.length} clusters` : ""}
        </span>
      </div>
      {data ? (
        <Scatter nb={nb} dy={data.dy} dx={data.dx} picking={a.pick !== null ? a.vectors[a.pick]?.name : undefined} vectors={a.vectors} />
      ) : (
        <p className="muted small">Loading…</p>
      )}
    </div>
  );
}

function Scatter({ nb, dy, dx, picking, vectors }: {
  nb: DatasetSummary;
  dy: Float32Array;
  dx: Float32Array;
  picking: string | undefined;
  vectors: { name: string; dy: number | null; dx: number | null }[];
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [hover, setHover] = useState<number | null>(null);
  const theme = useUi((s) => s.theme);
  useLayoutEffect(() => {
    const el = wrapRef.current!;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Start with nearly all of the vectors in view.
  const auto = useMemo(() => {
    const m = Float32Array.from(dy, (v, i) => Math.max(Math.abs(v), Math.abs(dx[i]))).sort();
    return Math.max(1, (m[Math.floor(0.98 * (m.length - 1))] ?? 1) * 1.1);
  }, [dy, dx]);
  const [zoom, setZoom] = useState(1);
  const extent = auto / zoom;

  const side = Math.max(0, Math.min(size.w - MARGIN.left - MARGIN.right, size.h - MARGIN.top - MARGIN.bottom));
  const ox = MARGIN.left + (size.w - MARGIN.left - MARGIN.right - side) / 2;
  const oy = MARGIN.top;
  const sx = (v: number) => ox + ((v + extent) / (2 * extent)) * side;
  const sy = (v: number) => oy + ((v + extent) / (2 * extent)) * side; // y down, as in the image

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !side) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = size.w * dpr;
    c.height = size.h * dpr;
    const g = c.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, size.w, size.h);
    g.save();
    g.beginPath();
    g.rect(ox, oy, side, side);
    g.clip();
    const css = getComputedStyle(document.documentElement);
    g.fillStyle = css.getPropertyValue("--accent").trim() || "#4da3ff";
    // Faint dots: dense places build up, as kemstem's plot (alpha 0.1).
    g.globalAlpha = dy.length > 20000 ? 0.12 : 0.3;
    for (let i = 0; i < dy.length; i++) g.fillRect(sx(dx[i]) - 0.75, sy(dy[i]) - 0.75, 1.5, 1.5);
    g.restore();
  }, [dy, dx, size.w, size.h, extent, theme]); // eslint-disable-line react-hooks/exhaustive-deps

  const clusters = nb.atomic?.clusters ?? [];
  const step = niceStep(extent);
  const ticks: number[] = [];
  for (let t = -Math.floor(extent / step) * step; t <= extent + 1e-9; t += step) ticks.push(Number(t.toPrecision(6)));
  const set = vectors.filter((v) => v.dy !== null && v.dx !== null);

  return (
    <div
      ref={wrapRef}
      className="nb-plot"
      onWheel={(e) => setZoom((z) => Math.min(50, Math.max(0.5, z * (e.deltaY < 0 ? 1.2 : 1 / 1.2))))}
      onDoubleClick={() => setZoom(1)}
      title="Wheel to zoom, double-click to reset"
    >
      <canvas ref={canvasRef} style={{ width: size.w, height: size.h }} />
      {side > 0 && (
        <svg width={size.w} height={size.h}>
          <rect className="nb-frame" x={ox} y={oy} width={side} height={side} />
          <line className="nb-axis" x1={sx(0)} x2={sx(0)} y1={oy} y2={oy + side} />
          <line className="nb-axis" x1={ox} x2={ox + side} y1={sy(0)} y2={sy(0)} />
          {ticks.map((t) => (
            <g key={t} className="nb-tick">
              <text x={sx(t)} y={oy + side + 14} textAnchor="middle">{formatValue(t)}</text>
              <text x={ox - 6} y={sy(t) + 4} textAnchor="end">{formatValue(t)}</text>
            </g>
          ))}
          <text className="nb-tick" x={ox + side / 2} y={oy + side + 27} textAnchor="middle">dx (px)</text>
          <text className="nb-tick" transform={`translate(${ox - 34} ${oy + side / 2}) rotate(-90)`} textAnchor="middle">dy (px)</text>
          {set.map((v) => {
            const inView = Math.abs(v.dx!) <= extent && Math.abs(v.dy!) <= extent;
            return (
              inView && (
                <g key={v.name} className="nb-vector">
                  <line x1={sx(0)} y1={sy(0)} x2={sx(v.dx!)} y2={sy(v.dy!)} />
                  <text x={sx(v.dx!) + 6} y={sy(v.dy!) - 6}>{v.name}</text>
                </g>
              )
            );
          })}
          {clusters.map(([cy, cx], i) =>
            Math.abs(cx) <= extent && Math.abs(cy) <= extent ? (
              <circle
                key={i}
                className={`nb-cluster${hover === i ? " hover" : ""}${picking ? " pickable" : ""}`}
                cx={sx(cx)}
                cy={sy(cy)}
                r={hover === i ? 7 : 5}
                onPointerEnter={() => setHover(i)}
                onPointerLeave={() => setHover((h) => (h === i ? null : h))}
                onClick={() => picking && pickVector(nb.id, [cy, cx])}
              />
            ) : null,
          )}
          {hover !== null && clusters[hover] && (
            <text className="nb-readout" x={ox + 6} y={oy + 14}>
              ({formatValue(clusters[hover][0])}, {formatValue(clusters[hover][1])}) px · |v| {formatValue(Math.hypot(...clusters[hover]))}
              {picking ? ` · click: ${picking}` : ""}
            </text>
          )}
        </svg>
      )}
    </div>
  );
}
