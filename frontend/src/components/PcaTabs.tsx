import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { fetchPcaFactor } from "../api/arrays";
import type { DatasetSummary } from "../api/client";
import { formatValue } from "../format";
import { keptOf, usePca } from "../state/pca";

const MARGIN = { left: 56, right: 14, top: 12, bottom: 34 };

/** A round step for about ``n`` ticks over ``range``. */
function niceStep(range: number, n = 4): number {
  const raw = range / n || 1;
  const pow = 10 ** Math.floor(Math.log10(raw));
  return ([1, 2, 5, 10].find((m) => m * pow >= raw) ?? 10) * pow;
}

function ticksOf(lo: number, hi: number, n = 4): number[] {
  const step = niceStep(hi - lo, n);
  const out: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) out.push(Number(t.toPrecision(6)));
  return out;
}

/** The plot area's size, followed as the pane is resized. */
function useSize() {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  useLayoutEffect(() => {
    const el = ref.current!;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return { ref, ...size };
}

/** Axes with ticks, in an SVG of the given size. */
function Axes({ w, h, x, y, xLabel, yLabel, yFormat = formatValue, children }: {
  w: number; h: number;
  x: { lo: number; hi: number; to: (v: number) => number; ticks: number[] };
  y: { lo: number; hi: number; to: (v: number) => number; ticks: number[] };
  xLabel: string; yLabel: string; yFormat?: (v: number) => string; children: ReactNode;
}) {
  const [x0, x1, y0, y1] = [MARGIN.left, w - MARGIN.right, MARGIN.top, h - MARGIN.bottom];
  return (
    <svg width={w} height={h}>
      <rect className="drift-frame" x={x0} y={y0} width={Math.max(0, x1 - x0)} height={Math.max(0, y1 - y0)} />
      {x.ticks.map((t) => (
        <g key={`x${t}`} className="drift-tick">
          <line x1={x.to(t)} x2={x.to(t)} y1={y1} y2={y1 + 4} />
          <text x={x.to(t)} y={y1 + 16} textAnchor="middle">
            {formatValue(t)}
          </text>
        </g>
      ))}
      {y.ticks.map((t) => (
        <g key={`y${t}`} className="drift-tick">
          <line x1={x0 - 4} x2={x0} y1={y.to(t)} y2={y.to(t)} />
          <text x={x0 - 7} y={y.to(t) + 4} textAnchor="end">
            {yFormat(t)}
          </text>
        </g>
      ))}
      <text className="drift-label" x={(x0 + x1) / 2} y={h - 4} textAnchor="middle">
        {xLabel}
      </text>
      <text className="drift-label" transform={`translate(12 ${(y0 + y1) / 2}) rotate(-90)`} textAnchor="middle">
        {yLabel}
      </text>
      {children}
    </svg>
  );
}

/** The spectrum of the component shown (frame ``index`` of the PCA stack). */
export function ComponentSpectrum({ pca, index }: { pca: DatasetSummary; index: number }) {
  const { ref, w, h } = useSize();
  const [factor, setFactor] = useState<Awaited<ReturnType<typeof fetchPcaFactor>> | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const ac = new AbortController();
    fetchPcaFactor(pca.id, index, ac.signal)
      .then((f) => {
        setFactor(f);
        setError(null);
      })
      .catch((e) => !ac.signal.aborted && setError((e as Error).message));
    return () => ac.abort();
  }, [pca.id, index, pca.filter_rev]);

  let plot = null;
  if (factor && w > 0 && h > 0 && factor.data.length > 1) {
    const n = factor.data.length;
    const e = (i: number) => factor.offset + factor.scale * i;
    const [elo, ehi] = [Math.min(e(0), e(n - 1)), Math.max(e(0), e(n - 1))];
    let [vlo, vhi] = [0, 0];
    for (const v of factor.data) [vlo, vhi] = [Math.min(vlo, v), Math.max(vhi, v)];
    const pad = (vhi - vlo) * 0.05 || 1;
    [vlo, vhi] = [vlo - pad, vhi + pad];
    const [x0, x1, y0, y1] = [MARGIN.left, w - MARGIN.right, MARGIN.top, h - MARGIN.bottom];
    const sx = (v: number) => x0 + ((v - elo) / (ehi - elo)) * (x1 - x0);
    const sy = (v: number) => y1 - ((v - vlo) / (vhi - vlo)) * (y1 - y0);
    const d = Array.from(factor.data, (v, i) => `${i ? "L" : "M"}${sx(e(i)).toFixed(1)} ${sy(v).toFixed(1)}`).join("");
    plot = (
      <Axes
        w={w} h={h}
        x={{ lo: elo, hi: ehi, to: sx, ticks: ticksOf(elo, ehi, 6) }}
        y={{ lo: vlo, hi: vhi, to: sy, ticks: ticksOf(vlo, vhi) }}
        xLabel={factor.units ? `Energy (${factor.units})` : "Energy"}
        yLabel={`Component ${index + 1}`}
      >
        {vlo < 0 && <line className="drift-zero" x1={x0} x2={x1} y1={sy(0)} y2={sy(0)} />}
        <path className="pca-line" d={d} />
      </Axes>
    );
  }
  return (
    <div className="pca-plot" ref={ref}>
      {error ? <p className="field-error">{error}</p> : plot}
    </div>
  );
}

/** Explained variance of each component (log scale). The dashed line after
 * the last component kept (coloured) is dragged to keep more or fewer; a
 * click on a component shows it (``onShow``, 0-based), ringed as ``current``. */
export function ScreePlot({ pca, current, onShow }: {
  pca: DatasetSummary; current?: number; onShow?: (index: number) => void;
}) {
  const { ref, w, h } = useSize();
  const chosen = usePca((s) => s.kept[pca.id]);
  const kept = keptOf(pca, chosen);
  const [hover, setHover] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const [nearLine, setNearLine] = useState(false);
  const e = (pca.pca?.explained ?? []).map((v) => Math.max(v, 1e-12));
  let plot = null;
  if (e.length && w > 0 && h > 0) {
    const k = e.length;
    const [x0, x1, y0, y1] = [MARGIN.left, w - MARGIN.right, MARGIN.top, h - MARGIN.bottom];
    const lmax = Math.ceil(Math.log10(Math.max(...e)));
    const lmin = Math.min(lmax - 1, Math.floor(Math.log10(Math.min(...e))));
    const sx = (i: number) => x0 + (k > 1 ? ((i - 1) / (k - 1)) * (x1 - x0) : (x1 - x0) / 2);
    const sy = (v: number) => y1 - ((Math.log10(v) - lmin) / (lmax - lmin)) * (y1 - y0);
    const decades: number[] = [];
    for (let p = lmin; p <= lmax; p++) decades.push(10 ** p);
    // Position along the axis, in components (1 … k).
    const along = (clientX: number, rect: DOMRect) => 1 + ((clientX - rect.left - x0) / Math.max(1, x1 - x0)) * (k - 1);
    const pick = (clientX: number, rect: DOMRect) => Math.max(1, Math.min(k, Math.round(along(clientX, rect))));
    // The line sits halfway between the last kept and the next.
    const keptAt = (clientX: number, rect: DOMRect) => Math.max(1, Math.min(k, Math.round(along(clientX, rect) - 0.5)));
    const onLine = (clientX: number, rect: DOMRect) => Math.abs(clientX - rect.left - sx(kept + 0.5)) <= 6;
    const path = e.map((v, i) => `${i ? "L" : "M"}${sx(i + 1).toFixed(1)} ${sy(v).toFixed(1)}`).join("");
    plot = (
      <div
        className={`pca-scree${dragging || nearLine ? " on-line" : ""}`}
        onPointerDown={(ev) => {
          const rect = ev.currentTarget.getBoundingClientRect();
          if (onLine(ev.clientX, rect)) {
            ev.currentTarget.setPointerCapture(ev.pointerId);
            setDragging(true);
          } else {
            onShow?.(pick(ev.clientX, rect) - 1);
          }
        }}
        onPointerMove={(ev) => {
          const rect = ev.currentTarget.getBoundingClientRect();
          if (dragging) {
            const n = keptAt(ev.clientX, rect);
            if (n !== kept) usePca.getState().setKept(pca.id, n);
            return;
          }
          setNearLine(onLine(ev.clientX, rect));
          setHover(pick(ev.clientX, rect));
        }}
        onPointerUp={() => setDragging(false)}
        onPointerCancel={() => setDragging(false)}
        onPointerLeave={() => {
          setHover(null);
          setNearLine(false);
        }}
      >
        <Axes
          w={w} h={h}
          x={{ lo: 1, hi: k, to: sx, ticks: ticksOf(1, k, 6).filter((t) => Number.isInteger(t) && t >= 1) }}
          y={{ lo: 10 ** lmin, hi: 10 ** lmax, to: sy, ticks: decades }}
          xLabel="Component"
          yLabel="Explained variance"
          yFormat={(v) => `${formatValue(v * 100)} %`}
        >
          <line
            className={`pca-kept-line${dragging || nearLine ? " active" : ""}`}
            x1={sx(kept + 0.5)} x2={sx(kept + 0.5)} y1={y0} y2={y1}
          />
          <path className="pca-line" d={path} />
          {current !== undefined && current < k && (
            <circle className="pca-current" cx={sx(current + 1)} cy={sy(e[current])} r={7} />
          )}
          {e.map((v, i) => (
            <circle key={i} className={i < kept ? "pca-point kept" : "pca-point"} cx={sx(i + 1)} cy={sy(v)} r={i + 1 === hover ? 4.5 : 3} />
          ))}
          <text className="drift-hover" x={x1 - 6} y={y0 + 14} textAnchor="end">
            {dragging
              ? `Keeping 1–${kept}`
              : nearLine
                ? "Drag to keep more or fewer components"
                : hover !== null
                  ? `Component ${hover}: ${formatValue(e[hover - 1] * 100)} % · click to show it`
                  : ""}
          </text>
        </Axes>
      </div>
    );
  }
  return (
    <div className="pca-plot" ref={ref}>
      {plot}
      <span className="pca-kept-note muted small">
        Keeping {kept} of {e.length}
      </span>
    </div>
  );
}
