import { useEffect, useRef, useState } from "react";
import { useUi } from "../state/ui";
import { lineColor, plotlyScale, type LineMap } from "../viewer/diverging";

export type Plot3DMode = "surface" | "lines";

// Plotly is large: loaded the first time a 3D plot is shown.
type PlotlyModule = typeof import("plotly.js-dist-min");
let plotly: Promise<PlotlyModule> | null = null;
const loadPlotly = () => (plotly ??= import("plotly.js-dist-min").then((m) => (m.default ?? m) as PlotlyModule));

const cssVar = (name: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

interface Props {
  /** Energies (columns). */
  x: number[];
  /** Values of the row axis (time, position…). */
  y: number[];
  /** Intensities, one row per ``y``. */
  z: number[][];
  mode: Plot3DMode;
  xTitle: string;
  yTitle: string;
  zTitle: string;
  /** Colours along the row axis (first row to last). */
  cmap: LineMap;
}

/** A rotatable 3D plot (Plotly) of ``z`` over energy and the row axis: a
 * surface, or each spectrum as a line at its place on the row axis (a 3D
 * cascade); both coloured by ``cmap`` along the row axis. Drag rotates,
 * scroll zooms, right-drag pans; the view is kept when the data changes. */
export function Plot3D({ x, y, z, mode, xTitle, yTitle, zTitle, cmap }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const theme = useUi((s) => s.theme);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    loadPlotly()
      .then((Plotly) => {
        if (cancelled || !ref.current) return;
        const text = cssVar("--text");
        const grid = cssVar("--border");
        const axis = (title: string) => ({
          title: { text: title },
          color: text,
          gridcolor: grid,
          zerolinecolor: grid,
          showbackground: false,
        });
        const t = (i: number) => (y.length > 1 ? i / (y.length - 1) : 0);
        const surfaceColor = z.map((_, i) => z[0].map(() => t(i)));
        const hover = `${xTitle}: %{x:.4g}<br>${yTitle}: %{y:.4g}<br>${zTitle}: %{z:.4g}<extra></extra>`;
        const traces =
          mode === "surface"
            ? [{
                type: "surface" as const,
                x, y, z,
                surfacecolor: surfaceColor,
                colorscale: plotlyScale(cmap),
                cmin: 0,
                cmax: 1,
                showscale: false,
                hovertemplate: hover,
              }]
            : z.map((row, i) => ({
                type: "scatter3d" as const,
                mode: "lines" as const,
                x,
                y: x.map(() => y[i]),
                z: row,
                line: { color: lineColor(cmap, t(i)), width: 3 },
                hovertemplate: hover,
              }));
        const layout = {
          uirevision: "keep", // the camera stays when the data or mode changes
          margin: { l: 0, r: 0, t: 0, b: 0 },
          paper_bgcolor: "rgba(0,0,0,0)",
          showlegend: false,
          font: { family: "-apple-system, sans-serif", size: 11, color: text },
          scene: {
            xaxis: axis(xTitle),
            yaxis: axis(yTitle),
            zaxis: axis(zTitle),
            aspectmode: "manual" as const,
            aspectratio: { x: 1.4, y: 1, z: 0.7 },
            // Energy left to right, the row axis going back.
            camera: { eye: { x: 0.35, y: -1.9, z: 1.1 } },
          },
        };
        // Surface and line traces in one array, as Plotly takes them.
        const data = traces as unknown as Parameters<PlotlyModule["react"]>[1];
        return Plotly.react(ref.current, data, layout, { displaylogo: false, responsive: true }).then(() => {
          if (!cancelled) {
            setLoading(false);
            setError(null);
          }
        });
      })
      .catch((e) => !cancelled && (setError((e as Error).message), setLoading(false)));
    return () => {
      cancelled = true;
    };
  }, [x, y, z, mode, xTitle, yTitle, zTitle, theme, cmap]);

  // Follows the panel's size; purged when closed.
  useEffect(() => {
    const el = ref.current!;
    const ro = new ResizeObserver(() => plotly?.then((P) => el.isConnected && P.Plots.resize(el)));
    ro.observe(el);
    return () => {
      ro.disconnect();
      plotly?.then((P) => P.purge(el));
    };
  }, []);

  return (
    <div className="plot3d">
      <div ref={ref} className="plot3d-canvas" />
      {(loading || error) && (
        <div className={`image-message${error ? " error" : ""}`}>
          {loading && !error && <span className="spinner" aria-hidden />}
          {error ?? "Loading the 3D view…"}
        </div>
      )}
    </div>
  );
}
