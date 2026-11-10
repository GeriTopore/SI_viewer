import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api, type PlotBody, type PlotLimits } from "../api/client";
import { downloadBlob, safeFileName } from "../download";
import { useLayers } from "../state/layers";

/** A curve offered for export; its own x (profiles have one table per line). */
export interface ExportSeries {
  label: string;
  x: ArrayLike<number>;
  y: ArrayLike<number | null>;
  /** Any CSS colour, including var(--…). */
  color: string;
  dash?: number[];
  width?: number;
  axis?: "left" | "right";
}

export interface ExportMarker {
  energy: number;
  label?: string;
  color: string;
  weight: number;
}

export interface ExportWindow {
  lo: number;
  hi: number;
  color: string;
  label: string;
}

interface Props {
  series: ExportSeries[];
  /** X-ray lines (on by default when ``markersOn``). */
  markers?: ExportMarker[];
  markersOn?: boolean;
  /** Energy windows (on by default when ``windowsOn``). */
  windows?: ExportWindow[];
  windowsOn?: boolean;
  /** Default axis labels of the plot (the y label for the overlap style). */
  xLabel: string;
  yLabel: string;
  /** The plot's current zoom: the default x limits (none = the full range). */
  xlim?: [number, number] | null;
  /** The plot's Linear/Log setting. */
  log?: boolean;
  /** File name without extension. */
  name: string;
  onClose: () => void;
}

type Format = "png" | "svg" | "pdf";
type Style = "overlap" | "cascade";

/** Options remembered between exports (numbers as typed). */
interface Stored {
  style: Style;
  padding: string;
  reverse: boolean;
  titleSize: string;
  labelSize: string;
  tickSize: string;
  legendSize: string;
  legend: boolean;
  legendLoc: string;
  lineWidth: string;
  width: string;
  height: string;
  dpi: string;
  format: Format;
}

const STORAGE_KEY = "si-viewer.plotExport";
const DEFAULTS: Stored = {
  style: "overlap",
  padding: "1",
  reverse: false,
  titleSize: "12",
  labelSize: "11",
  tickSize: "10",
  legendSize: "10",
  legend: true,
  legendLoc: "best",
  lineWidth: "1.5",
  width: "6",
  height: "4",
  dpi: "300",
  format: "png",
};

function loadStored(): Stored {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "{}");
    return { ...DEFAULTS, ...saved };
  } catch {
    return DEFAULTS;
  }
}

function saveStored(s: Stored) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
  } catch {
    /* storage unavailable */
  }
}

const LEGEND_LOCATIONS = [
  "best", "upper right", "upper left", "lower left", "lower right", "right", "center left", "center right",
  "lower center", "upper center", "center",
];

const EXTENSIONS: Record<Format, string> = { png: "png", svg: "svg", pdf: "pdf" };

/** A CSS colour (or var(--…)) as #rrggbb, as the server needs it. */
export function toHex(color: string): string {
  let c = color.trim();
  const v = /^var\(\s*(--[\w-]+)\s*(?:,\s*(.+))?\)$/.exec(c);
  if (v) c = getComputedStyle(document.documentElement).getPropertyValue(v[1]).trim() || v[2]?.trim() || c;
  try {
    const ctx = document.createElement("canvas").getContext("2d")!;
    ctx.fillStyle = "#808080";
    ctx.fillStyle = c; // ignored when it isn't a colour
    const out = ctx.fillStyle as string;
    if (out.startsWith("#")) return out;
    const rgb = out.match(/[\d.]+/g)?.slice(0, 3).map((n) => Math.round(Number(n)));
    if (rgb?.length === 3) return `#${rgb.map((n) => n.toString(16).padStart(2, "0")).join("")}`;
  } catch {
    /* no canvas */
  }
  return "#808080";
}

const fmt = (v: number) => String(+v.toPrecision(7));
const PREVIEW_PX = 900;

/** The numbers as sent: gaps (NaN, ±∞, null) become null. */
const values = (y: ArrayLike<number | null>) => Array.from(y, (v) => (v !== null && Number.isFinite(v) ? v : null));

interface Row {
  include: boolean;
  label: string;
  axis: "left" | "right";
}

/** Dialog for saving a 1D plot (spectra, line profiles) as a PNG, SVG or PDF
 * drawn by the server with matplotlib: a live preview and the options of the
 * style, axes, limits, text and figure. */
export function PlotExportDialog(props: Props) {
  // The plot as it was when the dialog opened.
  const [init] = useState(() => ({
    data: props.series.map((s) => ({
      x: Array.from(s.x),
      y: values(s.y),
      color: toHex(s.color),
      dash: s.dash,
      width: s.width,
    })),
    markers: (props.markers ?? []).map((m) => ({ x: m.energy, label: m.label, color: toHex(m.color), weight: m.weight })),
    windows: (props.windows ?? []).map((w) => ({ lo: w.lo, hi: w.hi, color: toHex(w.color), label: w.label })),
    rows: props.series.map<Row>((s) => ({ include: true, label: s.label, axis: s.axis ?? "left" })),
    xlim: props.xlim ? ([fmt(props.xlim[0]), fmt(props.xlim[1])] as [string, string]) : (["", ""] as [string, string]),
    log: !!props.log,
    xLabel: props.xLabel,
    yLabel: props.yLabel,
    name: props.name,
  }));
  const { data, markers, windows, xLabel, yLabel, name } = init;
  const { onClose } = props;

  const [stored, setStored] = useState(loadStored);
  const patch = (p: Partial<Stored>) => setStored((s) => ({ ...s, ...p }));

  // Per plot: they start from the plot each time.
  const [rows, setRows] = useState(init.rows);
  const setRow = (i: number, p: Partial<Row>) => setRows((r) => r.map((row, j) => (j === i ? { ...row, ...p } : row)));
  const [normalise, setNormalise] = useState(false);
  const [title, setTitle] = useState("");
  const [xText, setXText] = useState<string>();
  const [yText, setYText] = useState<string>();
  const [y2Text, setY2Text] = useState("");
  const [log, setLog] = useState(init.log);
  const [xlim, setXlim] = useState(init.xlim);
  const [ylim, setYlim] = useState<[string, string]>(["", ""]);
  const [y2lim, setY2lim] = useState<[string, string]>(["", ""]);
  const [markersOn, setMarkersOn] = useState(props.markersOn ?? markers.length > 0);
  const [windowsOn, setWindowsOn] = useState(props.windowsOn ?? windows.length > 0);

  const cascade = stored.style === "cascade";
  const used = rows.filter((r) => r.include);
  const onRight = !cascade && used.some((r) => r.axis === "right") && used.some((r) => r.axis === "left");
  const xLabelShown = xText ?? xLabel;
  const yLabelShown = yText ?? (cascade ? "Intensity (a.u.)" : normalise ? "Normalised intensity" : yLabel);

  // The request, or what is wrong with the options.
  const built = useMemo<{ body: PlotBody } | { error: string }>(() => {
    try {
      const num = (what: string, text: string, min = 0) => {
        const v = Number(text);
        if (text.trim() === "" || !Number.isFinite(v)) throw new Error(`${what}: enter a number`);
        if (v < min) throw new Error(`${what} must be ${min} or more`);
        return v;
      };
      const limits = (what: string, [lo, hi]: [string, string]): PlotLimits => [
        lo.trim() === "" ? null : num(`${what} min`, lo, -Infinity),
        hi.trim() === "" ? null : num(`${what} max`, hi, -Infinity),
      ];
      if (!used.length) throw new Error("Select a series");
      return {
        body: {
          series: rows.flatMap((r, i) =>
            r.include
              ? [{
                  label: r.label,
                  x: data[i].x,
                  y: data[i].y,
                  color: data[i].color,
                  dash: data[i].dash,
                  width: data[i].width,
                  axis: cascade ? "left" : r.axis,
                } as PlotBody["series"][number]]
              : [],
          ),
          markers: markersOn ? markers : [],
          windows: windowsOn ? windows : [],
          options: {
            style: stored.style,
            padding: num("Spacing", stored.padding),
            normalise,
            reverse: stored.reverse,
            log: log && !cascade,
            title,
            x_label: xLabelShown,
            y_label: yLabelShown,
            y2_label: onRight ? y2Text : "",
            xlim: limits("X", xlim),
            ylim: limits("Y", ylim),
            y2lim: onRight ? limits("Y2", y2lim) : [null, null],
            title_size: num("Title size", stored.titleSize),
            label_size: num("Label size", stored.labelSize),
            tick_size: num("Tick size", stored.tickSize),
            legend_size: num("Legend size", stored.legendSize),
            legend: stored.legend,
            legend_loc: stored.legendLoc,
            line_width: num("Line width", stored.lineWidth),
            width: num("Width", stored.width),
            height: num("Height", stored.height),
            dpi: Math.round(num("DPI", stored.dpi)),
            format: stored.format,
          },
        },
      };
    } catch (e) {
      return { error: (e as Error).message };
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows, stored, normalise, title, xLabelShown, yLabelShown, y2Text, log, xlim, ylim, y2lim, markersOn, windowsOn]);
  const body = "body" in built ? built.body : null;
  const localError = "error" in built ? built.error : null;

  // Preview: rendered at a low resolution, refreshed shortly after a change;
  // a request still running for older options is dropped.
  const [url, setUrl] = useState<string | null>(null);
  const urlRef = useRef<string | null>(null);
  const [rendering, setRendering] = useState(false);
  const [serverError, setServerError] = useState<string | null>(null);
  useEffect(() => {
    if (!body) {
      setRendering(false);
      return;
    }
    setRendering(true);
    const ac = new AbortController();
    const timer = setTimeout(async () => {
      const o = body.options;
      const dpi = Math.max(30, Math.min(110, Math.floor(PREVIEW_PX / Math.max(o.width, o.height, 1))));
      try {
        const blob = await api.renderPlot({ ...body, options: { ...o, format: "png", dpi } }, ac.signal);
        if (urlRef.current) URL.revokeObjectURL(urlRef.current);
        urlRef.current = URL.createObjectURL(blob);
        setUrl(urlRef.current);
        setServerError(null);
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        setServerError((e as Error).message);
      }
      setRendering(false);
    }, 300);
    return () => {
      clearTimeout(timer);
      ac.abort();
    };
  }, [body]);
  useEffect(
    () => () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    },
    [],
  );

  const error = localError ?? serverError;
  const size = body
    ? body.options.format === "png"
      ? `${Math.round(body.options.width * body.options.dpi)} × ${Math.round(body.options.height * body.options.dpi)} px`
      : `${body.options.width} × ${body.options.height} in`
    : null;

  const [busy, setBusy] = useState(false);
  const save = async () => {
    if (!body || busy || error) return;
    setBusy(true);
    try {
      downloadBlob(await api.renderPlot(body), `${safeFileName(name)}.${EXTENSIONS[body.options.format]}`);
      saveStored(stored);
      onClose();
    } catch (e) {
      useLayers.getState().notify(`Could not save the plot: ${(e as Error).message}`);
      setBusy(false);
    }
  };
  const latest = useRef({ save, onClose });
  latest.current = { save, onClose };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") latest.current.onClose();
      else if (
        e.key === "Enter" && !e.metaKey && !e.ctrlKey && !e.altKey && !e.shiftKey &&
        !(e.target instanceof Element && e.target.closest("button, select, textarea"))
      ) {
        e.preventDefault();
        void latest.current.save();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const png = stored.format === "png";
  return createPortal(
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal png-dialog plot-dialog" role="dialog" aria-label="Save plot">
        <h2>Save plot</h2>
        <div className="png-body">
          <div className="png-preview-col">
            <div className={`png-preview plot-preview${rendering ? " stale" : ""}`}>
              {url ? <img src={url} alt="Preview" /> : <span className="muted small">{rendering ? "…" : ""}</span>}
            </div>
            <div className="muted small png-size">
              {rendering && <span className="spinner" aria-hidden />}
              {size}
            </div>
          </div>
          <div className="png-options">
            <h3>Series</h3>
            {rows.map((r, i) => (
              <div className="plot-series" key={i}>
                <input
                  type="checkbox"
                  checked={r.include}
                  onChange={(e) => setRow(i, { include: e.target.checked })}
                  title="Include"
                />
                <span className="plot-swatch" style={{ background: data[i].color }} />
                <input
                  type="text"
                  value={r.label}
                  disabled={!r.include}
                  onChange={(e) => setRow(i, { label: e.target.value })}
                  title="Legend text"
                />
                <select
                  value={r.axis}
                  disabled={!r.include || cascade}
                  onChange={(e) => setRow(i, { axis: e.target.value as "left" | "right" })}
                  title="Axis"
                >
                  <option value="left">Left</option>
                  <option value="right">Right</option>
                </select>
              </div>
            ))}

            <h3>Style</h3>
            <Row label="Layout">
              <Select
                value={stored.style}
                onChange={(style) => patch({ style })}
                options={[["overlap", "Overlap"], ["cascade", "Cascade"]]}
              />
            </Row>
            <Row label="Spacing">
              <Num value={stored.padding} onChange={(padding) => patch({ padding })} disabled={!cascade} />
            </Row>
            <Row label="Normalise">
              <input type="checkbox" checked={normalise} onChange={(e) => setNormalise(e.target.checked)} />
            </Row>
            <Row label="Reverse">
              <input
                type="checkbox"
                checked={stored.reverse}
                disabled={!cascade}
                onChange={(e) => patch({ reverse: e.target.checked })}
                title="Reverse order"
              />
            </Row>
            <Row label="Line width">
              <Num value={stored.lineWidth} onChange={(lineWidth) => patch({ lineWidth })} />
              <span className="muted">pt</span>
            </Row>

            <h3>Axes</h3>
            <Row label="Title">
              <input type="text" value={title} onChange={(e) => setTitle(e.target.value)} />
            </Row>
            <Row label="X label">
              <input type="text" value={xLabelShown} onChange={(e) => setXText(e.target.value)} />
            </Row>
            <Row label="Y label">
              <input type="text" value={yLabelShown} onChange={(e) => setYText(e.target.value)} />
            </Row>
            {onRight && (
              <Row label="Y2 label">
                <input type="text" value={y2Text} onChange={(e) => setY2Text(e.target.value)} />
              </Row>
            )}
            <Row label="Log y">
              <input
                type="checkbox"
                checked={log && !cascade}
                disabled={cascade}
                onChange={(e) => setLog(e.target.checked)}
              />
            </Row>

            <h3>Limits</h3>
            <Limits label="X" value={xlim} onChange={setXlim} reset={init.xlim} />
            <Limits label="Y" value={ylim} onChange={setYlim} reset={["", ""]} />
            {onRight && <Limits label="Y2" value={y2lim} onChange={setY2lim} reset={["", ""]} />}

            <h3>Text</h3>
            <Row label="Title">
              <Num value={stored.titleSize} onChange={(titleSize) => patch({ titleSize })} />
              <span className="muted">pt</span>
            </Row>
            <Row label="Labels">
              <Num value={stored.labelSize} onChange={(labelSize) => patch({ labelSize })} />
              <span className="muted">pt</span>
            </Row>
            <Row label="Ticks">
              <Num value={stored.tickSize} onChange={(tickSize) => patch({ tickSize })} />
              <span className="muted">pt</span>
            </Row>
            <Row label="Legend">
              <input type="checkbox" checked={stored.legend} onChange={(e) => patch({ legend: e.target.checked })} />
              <Select
                value={stored.legendLoc}
                onChange={(legendLoc) => patch({ legendLoc })}
                disabled={!stored.legend}
                options={LEGEND_LOCATIONS.map((l) => [l, l[0].toUpperCase() + l.slice(1)])}
              />
              <Num
                value={stored.legendSize}
                onChange={(legendSize) => patch({ legendSize })}
                disabled={!stored.legend}
                title="Font size (pt)"
              />
            </Row>

            {(markers.length > 0 || windows.length > 0) && (
              <>
                <h3>Annotations</h3>
                {markers.length > 0 && (
                  <Row label="X-ray lines">
                    <input type="checkbox" checked={markersOn} onChange={(e) => setMarkersOn(e.target.checked)} />
                  </Row>
                )}
                {windows.length > 0 && (
                  <Row label="Windows">
                    <input type="checkbox" checked={windowsOn} onChange={(e) => setWindowsOn(e.target.checked)} />
                  </Row>
                )}
              </>
            )}

            <h3>Figure</h3>
            <Row label="Size">
              <Num value={stored.width} onChange={(width) => patch({ width })} min={1} title="Width" />
              <span className="muted">×</span>
              <Num value={stored.height} onChange={(height) => patch({ height })} min={1} title="Height" />
              <span className="muted">in</span>
            </Row>
            <Row label="DPI">
              <Num value={stored.dpi} onChange={(dpi) => patch({ dpi })} disabled={!png} min={30} step="1" />
            </Row>
            <Row label="Format">
              <Select
                value={stored.format}
                onChange={(format) => patch({ format })}
                options={[["png", "PNG"], ["svg", "SVG"], ["pdf", "PDF"]]}
              />
            </Row>
          </div>
        </div>
        {error && <p className="field-error">{error}</p>}
        <div className="rebin-actions">
          <button className="secondary" onClick={onClose}>
            Cancel
          </button>
          <button className="primary" onClick={save} disabled={!body || !!error || busy}>
            Save
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="png-row">
      <span className="png-row-label">{label}</span>
      <span className="png-row-field">{children}</span>
    </div>
  );
}

/** Minimum and maximum of an axis (blank = automatic) and a Reset to ``reset``. */
function Limits({
  label, value, onChange, reset,
}: { label: string; value: [string, string]; onChange: (v: [string, string]) => void; reset: [string, string] }) {
  const changed = value[0] !== reset[0] || value[1] !== reset[1];
  return (
    <Row label={label}>
      <input
        type="number"
        step="any"
        value={value[0]}
        placeholder="Min"
        onChange={(e) => onChange([e.target.value, value[1]])}
      />
      <input
        type="number"
        step="any"
        value={value[1]}
        placeholder="Max"
        onChange={(e) => onChange([value[0], e.target.value])}
      />
      <button className="link-button" disabled={!changed} onClick={() => onChange(reset)}>
        Reset
      </button>
    </Row>
  );
}

function Num({
  value, onChange, disabled, min = 0, step = "any", title,
}: { value: string; onChange: (v: string) => void; disabled?: boolean; min?: number; step?: string; title?: string }) {
  return (
    <input
      type="number"
      min={min}
      step={step}
      value={value}
      disabled={disabled}
      title={title}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

function Select<T extends string>({
  value, onChange, options, disabled,
}: { value: T; onChange: (v: T) => void; options: [T, string][] | string[][]; disabled?: boolean }) {
  return (
    <select value={value} disabled={disabled} onChange={(e) => onChange(e.target.value as T)}>
      {(options as [T, string][]).map(([v, label]) => (
        <option key={v} value={v}>
          {label}
        </option>
      ))}
    </select>
  );
}
