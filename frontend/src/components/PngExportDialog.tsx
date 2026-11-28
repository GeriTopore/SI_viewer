import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { downloadBlob } from "../download";
import { scaleBarLabel, titledText } from "../format";
import { useLayers } from "../state/layers";
import {
  MAX_FIGURE_AREA, MAX_FIGURE_SIDE, autoBarLength, autoBarSizes, autoColourBar, autoFactor, autoLabelFont, autoSpacing,
  barPixels, canvasBlob, colourBarLayer, composeFigure, decorate, factorChoices, figureLayout, renderBase,
  type BarBackground, type Corner, type Edge, type FigureStyle, type Ink, type MarginStyle, type SnapshotPanel,
} from "../viewer/snapshot";

type BarOn = "first" | "all" | "none";
type GridBackground = "dark" | "black" | "white";

/** Options remembered between exports. */
interface Stored {
  show: boolean;
  barOn: BarOn;
  label: boolean;
  corner: Corner;
  colour: Ink;
  background: BarBackground;
  gridBackground: GridBackground;
  labels: boolean;
  labelColour: Ink;
  /** A single image's colour bar (beside which edge; strain as %), its axes,
   * and the margins they are drawn in. */
  cbShow: boolean;
  cbEdge: Edge;
  cbPercent: boolean;
  axesShow: boolean;
  marginBackground: MarginStyle["background"];
  marginColour: Ink;
  /** Sections of the options folded away. */
  folded: string[];
}

const STORAGE_KEY = "si-viewer.pngExport";
const DEFAULTS: Stored = {
  show: true,
  barOn: "first",
  label: true,
  corner: "lr",
  colour: "white",
  background: "none",
  gridBackground: "dark",
  labels: true,
  labelColour: "white",
  cbShow: false,
  cbEdge: "right",
  cbPercent: false,
  axesShow: false,
  marginBackground: "white",
  marginColour: "black",
  folded: [],
};
const GRID_COLOURS: Record<GridBackground, string> = { dark: "#0f1115", black: "#000", white: "#fff" };

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

/** Sizes typed in this session (text); absent = automatic. */
interface Typed {
  thickness?: string;
  font?: string;
  labelFont?: string;
  spacing?: string;
  padding?: string;
  cbWidth?: string;
  cbDistance?: string;
  cbFont?: string;
  cbTicks?: string;
  cbDecimals?: string;
  axesFont?: string;
  axesTicks?: string;
  axesDecimals?: string;
}
let typedInSession: Typed = {};

/** What is typed for one image (by its file name), kept while the app is
 * open: the scale bar length and the titles (absent = automatic). */
interface PerImage {
  length?: string;
  xTitle?: string;
  xUnits?: string;
  yTitle?: string;
  yUnits?: string;
  cbTitle?: string;
  cbUnits?: string;
}
const perImage = new Map<string, PerImage>();

const num = (t: string) => (t.trim() === "" ? NaN : Number(t));
const PREVIEW = 420;

interface Props {
  panels: SnapshotPanel[];
  /** Columns of the grid (1 for a single image). */
  cols: number | number[];
  /** A grid of labelled panels (else a single image). */
  grid: boolean;
  /** File name of the download. */
  name: string;
  /** Whether the image can carry a scale bar (false e.g. for an FFT). */
  scaleBar?: boolean;
  onClose: () => void;
}

/** Dialog for saving the image or grid as a PNG: a live preview and the
 * options of the scale bar, labels and layout. */
export function PngExportDialog({ panels, cols, grid, name, scaleBar = true, onClose }: Props) {
  const choices = useMemo(() => factorChoices(panels), [panels]);
  const [factor, setFactor] = useState(() => Math.min(...panels.map((p) => autoFactor(p.width, p.height))));
  const [stored, setStored] = useState(loadStored);
  const [typed, setTyped] = useState<Typed>(typedInSession);
  const [mine, setMine] = useState<PerImage>(() => perImage.get(name) ?? {});
  const setOwn = (k: keyof PerImage, v: string) => {
    const next = { ...(perImage.get(name) ?? {}), [k]: v };
    perImage.set(name, next);
    setMine(next);
  };
  const lengthText = mine.length;
  const setLengthText = (v: string) => setOwn("length", v);
  const [busy, setBusy] = useState(false);
  const patch = (p: Partial<Stored>) => setStored((s) => ({ ...s, ...p }));
  // Kept as soon as changed, whether the dialog is saved or closed.
  useEffect(() => saveStored(stored), [stored]);
  const fold = (id: string) => ({
    folded: stored.folded.includes(id),
    onToggle: () => patch({ folded: stored.folded.includes(id) ? stored.folded.filter((x) => x !== id) : [...stored.folded, id] }),
  });
  const type = (k: keyof Typed, v: string) => {
    typedInSession = { ...typedInSession, [k]: v };
    setTyped(typedInSession);
  };

  // The panels rendered once per factor; everything else is 2D drawing.
  const [bases, setBases] = useState<{ factor: number; canvases: HTMLCanvasElement[] } | null>(null);
  const [renderError, setRenderError] = useState<string | null>(null);
  useEffect(() => {
    setRenderError(null);
    try {
      setBases({ factor, canvases: panels.map((p) => renderBase(p, factor)) });
    } catch (e) {
      const msg = (e as Error).message;
      setRenderError(msg);
      useLayers.getState().notify(`Could not render the image: ${msg}`);
    }
  }, [panels, factor]);
  const ready = bases && bases.factor === factor ? bases.canvases : null;

  const first = panels[0];
  const autoLength = autoBarLength(first.calX, first.width);
  const barAvailable = scaleBar && autoLength !== null;
  const units = first.calX.units || "px";

  // Automatic values depend on the rendered size.
  const layout0 = useMemo(
    () => (ready ? figureLayout(ready.map((b) => ({ w: b.width, h: b.height })), cols, 0, 0) : null),
    [ready, cols],
  );
  const barAuto = autoBarSizes(layout0?.rects[0].w ?? 1024);
  const cbAuto = autoColourBar(layout0?.width ?? 1024, layout0?.height ?? 1024);
  const autos = {
    thickness: String(barAuto.thickness),
    font: String(barAuto.font),
    labelFont: String(autoLabelFont(layout0?.cellW ?? 1024)),
    spacing: layout0 ? String(autoSpacing(layout0.cellW, layout0.cellH)) : "0",
    padding: "0",
    cbWidth: String(cbAuto.width),
    cbDistance: String(cbAuto.distance),
    cbFont: String(Math.round(barAuto.font * 0.8)),
    cbTicks: "3",
    cbDecimals: "",
    axesFont: String(Math.round(barAuto.font * 0.8)),
    axesTicks: "5",
    axesDecimals: "",
  };
  const text = (k: keyof Typed) => typed[k] ?? autos[k];

  const barOn: BarOn = !barAvailable ? "none" : grid ? stored.barOn : stored.show ? "first" : "none";
  const barVisible = barOn !== "none";
  const length = num(lengthText ?? String(autoLength ?? ""));
  const thickness = num(text("thickness"));
  const font = num(text("font"));
  const labelFont = num(text("labelFont"));
  const spacing = grid ? num(text("spacing")) : 0;
  const padding = grid ? num(text("padding")) : 0;
  const labelsVisible = grid && stored.labels;
  // A colour bar for a single map drawn with a colour map.
  const cbLayer = grid ? null : colourBarLayer(first);
  const cbVisible = !!cbLayer && stored.cbShow;
  const cbWidth = num(text("cbWidth"));
  const cbDistance = num(text("cbDistance"));
  const cbFont = num(text("cbFont"));
  const cbTicks = num(text("cbTicks"));
  const cbDecimals = decimalsOf(text("cbDecimals"));
  const cbPercent = !!cbLayer?.fraction && stored.cbPercent;
  // Axes of a single image.
  const axesAvailable = !grid;
  const axesVisible = axesAvailable && stored.axesShow;
  const axesFont = num(text("axesFont"));
  const axesTicks = num(text("axesTicks"));
  const axesDecimals = decimalsOf(text("axesDecimals"));
  const marginsShown = cbVisible || axesVisible;
  // Titles: typed, else the image's own (percent: the colour bar's units are %).
  const auto = first.titles;
  const own = (k: keyof PerImage, fallback: string) => mine[k] ?? fallback;
  const xTitle = own("xTitle", auto?.x.title ?? "x"), xUnits = own("xUnits", auto?.x.units ?? first.calX.units);
  const yTitle = own("yTitle", auto?.y.title ?? "y"), yUnits = own("yUnits", auto?.y.units ?? (first.calY ?? first.calX).units);
  const cbTitle = own("cbTitle", auto?.value.title ?? ""), cbUnits = own("cbUnits", cbPercent ? "%" : auto?.value.units ?? "");

  const figure = useMemo(() => {
    if (!ready || !layout0) return { error: renderError };
    let error: string | null = null;
    if (barVisible) {
      if (!(length > 0)) error = "Length must be positive";
      else if (!(thickness > 0)) error = "Thickness must be positive";
      else if (stored.label && !(font > 0)) error = "Font size must be positive";
      else if (barPixels(length, first.calX, first.width, layout0.rects[0]) > layout0.rects[0].w)
        error = "Length exceeds the image width";
    }
    if (!error && labelsVisible && !(labelFont > 0)) error = "Label font size must be positive";
    if (!error && grid && !(spacing >= 0)) error = "Spacing must be zero or more";
    if (!error && grid && !(padding >= 0)) error = "Padding must be zero or more";
    if (!error && cbVisible) {
      if (!(cbWidth > 0)) error = "Colour bar width must be positive";
      else if (!(cbDistance >= 0)) error = "Colour bar distance must be zero or more";
      else if (!(cbFont > 0)) error = "Colour bar font size must be positive";
      else if (!(Number.isInteger(cbTicks) && cbTicks >= 2)) error = "Colour bar ticks: a whole number, 2 or more";
      else if (Number.isNaN(cbDecimals)) error = "Colour bar decimals: a whole number from 0 to 10, or empty";
    }
    if (!error && axesVisible) {
      if (!(axesFont > 0)) error = "Axes font size must be positive";
      else if (!(Number.isInteger(axesTicks) && axesTicks >= 2)) error = "Axes ticks: a whole number, 2 or more";
      else if (Number.isNaN(axesDecimals)) error = "Axes decimals: a whole number from 0 to 10, or empty";
    }
    if (error) return { error };
    const style: FigureStyle = {
      cols,
      spacing,
      padding,
      background: GRID_COLOURS[stored.gridBackground],
      labels: labelsVisible ? { font: labelFont, colour: stored.labelColour } : null,
      bar: barVisible
        ? {
            length,
            thickness,
            font: stored.label ? font : barAuto.font,
            label: stored.label,
            corner: stored.corner,
            colour: stored.colour,
            background: stored.background,
          }
        : null,
      barOn: barOn === "all" ? "all" : "first",
    };
    const size = figureLayout(ready.map((b) => ({ w: b.width, h: b.height })), cols, spacing, padding);
    if (size.width > MAX_FIGURE_SIDE || size.height > MAX_FIGURE_SIDE || size.width * size.height > MAX_FIGURE_AREA)
      return { error: "Too large; use a smaller scale" };
    try {
      const figure = composeFigure(
        ready.map((base, i) => ({
          base,
          dataWidth: panels[i].width,
          calX: panels[i].calX,
          label: panels[i].label,
          colour: panels[i].colour,
        })),
        style,
      );
      if (!marginsShown) return { canvas: figure };
      return {
        canvas: decorate(
          figure, first, cbLayer,
          axesVisible ? {
            font: axesFont, ticks: axesTicks, decimals: axesDecimals,
            x: titledText({ title: xTitle, units: xUnits }), y: titledText({ title: yTitle, units: yUnits }),
          } : null,
          cbVisible ? {
            edge: stored.cbEdge, width: cbWidth, distance: cbDistance, font: cbFont,
            ticks: cbTicks, decimals: cbDecimals, percent: cbPercent,
            title: titledText({ title: cbTitle, units: cbUnits }),
          } : null,
          { background: stored.marginBackground, colour: stored.marginColour },
        ),
      };
    } catch (e) {
      return { error: (e as Error).message };
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, layout0, stored, length, thickness, font, labelFont, spacing, padding, grid, cols, barOn, cbLayer, cbWidth, cbDistance, cbFont, cbTicks, cbDecimals, cbPercent, axesVisible, axesFont, axesTicks, axesDecimals, xTitle, xUnits, yTitle, yUnits, cbTitle, cbUnits]);
  const canvas = "canvas" in figure ? figure.canvas : null;
  const error = "error" in figure ? figure.error : null;

  // Output size (of the last valid figure).
  const [size, setSize] = useState<string>();
  const preview = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const el = preview.current;
    if (!canvas || !el) return;
    setSize(`${canvas.width} × ${canvas.height} px`);
    const room = Math.min(PREVIEW, (el.parentElement?.clientWidth ?? PREVIEW) - 18);
    const f = Math.min(room / canvas.width, PREVIEW / canvas.height);
    const dpr = window.devicePixelRatio || 1;
    el.style.width = `${Math.round(canvas.width * f)}px`;
    el.style.height = `${Math.round(canvas.height * f)}px`;
    el.width = Math.max(1, Math.round(canvas.width * f * dpr));
    el.height = Math.max(1, Math.round(canvas.height * f * dpr));
    const ctx = el.getContext("2d")!;
    ctx.imageSmoothingEnabled = el.width < canvas.width;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(canvas, 0, 0, el.width, el.height);
  }, [canvas]);

  const save = async () => {
    if (!canvas || busy) return;
    setBusy(true);
    try {
      downloadBlob(await canvasBlob(canvas), name);
      saveStored(stored);
      onClose();
    } catch (e) {
      useLayers.getState().notify(`Could not save the image: ${(e as Error).message}`);
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

  const barDisabled = !barVisible;
  return createPortal(
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal png-dialog" role="dialog" aria-label="Save PNG">
        <h2>Save PNG</h2>
        <div className="png-body">
          <div className="png-preview-col">
            <div className="png-preview">
              <canvas ref={preview} />
            </div>
            <div className="muted small png-size">{size ?? "…"}</div>
          </div>
          <div className="png-options">
            <Row label="Scale">
              <select value={factor} onChange={(e) => setFactor(Number(e.target.value))}>
                {choices.map((f) => (
                  <option key={f} value={f}>
                    {f}×
                  </option>
                ))}
              </select>
            </Row>

            {barAvailable && (
              <Fold title="Scale bar" {...fold("bar")}>
                {grid ? (
                  <Row label="On">
                    <Select
                      value={stored.barOn}
                      onChange={(barOn) => patch({ barOn })}
                      options={[["first", "First panel"], ["all", "All panels"], ["none", "None"]]}
                    />
                  </Row>
                ) : (
                  <Row label="Show">
                    <input type="checkbox" checked={stored.show} onChange={(e) => patch({ show: e.target.checked })} />
                  </Row>
                )}
                <Row label="Length">
                  <Num
                    value={lengthText ?? String(autoLength)}
                    onChange={setLengthText}
                    disabled={barDisabled}
                    title="In the calibrated units of the image"
                  />
                  <span className="muted">
                    {units}
                    {length > 0 && scaleBarLabel(length, first.calX.units) !== `${length} ${units}` &&
                      ` · ${scaleBarLabel(length, first.calX.units)}`}
                  </span>
                </Row>
                <Row label="Thickness">
                  <Num value={text("thickness")} onChange={(v) => type("thickness", v)} disabled={barDisabled} />
                  <span className="muted">px</span>
                </Row>
                <Row label="Label">
                  <input
                    type="checkbox"
                    checked={stored.label}
                    disabled={barDisabled}
                    onChange={(e) => patch({ label: e.target.checked })}
                  />
                  <Num
                    value={text("font")}
                    onChange={(v) => type("font", v)}
                    disabled={barDisabled || !stored.label}
                    title="Font size"
                  />
                  <span className="muted">px</span>
                </Row>
                <Row label="Position">
                  <Select
                    value={stored.corner}
                    onChange={(corner) => patch({ corner })}
                    disabled={barDisabled}
                    options={[["lr", "Lower right"], ["ll", "Lower left"], ["ur", "Upper right"], ["ul", "Upper left"]]}
                  />
                </Row>
                <Row label="Colour">
                  <Select
                    value={stored.colour}
                    onChange={(colour) => patch({ colour })}
                    disabled={barDisabled}
                    options={[["white", "White"], ["black", "Black"]]}
                  />
                </Row>
                <Row label="Background">
                  <Select
                    value={stored.background}
                    onChange={(background) => patch({ background })}
                    disabled={barDisabled}
                    options={[["none", "None"], ["black", "Black"], ["white", "White"], ["translucent", "Translucent"]]}
                  />
                </Row>
              </Fold>
            )}

            {cbLayer && (
              <Fold title="Colour bar" {...fold("cb")}>
                <Row label="Show">
                  <input type="checkbox" checked={stored.cbShow} onChange={(e) => patch({ cbShow: e.target.checked })} />
                </Row>
                <Row label="Edge">
                  <Select
                    value={stored.cbEdge}
                    onChange={(cbEdge) => patch({ cbEdge })}
                    disabled={!cbVisible}
                    options={[["right", "Right"], ["left", "Left"], ["top", "Top"], ["bottom", "Bottom"]]}
                  />
                </Row>
                <Row label="Width">
                  <Num value={text("cbWidth")} onChange={(v) => type("cbWidth", v)} disabled={!cbVisible} />
                  <span className="muted">px</span>
                </Row>
                <Row label="Distance">
                  <Num value={text("cbDistance")} onChange={(v) => type("cbDistance", v)} disabled={!cbVisible} title="From the image, or its axis" />
                  <span className="muted">px</span>
                </Row>
                <TitleRow label="Title" title={cbTitle} units={cbUnits} disabled={!cbVisible}
                  onTitle={(v) => setOwn("cbTitle", v)} onUnits={(v) => setOwn("cbUnits", v)} />
                <TickRows prefix="cb" text={text} type={type} disabled={!cbVisible} />
                {cbLayer.fraction && (
                  <Row label="Percent">
                    <input type="checkbox" checked={stored.cbPercent} disabled={!cbVisible}
                      onChange={(e) => patch({ cbPercent: e.target.checked })} title="Values × 100" />
                  </Row>
                )}
              </Fold>
            )}

            {axesAvailable && (
              <Fold title="Axes" {...fold("axes")}>
                <Row label="Show">
                  <input type="checkbox" checked={stored.axesShow} onChange={(e) => patch({ axesShow: e.target.checked })} />
                </Row>
                <TitleRow label="X" title={xTitle} units={xUnits} disabled={!axesVisible}
                  onTitle={(v) => setOwn("xTitle", v)} onUnits={(v) => setOwn("xUnits", v)} />
                <TitleRow label="Y" title={yTitle} units={yUnits} disabled={!axesVisible}
                  onTitle={(v) => setOwn("yTitle", v)} onUnits={(v) => setOwn("yUnits", v)} />
                <TickRows prefix="axes" text={text} type={type} disabled={!axesVisible} />
              </Fold>
            )}

            {(cbLayer || axesAvailable) && (
              <Fold title="Margins" {...fold("margins")}>
                <Row label="Background">
                  <Select
                    value={stored.marginBackground}
                    onChange={(marginBackground) => patch({ marginBackground })}
                    disabled={!marginsShown}
                    options={[["white", "White"], ["black", "Black"], ["none", "None"]]}
                  />
                </Row>
                <Row label="Text">
                  <Select
                    value={stored.marginColour}
                    onChange={(marginColour) => patch({ marginColour })}
                    disabled={!marginsShown}
                    options={[["black", "Black"], ["white", "White"]]}
                  />
                </Row>
              </Fold>
            )}

            {grid && (
              <>
                <Fold title="Labels" {...fold("labels")}>
                <Row label="Show">
                  <input type="checkbox" checked={stored.labels} onChange={(e) => patch({ labels: e.target.checked })} />
                </Row>
                <Row label="Font">
                  <Num
                    value={text("labelFont")}
                    onChange={(v) => type("labelFont", v)}
                    disabled={!stored.labels}
                  />
                  <span className="muted">px</span>
                </Row>
                <Row label="Colour">
                  <Select
                    value={stored.labelColour}
                    onChange={(labelColour) => patch({ labelColour })}
                    disabled={!stored.labels}
                    options={[["white", "White"], ["black", "Black"]]}
                  />
                </Row>
                </Fold>
                <Fold title="Layout" {...fold("layout")}>
                <Row label="Spacing">
                  <Num value={text("spacing")} onChange={(v) => type("spacing", v)} min={0} />
                  <span className="muted">px</span>
                </Row>
                <Row label="Padding">
                  <Num value={text("padding")} onChange={(v) => type("padding", v)} min={0} />
                  <span className="muted">px</span>
                </Row>
                <Row label="Background">
                  <Select
                    value={stored.gridBackground}
                    onChange={(gridBackground) => patch({ gridBackground })}
                    options={[["dark", "Dark"], ["black", "Black"], ["white", "White"]]}
                  />
                </Row>
                </Fold>
              </>
            )}
          </div>
        </div>
        {error && <p className="field-error">{error}</p>}
        <div className="rebin-actions">
          <button className="secondary" onClick={onClose}>
            Cancel
          </button>
          <button className="primary" onClick={save} disabled={!canvas || busy}>
            Save
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}

/** Decimals typed: empty is automatic (null); NaN if not 0–10. */
function decimalsOf(t: string): number | null {
  if (t.trim() === "") return null;
  const n = Number(t);
  return Number.isInteger(n) && n >= 0 && n <= 10 ? n : NaN;
}

/** A section of the options: its heading folds it away. */
function Fold({ title, folded, onToggle, children }: {
  title: string;
  folded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <>
      <h3 className={folded ? "collapsed" : undefined}>
        <button type="button" className="process-group-toggle" aria-expanded={!folded} onClick={onToggle}>
          <span className="process-chevron" aria-hidden="true" />
          {title}
        </button>
      </h3>
      {!folded && children}
    </>
  );
}

/** A title and its units (written "title (units)"). */
function TitleRow({ label, title, units, disabled, onTitle, onUnits }: {
  label: string;
  title: string;
  units: string;
  disabled: boolean;
  onTitle: (v: string) => void;
  onUnits: (v: string) => void;
}) {
  return (
    <Row label={label}>
      <input type="text" className="png-title" value={title} disabled={disabled} placeholder="Title"
        onChange={(e) => onTitle(e.target.value)} />
      <input type="text" className="png-units" value={units} disabled={disabled} placeholder="Units"
        onChange={(e) => onUnits(e.target.value)} />
    </Row>
  );
}

/** A scale's label font, tick count and decimals. */
function TickRows({ prefix, text, type, disabled }: {
  prefix: "cb" | "axes";
  text: (k: keyof Typed) => string;
  type: (k: keyof Typed, v: string) => void;
  disabled: boolean;
}) {
  const k = (s: "Font" | "Ticks" | "Decimals") => `${prefix}${s}` as keyof Typed;
  return (
    <>
      <Row label="Font">
        <Num value={text(k("Font"))} onChange={(v) => type(k("Font"), v)} disabled={disabled} />
        <span className="muted">px</span>
      </Row>
      <Row label="Ticks">
        <Num value={text(k("Ticks"))} onChange={(v) => type(k("Ticks"), v)} disabled={disabled} min={2} />
      </Row>
      <Row label="Decimals">
        <Num value={text(k("Decimals"))} onChange={(v) => type(k("Decimals"), v)} disabled={disabled} placeholder="auto" />
      </Row>
    </>
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

function Num({
  value, onChange, disabled, min = 0, title, placeholder,
}: { value: string; onChange: (v: string) => void; disabled?: boolean; min?: number; title?: string; placeholder?: string }) {
  return (
    <input
      type="number"
      min={min}
      step="any"
      value={value}
      disabled={disabled}
      title={title}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

function Select<T extends string>({
  value, onChange, options, disabled,
}: { value: T; onChange: (v: T) => void; options: [T, string][]; disabled?: boolean }) {
  return (
    <select value={value} disabled={disabled} onChange={(e) => onChange(e.target.value as T)}>
      {options.map(([v, label]) => (
        <option key={v} value={v}>
          {label}
        </option>
      ))}
    </select>
  );
}
