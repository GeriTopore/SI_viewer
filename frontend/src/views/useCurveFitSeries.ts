import { createElement, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { dataVersion, type DatasetSummary, type FitSource } from "../api/client";
import type { PlotHandle, PlotWindow, SpectrumSeries } from "../components/SpectrumPlot";
import { ContextMenu, type MenuItem } from "../components/ContextMenu";
import { isLineScan } from "../format";
import {
  drawModel, duplicateComponent, fetchPixelFit, showsPixelFit, estimateComponent, modeOf, paramRole, placePeak, removeComponent,
  setAllFree, setParamLimits, updateComponent, useCurveFit, type ModelComponent,
} from "../state/curveFit";
import { shapeToRegion, useSi, useSiState } from "../state/si";
import { dragParams, handlesOf, type DragStart } from "./curveHandles";
import { CURVE_BG_COLOUR } from "../components/CurveFitPanel";
import { snapToAxis } from "../state/pca";
import { FIT_COLOR } from "./useFitSeries";

/** The spectrum a choice stands for ("all", "cursor" or a region's id), or null. */
export function useCurveSource(ds: DatasetSummary, choice: string): FitSource | null {
  const si = useSiState(ds.id);
  const cursorSize = useSi((s) => s.cursorSize);
  if (ds.kind !== "spectrum_image" || choice === "all") return { type: "all" };
  if (choice === "cursor") {
    if (!si.cursor) return null;
    return { type: "pixel", x: si.cursor.x, y: si.cursor.y, size: isLineScan(ds) ? 1 : cursorSize };
  }
  const roi = si.rois.find((r) => r.id === choice);
  return roi ? { type: "region", region: shapeToRegion(roi.shape) } : null;
}

export interface CurvePlotProps {
  handles?: PlotHandle[];
  onHandleDrag?: (id: string, x: number, y: number, final: boolean) => void;
  placing?: boolean;
  onPlace?: (p: { x: number; y: number } | null) => void;
  pickable?: string[];
  onPick?: (label: string | null) => void;
  onSeriesMenu?: (label: string, x: number, y: number) => boolean;
  menu?: ReactNode;
}

/** The fit of every pixel at the cursor while the plot shows it (see
 * ``showsPixelFit``), else null. */
export function usePixelCurveFit(ds: DatasetSummary, source: FitSource | null) {
  const st = useCurveFit((s) => s.byId[ds.id]);
  const shows = showsPixelFit(ds, st, source) && source?.type === "pixel";
  const at = shows && source?.type === "pixel" ? `${ds.id}|${dataVersion(ds)}|${source.x}|${source.y}|${source.size}` : null;
  useEffect(() => {
    if (at && source?.type === "pixel") void fetchPixelFit(ds, at, source.x, source.y, source.size);
  }, [at]); // eslint-disable-line react-hooks/exhaustive-deps
  // The last one stays while the next is fetched.
  return shows ? (st?.pixelFit?.result ?? null) : null;
}

/** Curve fitting (Process tab) on a spectrum plot, while its section is open: the model and each active
 * component (per pixel with ``perPixel``, to match spectra plotted as means),
 * and the interaction: click a component's curve to select it, drag its handles,
 * click to place a peak, right-click a curve for its menu. ``plot`` goes to
 * ``SpectrumPlot``. */
export function useCurveFitSeries(ds: DatasetSummary, x: Float64Array, perPixel = false): {
  series: SpectrumSeries[];
  /** The selected parameter's limits on the energy axis, and their edit. */
  bands: PlotWindow[];
  onEdit: (id: string, range: { lo: number; hi: number }) => boolean;
  plot: CurvePlotProps;
} {
  const st = useCurveFit((s) => s.byId[ds.id]);
  const shown = useCurveFit((s) => s.shown === ds.id);
  const cat = useCurveFit((s) => s.catalogue);
  const source = useCurveSource(ds, st?.source ?? "all");
  const pixel = usePixelCurveFit(ds, source);
  const result = pixel ?? st?.result;
  const on = shown && !!st;
  const scale = result && perPixel ? 1 / Math.max(1, result.pixels) : 1;
  const selected = st?.selected ?? null;

  // Each active component's curve as plotted.
  const curves = useMemo(() => {
    const out = new Map<string, number[]>();
    if (!on || !result) return out;
    for (const c of result.components) if (c.curve) out.set(c.name, c.curve.map((v) => v * scale));
    return out;
  }, [on, result, scale]);

  const series = useMemo<SpectrumSeries[]>(() => {
    if (!on || !result || !st.components.length || x.length < 2) return [];
    const i0 = Math.round((result.x[0] - x[0]) / (x[1] - x[0]));
    const place = (curve: number[]) => {
      const out: (number | null)[] = new Array(x.length).fill(null);
      curve.forEach((v, j) => {
        const i = i0 + j;
        if (i >= 0 && i < out.length) out[i] = v;
      });
      return out;
    };
    const out: SpectrumSeries[] = [];
    for (const c of st.components) {
      const curve = curves.get(c.name);
      if (!curve) continue;
      const sel = c.name === selected;
      // The selected component bold; the others faded while one is.
      const color = selected && !sel ? `${c.color}88` : c.color;
      out.push({ label: c.name, data: place(curve), color, dash: sel ? undefined : [4, 3], width: sel ? 2.2 : 1.25, fill: false });
    }
    out.push({ label: result.fitted ? "Fit" : "Model", data: place(result.model.map((v) => v * scale)), color: FIT_COLOR, width: 1.6, fill: false });
    return out;
  }, [on, result, st?.components, curves, selected, x, scale]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---------- the selected component's handles ----------
  const comp: ModelComponent | undefined = st?.components.find((c) => c.name === selected && c.active);
  const [dragAt, setDragAt] = useState<{ id: string; x: number; y: number } | null>(null);
  const drag = useRef<DragStart | null>(null);
  // The component's handles where its curve puts them (before any drag).
  const base = useMemo<PlotHandle[]>(() => {
    if (!on || !comp || !result || !shown || pixel) return []; // the fit of every pixel: nothing to drag
    const curve = curves.get(comp.name);
    return curve ? handlesOf(comp, result.x, curve, result.range) : [];
  }, [on, comp, result, curves, shown]);

  // ---------- the selected parameter's limits ----------
  // Only a position (a peak's centre, a step's place) is drawn: a band on the
  // energy axis, dragged by its edges, while the parameter is limited.
  const param = comp && st?.selectedParam ? comp.params[st.selectedParam] : undefined;
  const limited = !!param && modeOf(param) === "limited";
  const role = comp && st?.selectedParam ? paramRole(comp.kind, st.selectedParam) : "other";
  const [xFirst, xLast] = [x[0], x[x.length - 1]];
  const LIMIT = "curve-fit:limits";
  const limitBands = useMemo<PlotWindow[]>(() => {
    if (!limited || role !== "position" || !comp || !param) return [];
    return [{
      id: LIMIT, kind: "signal", lo: param.bmin ?? xFirst, hi: param.bmax ?? xLast, color: comp.color,
      fillOpacity: 0.1, edgesOnly: true, label: `${comp.name} ${st!.selectedParam} limits`,
    }];
  }, [limited, comp, param, role, xFirst, xLast]); // eslint-disable-line react-hooks/exhaustive-deps
  const onLimitEdit = (id: string, r: { lo: number; hi: number }) => {
    if (id !== LIMIT || !comp || !st?.selectedParam || role !== "position") return false;
    const step = x.length > 1 ? x[1] - x[0] : 1;
    const lo = r.lo <= xFirst + step / 2 ? null : r.lo;
    const hi = r.hi >= xLast - step / 2 ? null : r.hi;
    setParamLimits(ds, comp.name, st.selectedParam, lo, hi);
    if (source) void drawModel(ds, source);
    return true;
  };

  // The background window: hatched, dragged by its edges or inside.
  const BG = "curve-fit:background";
  const bgBands = useMemo<PlotWindow[]>(
    () => (on && st.bgOn && st.bgLo !== null && st.bgHi !== null
      ? [{ id: BG, kind: "background", lo: st.bgLo, hi: st.bgHi, color: CURVE_BG_COLOUR, label: "Background" }]
      : []),
    [on, st?.bgOn, st?.bgLo, st?.bgHi], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const handles = useMemo<PlotHandle[]>(() => {
    const hs = base;
    return dragAt ? hs.map((h) => (h.id === dragAt.id ? { ...h, x: dragAt.x, y: dragAt.y } : h)) : hs;
  }, [base, dragAt]);

  const onHandleDrag = (id: string, hx: number, hy: number, final: boolean) => {
    if (!comp || !result || !source) return;
    if (!drag.current) {
      drag.current = {
        params: structuredClone(comp.params),
        handles: Object.fromEntries(handles.map((h) => [h.id, { x: h.x, y: h.y }])),
      };
    }
    const params = dragParams(comp, id, hx, hy, drag.current, result.factor * scale);
    // Limited parameters stay within their limits.
    for (const p of Object.values(params)) {
      if (!p.free) continue;
      if (p.bmin !== null) p.value = Math.max(p.value, p.bmin);
      if (p.bmax !== null) p.value = Math.min(p.value, p.bmax);
    }
    updateComponent(ds, comp.name, { params });
    void drawModel(ds, source);
    if (final) {
      drag.current = null;
      setDragAt(null);
    } else {
      const h = handles.find((x) => x.id === id);
      setDragAt({ id, x: h?.axis === "y" ? h.x : hx, y: h?.axis === "x" ? h.y : hy });
    }
  };

  // ---------- the right-click menu of a curve ----------
  const [menu, setMenu] = useState<{ name: string; x: number; y: number } | null>(null);
  const menuComp = menu ? st?.components.find((c) => c.name === menu.name) : undefined;
  const items: MenuItem[] = menuComp
    ? [
        ...(cat?.components.find((k) => k.kind === menuComp.kind)?.estimate && source
          ? [{ label: "Estimate", title: "Starting values from the spectrum", onSelect: () => void estimateComponent(ds, menuComp.name, source) }]
          : []),
        { label: "Duplicate", onSelect: () => duplicateComponent(ds, menuComp.name, source) },
        { label: "Fix all", onSelect: () => setAllFree(ds, menuComp.name, false) },
        { label: "Free all", onSelect: () => setAllFree(ds, menuComp.name, true) },
        { label: `Delete ${menuComp.name}`, danger: true, onSelect: () => removeComponent(ds, menuComp.name) },
      ]
    : [];

  const names = on ? st.components.filter((c) => curves.has(c.name)).map((c) => c.name) : [];
  const plot: CurvePlotProps = on
    ? {
        handles,
        onHandleDrag,
        placing: !!st.placing && !!source,
        onPlace: (p) => {
          const kind = useCurveFit.getState().byId[ds.id]?.placing;
          if (!p || !kind || !source) useCurveFit.getState().patch(ds, { placing: null });
          else void placePeak(ds, kind, p, source, perPixel);
        },
        pickable: names,
        onPick: (label) => useCurveFit.getState().patch(ds, { selected: label }),
        onSeriesMenu: (label, mx, my) => {
          setMenu({ name: label, x: mx, y: my });
          useCurveFit.getState().patch(ds, { selected: label });
          return true;
        },
        menu: menu && items.length ? createElement(ContextMenu, { x: menu.x, y: menu.y, items, onClose: () => setMenu(null) }) : null,
      }
    : {};
  const bands = useMemo(() => [...bgBands, ...limitBands], [bgBands, limitBands]);
  const onEdit = (id: string, r: { lo: number; hi: number }) => {
    if (id === BG) {
      useCurveFit.getState().patch(ds, { bgLo: snapToAxis(ds, r.lo), bgHi: snapToAxis(ds, r.hi) });
      return true;
    }
    return onLimitEdit(id, r);
  };
  return { series, bands, onEdit, plot };
}
