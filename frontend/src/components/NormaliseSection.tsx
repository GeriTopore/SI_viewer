import { useEffect, useMemo } from "react";
import type { DatasetSummary } from "../api/client";
import type { EnergyRange, PlotWindow } from "./SpectrumPlot";
import { energyRange } from "../state/background";
import { normText, useNorm, useNormSettings, type NormMode } from "../state/normalise";
import { snapToAxis } from "../state/pca";
import { EnergyInput } from "./BackgroundSection";

const MODES: { id: NormMode; label: string; title: string }[] = [
  { id: "range", label: "0–1", title: "Each spectrum scaled from its minimum (0) to its maximum (1)" },
  { id: "value", label: "Value", title: "Each spectrum divided by its value at an energy" },
  { id: "mean", label: "Mean", title: "Each spectrum divided by its mean over an energy window" },
];

/** Process tab, spectrum images: how the spectrum plot normalises its
 * spectra (shown normalised with the plot's toggle, or "Normalised" here).
 * The energy or window is drawn on the spectrum, and dragged there. */
export function NormaliseSection({ d }: { d: DatasetSummary }) {
  const s = useNormSettings(d);
  const set = (patch: Parameters<ReturnType<typeof useNorm.getState>["set"]>[1]) => useNorm.getState().set(d, patch);
  const range = energyRange(d);
  // The energy or window is drawn while this section is open.
  useEffect(() => {
    useNorm.getState().set(d, { shown: true });
    return () => useNorm.getState().set(d, { shown: false });
  }, [d.id]); // eslint-disable-line react-hooks/exhaustive-deps
  const snap = (v: number) => snapToAxis(d, v);
  return (
    <section className="q-form">
      <label title="How each spectrum is normalised">
        <span>To</span>
        <div className="segmented" role="group" aria-label="Normalise to">
          {MODES.map((m) => (
            <button
              key={m.id}
              className={s.mode === m.id ? "active" : undefined}
              aria-pressed={s.mode === m.id}
              title={m.title}
              onClick={() => set({ mode: m.id, on: true })}
            >
              {m.label}
            </button>
          ))}
        </div>
      </label>
      {s.mode === "value" && (
        <label title="Energy whose value each spectrum is divided by">
          <span>Energy</span>
          <span className="q-inline bg-range">
            <EnergyInput value={s.lo ?? 0} step={range?.step ?? 1} label="Energy" onCommit={(v) => set({ lo: snap(v) })} />
            {range?.units}
          </span>
        </label>
      )}
      {s.mode === "mean" && (
        <label title="Window whose mean each spectrum is divided by">
          <span>
            <i className="bg-swatch" style={{ background: NORM_COLOUR }} />
            Window
          </span>
          <span className="q-inline bg-range">
            <EnergyInput value={s.lo ?? 0} step={range?.step ?? 1} label="Window from" onCommit={(v) => set({ lo: snap(v) })} />
            –
            <EnergyInput value={s.hi ?? 0} step={range?.step ?? 1} label="Window to" onCommit={(v) => set({ hi: snap(v) })} />
            {range?.units}
          </span>
        </label>
      )}
      <label className="q-check" title="Show the normalised spectra">
        <input type="checkbox" checked={s.on} onChange={(e) => set({ on: e.target.checked })} />
        Normalised ({normText(s, range?.units)})
      </label>
    </section>
  );
}

export const NORM_COLOUR = "#a855f7";
const NORM_BAND = "norm:energy";

/** The normalisation's energy (a line) or window (a band) on a spectrum
 * plot while its Process section is open; dragged to change it. */
export function useNormBand(d: DatasetSummary): {
  bands: PlotWindow[];
  onEdit: (id: string, range: EnergyRange) => boolean;
} {
  const s = useNormSettings(d);
  const bands = useMemo<PlotWindow[]>(() => {
    if (!s.shown || s.lo === null || s.mode === "range") return [];
    return s.mode === "value"
      ? [{ id: NORM_BAND, kind: "threshold", lo: s.lo, hi: s.lo, color: NORM_COLOUR, label: "Normalise" }]
      : [{ id: NORM_BAND, kind: "signal", lo: s.lo, hi: s.hi ?? s.lo, color: NORM_COLOUR, label: "Normalise" }];
  }, [s.shown, s.mode, s.lo, s.hi]);
  const onEdit = (id: string, r: EnergyRange) => {
    if (id !== NORM_BAND) return false;
    useNorm.getState().set(d, s.mode === "value" ? { lo: snapToAxis(d, r.lo) } : { lo: snapToAxis(d, r.lo), hi: snapToAxis(d, r.hi) });
    return true;
  };
  return { bands, onEdit };
}

/** The spectrum plot's checkbox: the spectra as they are, or normalised as
 * chosen in the Process tab (the same setting as its "Normalised"). */
export function NormaliseToggle({ d }: { d: DatasetSummary }) {
  const s = useNormSettings(d);
  const units = energyRange(d)?.units;
  return (
    <label className="field" title={`Normalise each spectrum: ${normText(s, units)}`}>
      <input type="checkbox" checked={s.on} onChange={(e) => useNorm.getState().set(d, { on: e.target.checked })} />
      Normalised
    </label>
  );
}
