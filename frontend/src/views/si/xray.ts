import { useEffect, useMemo, useState } from "react";
import { api, type DatasetSummary } from "../../api/client";
import type { Marker } from "../../components/SpectrumPlot";
import { useDatasets } from "../../state/datasets";
import { useDisplay } from "../../state/display";
import { useLayers } from "../../state/layers";

export interface XrayLine {
  line: string;
  energy: number;
  weight: number;
}

export interface ElementInfo {
  symbol: string;
  Z: number;
  name: string;
  lines: XrayLine[];
}

let tablePromise: Promise<ElementInfo[]> | null = null;

/** exSpy's element/X-ray line table (fetched once). */
export function useElementTable(): ElementInfo[] {
  const [table, setTable] = useState<ElementInfo[]>([]);
  useEffect(() => {
    tablePromise ??= fetch("/api/elements").then((r) => r.json());
    tablePromise.then(setTable).catch(() => (tablePromise = null));
  }, []);
  return table;
}

const GREEK: Record<string, string> = { a: "α", b: "β", g: "γ", z: "ζ", n: "η", l: "ℓ" };
/** "Ka" → "Kα", "Lb3" → "Lβ3". */
export const lineLabel = (line: string) => line[0] + (GREEK[line[1]] ?? line[1] ?? "") + line.slice(2);

const MIN_WEIGHT = 0.05;
const MIN_ENERGY = 0.1; // keV
const DEFAULT_MARKER = "#e0b84a";

/** The line an element is analysed by when exSpy hasn't chosen one, using
 * exSpy's rule (add_lines): of the α lines inside the spectrum's energy range,
 * K before L before M, the first below half the beam energy (else the last). */
function defaultLine(lines: XrayLine[], beamEnergy: number | null, range: [number, number]) {
  const alpha = lines
    .filter((l) => l.line[1] === "a" && l.energy >= range[0] && l.energy <= range[1])
    .sort((a, b) => a.line.localeCompare(b.line));
  const limit = beamEnergy ? beamEnergy / 2 : range[1];
  return (alpha.find((l) => l.energy < limit) ?? alpha.at(-1))?.line;
}

/** Line markers for ``elements`` (plus a dashed preview of ``preview``).
 *
 * Only each element's analysis line is labelled: the one in ``xrayLines``
 * (exSpy's choice, e.g. "Fe_Ka"), else ``defaultLine``. Other lines are
 * drawn unlabelled, with heights from their relative weights.
 */
export function useLineMarkers(
  elements: string[],
  preview: string | null,
  beamEnergy: number | null,
  xrayLines: string[] = [],
  range: [number, number] = [0, Infinity],
) {
  const table = useElementTable();
  const colours = useDisplay((s) => s.elementColours);
  return useMemo(() => {
    const bySymbol = new Map(table.map((e) => [e.symbol, e]));
    const shown: [string, boolean][] = elements.map((e) => [e, false]);
    if (preview && !elements.includes(preview)) shown.push([preview, true]);
    const markers: Marker[] = [];
    for (const [symbol, dashed] of shown) {
      const lines = bySymbol.get(symbol)?.lines ?? [];
      const chosen = xrayLines.find((l) => l.startsWith(`${symbol}_`))?.split("_")[1] ?? defaultLine(lines, beamEnergy, range);
      for (const l of lines) {
        if (l.weight < MIN_WEIGHT || l.energy < MIN_ENERGY) continue;
        if (beamEnergy && l.energy > beamEnergy) continue;
        markers.push({
          energy: l.energy,
          weight: l.weight,
          color: colours[symbol] ?? DEFAULT_MARKER,
          label: l.line === chosen ? `${symbol} ${lineLabel(l.line)}` : undefined,
          dashed,
          element: symbol,
        });
      }
    }
    return markers;
  }, [table, colours, elements, preview, beamEnergy, xrayLines, range[0], range[1]]); // eslint-disable-line react-hooks/exhaustive-deps
}

/** Peak FWHM (keV) at ``energy`` from the detector's Mn Kα resolution (eV),
 * as in exSpy's ``get_FWHM_at_Energy``. */
export function fwhmAt(energy: number, resolutionMnKa = 130): number {
  return Math.sqrt(2.5 * (energy - 5.898) * 1000 + resolutionMnKa ** 2) / 1000;
}

/** Set a dataset's elements (optimistically, then from the backend's reply,
 * which includes the file: a spectrum image gains or loses computed maps). */
export async function setElements(ds: DatasetSummary, elements: string[]) {
  const { updateDataset, replaceFile } = useDatasets.getState();
  if (!ds.eds) return;
  for (const e of elements) useDisplay.getState().ensureElementColour(e);
  updateDataset({ ...ds, eds: { ...ds.eds, elements } });
  try {
    const { file, warnings } = await api.setElements(ds.id, elements);
    replaceFile(file);
    if (warnings?.length) useLayers.getState().notify(warnings.join(" · "));
  } catch (e) {
    updateDataset(ds);
    useLayers.getState().notify(`Could not set elements: ${(e as Error).message}`);
  }
}
