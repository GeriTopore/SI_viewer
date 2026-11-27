import { useEffect } from "react";
import type { Frame } from "../api/arrays";
import { DEFAULT_COLORMAP, useDisplay } from "../state/display";
import { autoRange, sortedSample } from "../viewer/contrast";
import type { DisplayParams } from "../viewer/renderer";

// The data version each display key was last auto-contrasted for.
const contrastedFor = new Map<string, string>();

/** Auto contrast for ``key`` from ``frame``: the first time, and again when
 * the data's ``version`` changes (e.g. a filter was applied), keeping the
 * colormap and gamma. */
export function autoContrast(key: string, frame: Frame, version = "", colormap?: () => string) {
  const { display, initDisplay, setDisplay } = useDisplay.getState();
  const seen = contrastedFor.get(key);
  contrastedFor.set(key, version);
  if (display[key] && (seen === undefined || seen === version)) return;
  const cmap = display[key]?.colormap ?? colormap?.() ?? DEFAULT_COLORMAP;
  const sorted = sortedSample(frame.data);
  // A direction drawn with its magnitude as brightness (DPC): the whole
  // circle. A phase (cmocean's phase map): 0–2π, as kemstem plots it.
  // Diverging maps (strain): symmetric about 0.
  let [lo, hi] = frame.shade ? [-180, 180] : cmap === "phase" ? [0, 2 * Math.PI] : autoRange(sorted);
  // Colours darkened by a value (an orientation map's IPF images, by the
  // correlation): from 0, so the shading is a fraction of the best.
  if (frame.rgb) lo = 0;
  if (cmap === "rdbu" && !frame.shade) {
    const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
    [lo, hi] = [-m, m];
  }
  if (display[key]) setDisplay(key, display[key].symmetric ? { lo: -Math.max(Math.abs(lo), Math.abs(hi)), hi: Math.max(Math.abs(lo), Math.abs(hi)) } : { lo, hi });
  // Diverging maps start with their limits locked symmetric.
  else initDisplay(key, { lo, hi, gamma: 1, colormap: cmap, ...(cmap === "rdbu" && !frame.shade && { symmetric: true }) });
}

/** Display settings for ``key``, auto-contrasted from the first frame seen.
 *
 * ``frame`` must be the image that ``key`` describes at ``version`` (or null
 * while it loads). ``colormap`` gives the starting colormap (e.g. an element
 * colour).
 */
export function useImageDisplay(key: string, frame: Frame | null, colormap?: () => string, version?: string) {
  const display = useDisplay((s) => s.display[key]);
  const setDisplay = useDisplay((s) => s.setDisplay);

  useEffect(() => {
    if (frame) autoContrast(key, frame, version, colormap);
  }, [key, frame]); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    display,
    setDisplay: (patch: Partial<DisplayParams>) => setDisplay(key, patch),
  };
}
