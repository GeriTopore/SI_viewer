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
  const [lo, hi] = autoRange(sortedSample(frame.data));
  if (display[key]) setDisplay(key, { lo, hi });
  else initDisplay(key, { lo, hi, gamma: 1, colormap: colormap?.() ?? DEFAULT_COLORMAP });
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
