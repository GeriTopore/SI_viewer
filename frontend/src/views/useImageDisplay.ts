import { useEffect } from "react";
import type { Frame } from "../api/arrays";
import { DEFAULT_COLORMAP, useDisplay } from "../state/display";
import { autoRange, sortedSample } from "../viewer/contrast";
import type { DisplayParams } from "../viewer/renderer";

/** Display settings for ``key``, auto-contrasted from the first frame seen.
 *
 * ``frame`` must be the image that ``key`` describes (or null while it
 * loads). ``colormap`` gives the starting colormap (e.g. an element colour).
 */
export function useImageDisplay(key: string, frame: Frame | null, colormap?: () => string) {
  const display = useDisplay((s) => s.display[key]);
  const setDisplay = useDisplay((s) => s.setDisplay);

  useEffect(() => {
    if (!frame || useDisplay.getState().display[key]) return;
    const [lo, hi] = autoRange(sortedSample(frame.data));
    const cm = colormap?.() ?? DEFAULT_COLORMAP;
    useDisplay.getState().initDisplay(key, { lo, hi, gamma: 1, colormap: cm });
  }, [key, frame]); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    display,
    setDisplay: (patch: Partial<DisplayParams>) => setDisplay(key, patch),
  };
}
