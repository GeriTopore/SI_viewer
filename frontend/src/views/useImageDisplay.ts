import { useEffect } from "react";
import type { Frame } from "../api/arrays";
import { DEFAULT_COLORMAP, useDisplay } from "../state/display";
import { autoRange, sortedSample } from "../viewer/contrast";
import type { DisplayParams } from "../viewer/renderer";

/** Display settings for ``key``, auto-contrasted from the first frame seen. */
export function useImageDisplay(key: string, frame: Frame | null) {
  const display = useDisplay((s) => s.display[key]);
  const setDisplay = useDisplay((s) => s.setDisplay);

  useEffect(() => {
    if (!frame || useDisplay.getState().display[key]) return;
    const [lo, hi] = autoRange(sortedSample(frame.data));
    useDisplay.getState().initDisplay(key, { lo, hi, gamma: 1, colormap: DEFAULT_COLORMAP });
  }, [key, frame]);

  return {
    display,
    setDisplay: (patch: Partial<DisplayParams>) => setDisplay(key, patch),
  };
}
