import { useEffect, useRef, useState } from "react";
import { Coalescer } from "../../api/coalesce";
import { fetchPixelSpectrum, fetchRegionSpectrum, type PixelsSpectrum } from "../../api/arrays";
import { shapeToRegion, type SiState } from "../../state/si";
import { useLayers } from "../../state/layers";

export const CURSOR_CHANNEL = "cursor";

/** Spectra for the cursor and each visible region of a spectrum image,
 * refetched (latest-only) whenever their geometry changes. */
export function useSiSpectra(si: string, state: SiState, cursorSize: number, ready: boolean) {
  const [spectra, setSpectra] = useState<Record<string, PixelsSpectrum>>({});
  const coalescer = useRef<Coalescer<PixelsSpectrum>>();
  if (!coalescer.current) {
    coalescer.current = new Coalescer(
      (channel, value) => setSpectra((prev) => ({ ...prev, [channel]: value })),
      (_channel, e) => useLayers.getState().notify(`Spectrum failed: ${e.message}`),
    );
  }

  const { cursor, showCursor, rois } = state;
  useEffect(() => {
    if (!ready) return;
    const c = coalescer.current!;
    if (cursor && showCursor) {
      const { x, y } = cursor;
      c.request(CURSOR_CHANNEL, `${x},${y},${cursorSize}`, () => fetchPixelSpectrum(si, x, y, cursorSize));
    }
    for (const roi of rois) {
      if (!roi.visible) continue;
      const region = shapeToRegion(roi.shape);
      c.request(roi.id, JSON.stringify(region), () => fetchRegionSpectrum(si, region));
    }
  }, [si, ready, cursor, showCursor, cursorSize, rois]);

  // Drop results for regions that no longer exist.
  useEffect(() => {
    const alive = new Set([CURSOR_CHANNEL, ...rois.map((r) => r.id)]);
    setSpectra((prev) => {
      const stale = Object.keys(prev).filter((k) => !alive.has(k));
      if (!stale.length) return prev;
      stale.forEach((k) => coalescer.current!.forget(k));
      return Object.fromEntries(Object.entries(prev).filter(([k]) => alive.has(k)));
    });
  }, [rois]);

  return spectra;
}
