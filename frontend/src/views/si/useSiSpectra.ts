import { useEffect, useRef, useState } from "react";
import { Coalescer } from "../../api/coalesce";
import { fetchPixelSpectrum, fetchRegionSpectrum, type PixelsSpectrum } from "../../api/arrays";
import { shapeToRegion, type SiState } from "../../state/si";
import { useLayers } from "../../state/layers";

export const CURSOR_CHANNEL = "cursor";
export const CURSOR2_CHANNEL = "cursor2";
export const LINE_CHANNEL = "line";

/** Part of a profile line whose spectrum is shown: the band's polygon, and
 * the pixel to use if the band holds no pixel centre. */
export interface LineBand {
  points: [number, number][];
  pixel: [number, number];
}

/** Spectra for the cursor, each visible region and a picked part of a
 * profile line of a spectrum image, refetched (latest-only) whenever their
 * geometry or the data ``version`` (e.g. its filters) changes. */
export function useSiSpectra(
  si: string, version: string, state: SiState, cursorSize: number, ready: boolean, line: LineBand | null = null,
) {
  const [spectra, setSpectra] = useState<Record<string, PixelsSpectrum>>({});
  const coalescer = useRef<Coalescer<PixelsSpectrum>>();
  if (!coalescer.current) {
    coalescer.current = new Coalescer(
      (channel, value) => setSpectra((prev) => ({ ...prev, [channel]: value })),
      (_channel, e) => useLayers.getState().notify(`Spectrum failed: ${e.message}`),
    );
  }

  const { cursor, showCursor, cursor2, showCursor2, rois } = state;
  useEffect(() => {
    if (!ready) return;
    const c = coalescer.current!;
    if (cursor && showCursor) {
      const { x, y } = cursor;
      c.request(CURSOR_CHANNEL, `${x},${y},${cursorSize}@${version}`, () => fetchPixelSpectrum(si, x, y, cursorSize));
    }
    if (cursor2 && showCursor2) {
      const { x, y } = cursor2;
      c.request(CURSOR2_CHANNEL, `${x},${y},${cursorSize}@${version}`, () => fetchPixelSpectrum(si, x, y, cursorSize));
    }
    for (const roi of rois) {
      if (!roi.visible) continue;
      const region = shapeToRegion(roi.shape);
      c.request(roi.id, `${JSON.stringify(region)}@${version}`, () => fetchRegionSpectrum(si, region));
    }
  }, [si, version, ready, cursor, showCursor, cursor2, showCursor2, cursorSize, rois]);

  const lineKey = line && JSON.stringify(line);
  useEffect(() => {
    if (!ready || !line) return;
    const region = { kind: "polygon", points: line.points };
    coalescer.current!.request(LINE_CHANNEL, `${lineKey}@${version}`, async () => {
      const r = await fetchRegionSpectrum(si, region);
      return r.pixels > 0 ? r : fetchPixelSpectrum(si, line.pixel[0], line.pixel[1], 1);
    });
  }, [si, version, ready, lineKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // Drop results for regions (or a line pick) that no longer exist.
  useEffect(() => {
    const alive = new Set([CURSOR_CHANNEL, ...(cursor2 ? [CURSOR2_CHANNEL] : []), ...rois.map((r) => r.id), ...(line ? [LINE_CHANNEL] : [])]);
    setSpectra((prev) => {
      const stale = Object.keys(prev).filter((k) => !alive.has(k));
      if (!stale.length) return prev;
      stale.forEach((k) => coalescer.current!.forget(k));
      return Object.fromEntries(Object.entries(prev).filter(([k]) => alive.has(k)));
    });
  }, [rois, !line]); // eslint-disable-line react-hooks/exhaustive-deps

  return spectra;
}
