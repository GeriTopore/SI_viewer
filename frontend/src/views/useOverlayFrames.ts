import { useEffect, useState } from "react";
import { fetchDpcFrame, fetchFrame, fetchIpfFrame, type Frame } from "../api/arrays";
import { contrastVersion, dataVersion } from "../api/client";
import { useLayers } from "../state/layers";

// Image data only changes when a map is recomputed or filtered (new
// version), so each (dataset, version) is fetched once.
const cache = new Map<string, Promise<Frame>>();
const keyOf = (d: ImageRef) => `${d.id}@${dataVersion(d)}`;

export interface ImageRef {
  id: string;
  revision: number;
  filter_rev: number;
  data_rev?: number;
  /** DPC images: the colour one is drawn with its magnitude as brightness. */
  dpc?: { output: string } | null;
  /** Orientation maps' IPF images: drawn in their colours. */
  orientation?: { output: string } | null;
}

/** Frames of the given image datasets (by id), filled in as they arrive. */
export function useOverlayFrames(images: ImageRef[]): Record<string, Frame> {
  const [frames, setFrames] = useState<Record<string, { key: string; contrast: string; frame: Frame }>>({});
  const key = images.map(keyOf).join(",");

  useEffect(() => {
    let live = true;
    for (const d of images) {
      const k = keyOf(d);
      if (frames[d.id]?.key === k) continue;
      let p = cache.get(k);
      if (!p) {
        p = d.dpc?.output === "colour" ? fetchDpcFrame(d.id) : d.orientation?.output === "ipf" ? fetchIpfFrame(d.id) : fetchFrame(d.id);
        cache.set(k, p);
      }
      const contrast = contrastVersion(d);
      p.then((frame) => live && setFrames((prev) => ({ ...prev, [d.id]: { key: k, contrast, frame } }))).catch((e) => {
        cache.delete(k);
        useLayers.getState().notify(`Could not load image: ${(e as Error).message}`);
      });
    }
    return () => {
      live = false;
    };
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps

  // Only frames of the current versions; the next frame of a series (same
  // contrast version) is waited for with the last one still shown.
  const current: Record<string, Frame> = {};
  for (const d of images) {
    const f = frames[d.id];
    if (f && (f.key === keyOf(d) || f.contrast === contrastVersion(d))) current[d.id] = f.frame;
  }
  return current;
}
