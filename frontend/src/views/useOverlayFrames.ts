import { useEffect, useState } from "react";
import { fetchFrame, type Frame } from "../api/arrays";
import { dataVersion } from "../api/client";
import { useLayers } from "../state/layers";

// Image data only changes when a map is recomputed or filtered (new
// version), so each (dataset, version) is fetched once.
const cache = new Map<string, Promise<Frame>>();
const keyOf = (d: ImageRef) => `${d.id}@${dataVersion(d)}`;

export interface ImageRef {
  id: string;
  revision: number;
  filter_rev: number;
}

/** Frames of the given image datasets (by id), filled in as they arrive. */
export function useOverlayFrames(images: ImageRef[]): Record<string, Frame> {
  const [frames, setFrames] = useState<Record<string, { key: string; frame: Frame }>>({});
  const key = images.map(keyOf).join(",");

  useEffect(() => {
    let live = true;
    for (const d of images) {
      const k = keyOf(d);
      if (frames[d.id]?.key === k) continue;
      let p = cache.get(k);
      if (!p) {
        p = fetchFrame(d.id);
        cache.set(k, p);
      }
      p.then((frame) => live && setFrames((prev) => ({ ...prev, [d.id]: { key: k, frame } }))).catch((e) => {
        cache.delete(k);
        useLayers.getState().notify(`Could not load image: ${(e as Error).message}`);
      });
    }
    return () => {
      live = false;
    };
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps

  // Only frames of the current versions.
  const current: Record<string, Frame> = {};
  for (const d of images) if (frames[d.id]?.key === keyOf(d)) current[d.id] = frames[d.id].frame;
  return current;
}
