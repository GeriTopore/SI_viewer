import { useEffect, useState } from "react";
import { fetchFrame, type Frame } from "../api/arrays";
import { useLayers } from "../state/layers";

// Image datasets don't change, so each is fetched once.
const cache = new Map<string, Promise<Frame>>();

/** Frames of the given image datasets, filled in as they arrive. */
export function useOverlayFrames(ids: string[]): Record<string, Frame> {
  const [frames, setFrames] = useState<Record<string, Frame>>({});
  const key = ids.join(",");

  useEffect(() => {
    let live = true;
    for (const id of ids) {
      if (frames[id]) continue;
      let p = cache.get(id);
      if (!p) {
        p = fetchFrame(id);
        cache.set(id, p);
      }
      p.then((f) => live && setFrames((prev) => (prev[id] ? prev : { ...prev, [id]: f }))).catch((e) => {
        cache.delete(id);
        useLayers.getState().notify(`Could not load image: ${(e as Error).message}`);
      });
    }
    return () => {
      live = false;
    };
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps

  return frames;
}
