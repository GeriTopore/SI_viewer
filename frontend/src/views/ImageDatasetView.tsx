import { useEffect, useState } from "react";
import { fetchFrame, type Frame } from "../api/arrays";
import type { DatasetSummary } from "../api/client";
import { ContrastControls } from "../components/ContrastControls";
import { ImageView } from "../components/ImageView";
import { imageGeometry, isAbort } from "./geometry";
import { useImageDisplay } from "./useImageDisplay";

export function ImageDatasetView({ ds }: { ds: DatasetSummary }) {
  const [frame, setFrame] = useState<Frame | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { display, setDisplay } = useImageDisplay(ds.id, frame);
  const g = imageGeometry(ds, "signal");

  useEffect(() => {
    const ac = new AbortController();
    setFrame(null);
    setError(null);
    fetchFrame(ds.id, {}, ac.signal)
      .then(setFrame)
      .catch((e) => !isAbort(e) && setError((e as Error).message));
    return () => ac.abort();
  }, [ds.id]);

  return (
    <div className="view">
      <ContrastControls frame={frame} display={display} onChange={setDisplay} />
      <ImageView
        viewKey={ds.id}
        frame={frame}
        fullWidth={g.width}
        fullHeight={g.height}
        display={display}
        calX={g.calX}
        calY={g.calY}
        loading={frame ? null : "Loading image…"}
        error={error}
      />
    </div>
  );
}
