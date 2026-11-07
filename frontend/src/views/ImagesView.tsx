import type { DatasetSummary } from "../api/client";
import { LayeredViewer } from "../components/LayeredViewer";
import { planeSize } from "../state/datasets";
import { imageGeometry } from "./geometry";

/** An image (or map) on its own: overlays are only made on spectrum images. */
export function ImagesView({ images }: { images: DatasetSummary[] }) {
  const [w, h] = planeSize(images[0]);
  return (
    <div className="view">
      <LayeredViewer
        // Shared by all images of this size, so zoom survives adding/removing maps.
        viewKey={`images:${w}x${h}`}
        base={null}
        images={images}
        geometry={imageGeometry(images[0], "signal")}
        overlays={false}
      />
    </div>
  );
}
