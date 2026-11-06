import type { DatasetSummary } from "../api/client";
import { LayeredViewer } from "../components/LayeredViewer";
import { planeSize } from "../state/datasets";
import { imageGeometry } from "./geometry";

/** One or more same-size images, composed additively in their colours. */
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
      />
    </div>
  );
}
