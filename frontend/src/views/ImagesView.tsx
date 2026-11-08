import type { DatasetSummary } from "../api/client";
import { LayeredViewer } from "../components/LayeredViewer";
import { planeSize } from "../state/datasets";
import { imageGeometry } from "./geometry";

/** Plain images (or maps), merged into one composite when they are the same
 * size, or side by side in a grid. */
export function ImagesView({ images }: { images: DatasetSummary[] }) {
  const [w, h] = planeSize(images[0]);
  return (
    <div className="view">
      <LayeredViewer
        // Shared by all images of the first image's size, so zoom survives adding/removing images.
        viewKey={`images:${w}x${h}`}
        base={null}
        images={images}
        geometry={imageGeometry(images[0], "signal")}
        gridContext="images"
      />
    </div>
  );
}
