import type { ReactNode } from "react";
import type { DatasetSummary } from "../api/client";
import { LayeredViewer } from "../components/LayeredViewer";
import { planeSize } from "../state/datasets";
import { imageGeometry } from "./geometry";

/** Plain images (or maps), merged into one composite when they are the same
 * size, or side by side in a grid; with column items (kemstem) drawn over
 * their image (``implied``: images shown only under items). ``footer``:
 * under the image (e.g. the Bragg vector map's Raw / Calibrated switch). */
export function ImagesView({ images, items = [], implied = [], footer }: {
  images: DatasetSummary[];
  items?: DatasetSummary[];
  implied?: string[];
  footer?: ReactNode;
}) {
  const [w, h] = planeSize(images[0]);
  return (
    <div className="view">
      <LayeredViewer
        // Shared by all images of the first image's size, so zoom survives adding/removing images.
        viewKey={`images:${w}x${h}`}
        base={null}
        images={images}
        items={items}
        implied={implied}
        geometry={imageGeometry(images[0], "signal")}
        gridContext="images"
        footer={footer}
      />
    </div>
  );
}
