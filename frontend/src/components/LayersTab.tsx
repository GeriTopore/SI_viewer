import { useLayersTab } from "../state/layersTab";
import { LayersPanel } from "./LayersPanel";

/** Layers tab of the right column: the main viewer's layers (visibility,
 * colour, opacity, remove; click one to edit its contrast), as in the panel
 * over the image, which shows only once maps are added. */
export function LayersTab() {
  const panel = useLayersTab((s) => s.panel);
  if (!panel?.layers.length) return <p className="muted">No image in the viewer.</p>;
  return <LayersPanel {...panel} docked />;
}
