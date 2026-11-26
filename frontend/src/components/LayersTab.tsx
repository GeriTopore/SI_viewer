import { SECTION_ORDER, useLayersTab } from "../state/layersTab";
import { LayersPanel } from "./LayersPanel";

/** Layers tab of the right column: the main viewer's layers (visibility,
 * colour, opacity, remove; click one to edit its contrast), as in the panel
 * over the image, which shows only once maps are added. A view of several
 * panes (4D-STEM) has a section per pane. */
export function LayersTab() {
  const panel = useLayersTab((s) => s.panel);
  const sections = useLayersTab((s) => s.sections);
  const titles = Object.keys(sections).sort(
    (a, b) => (SECTION_ORDER.indexOf(a) + 1 || 99) - (SECTION_ORDER.indexOf(b) + 1 || 99),
  );
  if (titles.length) {
    return (
      <>
        {titles.map((t) => (
          <section key={t} className="layers-section">
            <h4 className="process-heading">{t}</h4>
            {sections[t].panel.layers.length ? <LayersPanel {...sections[t].panel} docked /> : <p className="muted small">Nothing shown.</p>}
          </section>
        ))}
      </>
    );
  }
  if (!panel?.layers.length) return <p className="muted">No image in the viewer.</p>;
  return <LayersPanel {...panel} docked />;
}
