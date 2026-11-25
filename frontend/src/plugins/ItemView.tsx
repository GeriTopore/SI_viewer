import type { DatasetSummary } from "../api/client";
import { usePlugins } from "./host";

/** A plugin's tree item in the middle column: its tab, or, without the
 * plugin, what it needs. */
export function PluginItemView({ d }: { d: DatasetSummary }) {
  const view = usePlugins((s) => s.itemViews.find((v) => v.plugin === d.plugin?.id && v.item === d.plugin?.item));
  if (!view) {
    return (
      <div className="placeholder">
        <h1>{d.title}</h1>
        <p className="muted">Needs the {d.plugin?.id ?? "missing"} plugin</p>
      </div>
    );
  }
  return (
    <div className="item-view">
      <div className="viewer-tabs" role="tablist">
        <button role="tab" aria-selected className="active">
          {view.label}
        </button>
      </div>
      <div className="viewer-tab-body">{view.render(d)}</div>
    </div>
  );
}
