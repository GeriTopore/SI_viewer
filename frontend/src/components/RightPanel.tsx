import { useUi, type RightTab } from "../state/ui";
import { Inspector } from "./Inspector";
import { CollapseButton } from "./PaneToggle";
import { QuantifyPanel } from "./QuantifyPanel";

const TABS: { id: RightTab; label: string }[] = [
  { id: "info", label: "Info" },
  { id: "quantify", label: "Quantify" },
];

/** Right-hand column: Info (dataset details) and Quantify (fitting) tabs. */
export function RightPanel() {
  const { rightTab, setRightTab } = useUi();
  return (
    <>
      <div className="panel-tabs" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={rightTab === t.id}
            className={rightTab === t.id ? "active" : undefined}
            onClick={() => setRightTab(t.id)}
          >
            {t.label}
          </button>
        ))}
        <span className="panel-tabs-spacer" />
        <CollapseButton side="right" label="Info / Quantify" />
      </div>
      <div className="panel-tab-body">{rightTab === "info" ? <Inspector /> : <QuantifyPanel />}</div>
    </>
  );
}
