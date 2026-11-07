import { useUi, type RightTab } from "../state/ui";
import { FftPanel } from "./FftPanel";
import { Inspector } from "./Inspector";
import { CollapseButton } from "./PaneToggle";
import { ProcessPanel } from "./ProcessPanel";
import { QuantifyPanel } from "./QuantifyPanel";

const TABS: { id: RightTab; label: string }[] = [
  { id: "info", label: "Info" },
  { id: "process", label: "Process" },
  { id: "quantify", label: "Quantify" },
];

/** Right-hand column: the FFT of the viewer's image, then Info (dataset
 * details), Process (rebin, filters) and Quantify (fitting) tabs. */
export function RightPanel() {
  const { rightTab, setRightTab } = useUi();
  return (
    <>
      <FftPanel />
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
        <CollapseButton side="right" label="right-hand" />
      </div>
      <div className="panel-tab-body">
        {rightTab === "info" ? <Inspector /> : rightTab === "quantify" ? <QuantifyPanel /> : <ProcessPanel />}
      </div>
    </>
  );
}
