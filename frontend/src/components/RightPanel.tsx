import { useUi, type RightTab } from "../state/ui";
import { FftPanel } from "./FftPanel";
import { Inspector } from "./Inspector";
import { LayersTab } from "./LayersTab";
import { CollapseButton } from "./PaneToggle";
import { ProcessPanel } from "./ProcessPanel";
import { QuantifyPanel } from "./QuantifyPanel";
import { Splitter } from "./Splitter";

const TABS: { id: RightTab; label: string }[] = [
  { id: "info", label: "Info" },
  { id: "process", label: "Process" },
  { id: "quantify", label: "Quantify" },
  { id: "layers", label: "Layers" },
];

/** Right-hand column: the FFT of the viewer's image, then Info (dataset
 * details), Process (rebin, filters), Quantify (fitting) and Layers (what
 * the viewer draws) tabs. */
export function RightPanel() {
  const { rightTab, setRightTab, fftOpen } = useUi();
  return (
    <>
      <FftPanel />
      {fftOpen && (
        <Splitter
          direction="row"
          className="split-row-resizer fft-resizer"
          label="Resize the FFT"
          onDrag={(pos) => {
            // The FFT grows by as much as the pointer is below the pane's foot.
            const panel = document.querySelector(".fft-panel");
            const wrap = panel?.querySelector(".fft-canvas-wrap");
            if (!panel || !wrap) return;
            useUi.getState().setFftSize(wrap.getBoundingClientRect().width + pos - panel.getBoundingClientRect().bottom);
          }}
          onReset={() => useUi.getState().setFftSize(null)}
        />
      )}
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
        {rightTab === "info" ? (
          <Inspector />
        ) : rightTab === "quantify" ? (
          <QuantifyPanel />
        ) : rightTab === "layers" ? (
          <LayersTab />
        ) : (
          <ProcessPanel />
        )}
      </div>
    </>
  );
}
