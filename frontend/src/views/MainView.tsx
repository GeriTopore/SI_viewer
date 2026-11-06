import { useSelectedDataset } from "../state/datasets";
import { ImageDatasetView } from "./ImageDatasetView";
import { SpectrumImageView } from "./SpectrumImageView";
import { SpectrumView } from "./SpectrumView";
import { StackView } from "./StackView";

/** The viewer for the selected dataset, chosen by its kind. */
export function MainView() {
  const ds = useSelectedDataset();
  if (!ds) {
    return (
      <div className="placeholder">
        <h1>No dataset selected</h1>
        <p className="muted">Open .emd files or TIFF stacks to start (⌘/Ctrl+O).</p>
      </div>
    );
  }
  // Keyed by id: each dataset gets fresh view state.
  switch (ds.kind) {
    case "image":
      return <ImageDatasetView key={ds.id} ds={ds} />;
    case "stack":
      return <StackView key={ds.id} ds={ds} />;
    case "spectrum":
      return <SpectrumView key={ds.id} ds={ds} />;
    case "spectrum_image":
      return <SpectrumImageView key={ds.id} ds={ds} />;
    default:
      return (
        <div className="placeholder">
          <h1>{ds.title}</h1>
          <p className="muted">This kind of dataset can't be displayed yet.</p>
        </div>
      );
  }
}
