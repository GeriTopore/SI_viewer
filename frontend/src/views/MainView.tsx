import { useMemo } from "react";
import type { DatasetSummary } from "../api/client";
import { allDatasets, planeSize, useDatasets } from "../state/datasets";
import { ImagesView } from "./ImagesView";
import { SpectrumImageView } from "./SpectrumImageView";
import { SpectrumView } from "./SpectrumView";
import { StackView } from "./StackView";

/** Shows exactly the highlighted datasets: one spectrum image, stack or
 * spectrum (if any) with the highlighted images overlaid in colour. */
export function MainView() {
  const files = useDatasets((s) => s.files);
  const selection = useDatasets((s) => s.selection);
  const selected = useMemo(() => {
    const byId = new Map(allDatasets(files).map((d) => [d.id, d]));
    return selection.map((id) => byId.get(id)).filter((d): d is DatasetSummary => !!d);
  }, [files, selection]);

  if (!selected.length) {
    return (
      <div className="placeholder">
        <h1>Nothing selected</h1>
        <p className="muted">
          Click a spectrum image, stack or spectrum to view it; click maps to add them to the view.
          {!files.length && " Open files with Open… (⌘/Ctrl+O)."}
        </p>
      </div>
    );
  }

  const primary = selected.find((d) => d.kind !== "image");
  const images = selected.filter((d) => d.kind === "image");

  // Keyed by the primary dataset: each gets fresh view state.
  switch (primary?.kind) {
    case "spectrum_image":
      return <SpectrumImageView key={primary.id} ds={primary} images={images} />;
    case "stack":
      return <StackView key={primary.id} ds={primary} images={images} />;
    case "spectrum":
      return <SpectrumView key={primary.id} ds={primary} />;
    case undefined: {
      const [w, h] = planeSize(images[0]);
      return <ImagesView key={`images:${w}x${h}`} images={images} />;
    }
    default:
      return (
        <div className="placeholder">
          <h1>{primary?.title}</h1>
          <p className="muted">This kind of dataset can't be displayed yet.</p>
        </div>
      );
  }
}
