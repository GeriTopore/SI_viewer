import { useMemo } from "react";
import type { DatasetSummary } from "../api/client";
import { allDatasets, isColumnItem, planeSize, useDatasets } from "../state/datasets";
import { useAlignment } from "../state/alignment";
import { isLineScan } from "../format";
import { AlignmentWorkspace } from "./AlignmentWorkspace";
import { FourDView } from "./FourDView";
import { ImagesView } from "./ImagesView";
import { LineScanView } from "./LineScanView";
import { SpectrumImageView } from "./SpectrumImageView";
import { SpectrumView } from "./SpectrumView";
import { StackView } from "./StackView";

/** Shows exactly the highlighted datasets: one spectrum image, stack or
 * spectrum (if any) with the highlighted images overlaid in colour. */
export function MainView() {
  const files = useDatasets((s) => s.files);
  const selection = useDatasets((s) => s.selection);
  const alignOpen = useAlignment((s) => s.openId);
  const selected = useMemo(() => {
    const byId = new Map(allDatasets(files).map((d) => [d.id, d]));
    return selection.map((id) => byId.get(id)).filter((d): d is DatasetSummary => !!d);
  }, [files, selection]);

  if (!selected.length) {
    return (
      <div className="placeholder">
        <h1>Nothing selected</h1>
        <p className="muted">
          Click a dataset to view it; shift-click others to add them to the view.
          {!files.length && " Open files with Open… (⌘/Ctrl+O)."}
        </p>
      </div>
    );
  }

  const primary = selected.find((d) => d.kind !== "image" && !isColumnItem(d));
  const images = selected.filter((d) => d.kind === "image");
  // Column items (kemstem) are drawn over the image they came from: shown
  // as the background when it isn't highlighted itself.
  const items = primary ? [] : selected.filter(isColumnItem);
  const byId = new Map(allDatasets(files).map((d) => [d.id, d]));
  const imageOfItem = (d: DatasetSummary) => {
    let up: DatasetSummary | undefined = d;
    while (up && up.atomic?.output !== "columns" && isColumnItem(up)) up = byId.get(up.parent_id ?? "");
    return up?.atomic?.output === "columns" ? byId.get(up.parent_id ?? "") : undefined;
  };
  const implied = [...new Map(items.map(imageOfItem).filter((d): d is DatasetSummary => !!d && !images.some((x) => x.id === d.id)).map((d) => [d.id, d])).values()];

  // Keyed by the primary dataset and its revision: each gets fresh view
  // state, and so does rebinned data.
  const key = primary && `${primary.id}@${primary.revision}`;
  // Bragg vectors (a table): shown on their 4D-STEM data's patterns.
  if (primary?.origin === "disks") {
    const cube = byId.get(primary.parent_id ?? "");
    if (cube?.kind === "4dstem") return <FourDView key={`${cube.id}@${cube.revision}`} ds={cube} />;
  }
  switch (primary?.kind) {
    case "spectrum_image":
      if (isLineScan(primary)) return <LineScanView key={key} ds={primary} />;
      return <SpectrumImageView key={key} ds={primary} images={images} />;
    case "stack":
      // Drift correction of this stack fills the viewer while it is open.
      if (alignOpen === primary.id) return <AlignmentWorkspace key={key} ds={primary} />;
      return (
        <StackView
          key={key}
          ds={primary}
          images={images}
          others={selected.filter((d) => d.kind === "stack" && d.id !== primary.id)}
        />
      );
    case "spectrum":
      return <SpectrumView key={key} ds={primary} />;
    case "4dstem":
      return <FourDView key={key} ds={primary} />;
    case undefined: {
      const shown = [...images, ...implied];
      if (!shown.length) return <div className="placeholder"><p className="muted">Its image is gone.</p></div>;
      const [w, h] = planeSize(shown[0]);
      return <ImagesView key={`images:${w}x${h}`} images={shown} items={items} implied={implied.map((d) => d.id)} />;
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
