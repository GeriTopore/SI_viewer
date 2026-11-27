import { useEffect, useState, type ReactNode } from "react";
import { api, type DatasetSummary, type StructurePlots } from "../api/client";
import { LayeredViewer } from "../components/LayeredViewer";
import { IpfKey } from "../components/OrixPlots";
import { allDatasets, useDatasets } from "../state/datasets";
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
  // An orientation map's IPF images: orix's colour key of its phase under them.
  const ipf = images.find((d) => d.orientation?.output === "ipf");
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
        bottomTabs={ipf ? [{ id: "ipf-key" as const, label: "IPF key", content: <IpfKeyTab d={ipf} /> }] : []}
      />
    </div>
  );
}

/** The IPF colour key (orix's) of an IPF image's phase: the crystal direction
 * along the sample's x, y or z (z: the beam) coloured as in the map. */
function IpfKeyTab({ d }: { d: DatasetSummary }) {
  const files = useDatasets((s) => s.files);
  const templates = allDatasets(files).find((x) => x.id === d.orientation?.templates);
  const structure = templates?.parent_id ?? null;
  const [plots, setPlots] = useState<{ id: string; plots: StructurePlots } | null>(null);
  useEffect(() => {
    if (!structure) return;
    let live = true;
    api.structurePlots(structure).then((p) => live && setPlots({ id: structure, plots: p }), () => undefined);
    return () => {
      live = false;
    };
  }, [structure]);
  return (
    <div className="templates-tab">
      <figure className="ipf-key">
        {plots?.id === structure ? <IpfKey sector={plots.plots.sector} /> : <p className="muted small">{structure ? "Loading…" : "Its templates aren't open."}</p>}
        <figcaption className="small muted">
          {d.orientation?.phase} · along {d.orientation?.direction}{d.orientation?.direction === "z" ? " (beam)" : ""}
        </figcaption>
      </figure>
    </div>
  );
}
