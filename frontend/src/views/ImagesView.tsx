import { Fragment, useEffect, useState, type ReactNode } from "react";
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
  const phase = images.find((d) => d.orientation?.output === "phase");
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
        bottomTabs={[
          ...(ipf ? [{ id: "ipf-key" as const, label: "IPF key", content: <IpfKeyTab d={ipf} /> }] : []),
          ...(phase ? [{ id: "phase-key" as const, label: "Phase key", content: <PhaseKeyTab d={phase} /> }] : []),
        ]}
      />
    </div>
  );
}

/** The IPF colour key (orix's) of an IPF image's phase (each phase's for a
 * phase map): the crystal direction along the sample's x, y or z (z: the
 * beam) coloured as in the map. */
function IpfKeyTab({ d }: { d: DatasetSummary }) {
  const files = useDatasets((s) => s.files);
  const all = allDatasets(files);
  const ids = d.orientation?.templates?.split(",") ?? [];
  const names = d.orientation?.phases?.map((p) => p.name) ?? [d.orientation?.phase ?? ""];
  const structures = ids.map((id) => all.find((x) => x.id === id)?.parent_id ?? null);
  const [plots, setPlots] = useState<Record<string, StructurePlots>>({});
  useEffect(() => {
    let live = true;
    for (const s of structures) {
      if (!s || plots[s]) continue;
      api.structurePlots(s).then((p) => live && setPlots((x) => ({ ...x, [s]: p })), () => undefined);
    }
    return () => {
      live = false;
    };
  }, [structures.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div className="templates-tab">
      <div className="templates-panes">
        {structures.map((s, i) => (
          <figure key={i} className="ipf-key">
            {s && plots[s] ? <IpfKey sector={plots[s].sector} /> : <p className="muted small">{s ? "Loading…" : "Its templates aren't open."}</p>}
            <figcaption className="small muted">
              {names[i]} · along {d.orientation?.direction}{d.orientation?.direction === "z" ? " (beam)" : ""}
            </figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}

/** A phase map's key: each phase's colour. */
function PhaseKeyTab({ d }: { d: DatasetSummary }) {
  return (
    <div className="templates-tab">
      <dl className="props">
        {(d.orientation?.phases ?? []).map((p) => (
          <Fragment key={p.templates}>
            <dt><span className="match-dot" style={{ background: p.colour }} /></dt>
            <dd>{p.name}</dd>
          </Fragment>
        ))}
      </dl>
    </div>
  );
}
