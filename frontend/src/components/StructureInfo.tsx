import { useEffect, useRef, useState } from "react";
import { api, type DatasetSummary, type StructureInfo as Info, type StructurePlots } from "../api/client";
import { formatValue } from "../format";
import { allDatasets, useDatasets } from "../state/datasets";
import { useOrientation } from "../state/orientation";
import { useUi } from "../state/ui";
import { BottomPane, type PaneTab } from "./BottomPane";
import { IpfKey, SymmetryPlot } from "./OrixPlots";
import { StructureScene } from "./StructureScene";
import { TemplatesPanel } from "./TemplatesView";

const cellLabel = (c: number[]) =>
  `a ${formatValue(c[0])}, b ${formatValue(c[1])}, c ${formatValue(c[2])} Å · α ${formatValue(c[3])}°, β ${formatValue(c[4])}°, γ ${formatValue(c[5])}°`;

const compositionLabel = (comp: Info["composition"]) =>
  Object.entries(comp).map(([e, n]) => `${e} ${formatValue(n)}`).join(", ");

/** Info tab rows of a crystal structure (a CIF, read as an orix phase). */
export function StructureInfo({ s }: { s: Info }) {
  return (
    <>
      <dt>Formula</dt>
      <dd>{s.formula}</dd>
      <dt>Space group</dt>
      <dd>
        {s.spacegroup ? `${s.spacegroup.symbol} (${s.spacegroup.number})` : "–"}
        {s.point_group && <span className="muted"> · point group {s.point_group}</span>}
      </dd>
      <dt>Cell</dt>
      <dd>{cellLabel(s.cell)}</dd>
      <dt>Atoms per cell</dt>
      <dd>{compositionLabel(s.composition)}</dd>
    </>
  );
}

/** A crystal structure in the viewer: its 3D render above; below, its
templates (when made), orix's point group and colour key, its sites. */
export function StructureView({ d, templatesId }: { d: DatasetSummary; templatesId?: string }) {
  const s = d.structure!;
  const files = useDatasets((st) => st.files);
  const sets = allDatasets(files).filter((x) => x.origin === "templates" && x.templates && x.parent_id === d.id);
  const [chosen, setChosen] = useState<string | null>(templatesId ?? null);
  useEffect(() => {
    if (templatesId) {
      setChosen(templatesId);
      useUi.getState().setBottomTab("templates");
    }
  }, [templatesId]);
  const current = sets.find((x) => x.id === chosen) ?? sets.at(-1);
  const [plots, setPlots] = useState<StructurePlots | null>(null);
  useEffect(() => {
    let stale = false;
    api.structurePlots(d.id).then((r) => !stale && setPlots(r), () => undefined);
    return () => {
      stale = true;
    };
  }, [d.id]);
  const viewerRef = useRef<HTMLDivElement>(null);

  const tabs: PaneTab[] = [];
  if (current) {
    tabs.push({ id: "templates", label: "Templates", content: <TemplatesPanel sets={sets} chosen={current.id} onChoose={setChosen} /> });
  }
  tabs.push({
    id: "symmetry",
    label: "Symmetry",
    content: plots ? (
      <div className="templates-panes">
        <figure>
          <SymmetryPlot symmetry={plots.symmetry} />
          <figcaption className="small">Point group {plots.symmetry.name}</figcaption>
        </figure>
        <figure>
          <IpfKey sector={plots.sector} />
          <figcaption className="small">IPF colour key</figcaption>
        </figure>
      </div>
    ) : <p className="muted small">Loading…</p>,
  });
  tabs.push({
    id: "sites",
    label: "Sites",
    content: (
      <table className="axes">
        <thead>
          <tr>
            <th>Element</th>
            <th>x</th>
            <th>y</th>
            <th>z</th>
            <th>Occupancy</th>
          </tr>
        </thead>
        <tbody>
          {s.sites.map((a, i) => (
            <tr key={i}>
              <td>{a.element}</td>
              {a.xyz.map((x, j) => <td key={j}>{formatValue(x)}</td>)}
              <td>{formatValue(a.occupancy)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    ),
  });

  return (
    <div className="layered-viewer" ref={viewerRef}>
      <StructureScene d={d} initial={useOrientation.getState().views[d.id]}
        onOrientation={(o) => useOrientation.getState().setView(d.id, o)} />
      <BottomPane tabs={tabs} viewerRef={viewerRef} resizeLabel="Resize the structure and the pane below it" />
    </div>
  );
}
