import { useRef, useState } from "react";
import type { DatasetSummary } from "../api/client";
import { allDatasets, useDatasets } from "../state/datasets";
import type { Disk as FoundDisk } from "../state/disks";
import { comOptionsOf, useCom } from "../state/com";
import {
  defaultDetector, defaultLattice, imageRequest, isComResponse, mergeDetectors, mergeable, newGroup, responseLabel, sizeApertures,
  type Detector, type DetectorGroup, type Lattice,
} from "../state/fourd";
import { formatValue } from "../format";
import { RESPONSES, SHAPES, detectorText, segmented, useFourDState } from "../views/FourDView";
import { ContextMenu, type MenuItem } from "./ContextMenu";
import { EyeIcon, EyeOffIcon } from "./LayersPanel";

type Disk = { cx: number; cy: number; r: number };

const LATTICE_FIELDS = [
  ["n1", "g1 ±", "Repeats along g1, each way"],
  ["s1", "g1 shift", "Shift along g1 (e.g. 0.5 for half orders)"],
  ["n2", "g2 ±", "Repeats along g2, each way"],
  ["s2", "g2 shift", "Shift along g2"],
] as const;

/** The detector groups of a 4D-STEM dataset (each a virtual detector with its
 * response, name and colour, giving a virtual image): the list, the selected
 * group's shape, response, apertures and g-vector array, and keeping the
 * images as datasets. The detectors are drawn and dragged on the pattern. */
export function VirtualImageSection({ d }: { d: DatasetSummary }) {
  const { patternG, st, patch, group, det, patchGroup } = useFourDState(d);
  const cal = patternG.calX;
  const [w, h] = [patternG.width, patternG.height];
  const setDetector = (detector: Detector) => patchGroup(group.id, { detector });
  const vectorsOf = (files: ReturnType<typeof useDatasets.getState>["files"]) =>
    allDatasets(files).find((x) => x.origin === "disks" && x.parent_id === d.id) ?? null;
  const hasVectors = useDatasets((s) => !!vectorsOf(s.files));
  // A digital dark field only once the disks have been found.
  const ddf = group.source === "vectors" && hasVectors;
  // Centre-of-mass responses: the descan and rotation of the DPC / iCOM process.
  const comOpts = useCom((s) => comOptionsOf(s, d.id));
  const comLive = { plane: comOpts.plane, rotation: comOpts.rotation };
  const [disk, setDisk] = useState<Disk | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [addMenu, setAddMenu] = useState<{ x: number; y: number } | null>(null);

  // The bright-field disk of the mean pattern: sizes the apertures, places the array's origin, BF/ADF presets.
  const findDisk = async (): Promise<Disk | null> => {
    if (disk) return disk;
    setError(null);
    try {
      const res = await fetch(`/api/datasets/${d.id}/4d/bf-disk`);
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail ?? res.statusText);
      setDisk(body);
      return body;
    } catch (e) {
      setError((e as Error).message);
      return null;
    }
  };
  const bfRadius = async () => {
    const found = await findDisk();
    if (found) setDetector({ ...det, r: found.r });
  };
  const toggleLattice = async (on: boolean) => {
    if (!on) return setDetector({ ...det, lattice: null });
    const found = await findDisk();
    setDetector({ ...det, lattice: defaultLattice(found?.cx ?? w / 2, found?.cy ?? h / 2, w) });
  };
  const setLattice = (p: Partial<Lattice>) => det.lattice && setDetector({ ...det, lattice: { ...det.lattice, ...p } });

  // ---------- apertures from the disks found (the Bragg vectors) ----------
  // Those at the cursor shown (or the last one used): only once the disks
  // have been found (Disk detection), as is a digital dark field.
  const files = useDatasets((s) => s.files);
  const vectorsDs = vectorsOf(files);
  const NO_VECTORS = "Find the disks first (Disk detection)";
  const [brightest, setBrightest] = useState(0);
  const [withBeam, setWithBeam] = useState(false);
  const [finding, setFinding] = useState(false);
  const disksShown = async (): Promise<{ disks: FoundDisk[]; r: number }> => {
    if (!vectorsDs?.disks) throw new Error(NO_VECTORS);
    const cursor = st.cursors.find((c) => c.id === st.source) ?? st.cursors.find((c) => c.id === st.lastCursor) ?? st.cursors[0];
    const res = await fetch(`/api/datasets/${d.id}/4d/disks?x=${cursor.x}&y=${cursor.y}`);
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail ?? res.statusText);
    return { disks: body.disks, r: vectorsDs.disks.template.r };
  };
  const fromDisks = async () => {
    setFinding(true);
    setError(null);
    try {
      const found = await disksShown();
      let rows = [...found.disks].sort((a, b) => b[2] - a[2]);
      // The direct beam: the disk on the bright-field disk of the mean pattern.
      if (!withBeam) {
        const bfDisk = await findDisk();
        if (bfDisk) rows = rows.filter(([ky, kx]) => Math.hypot(kx - bfDisk.cx, ky - bfDisk.cy) > bfDisk.r);
      }
      if (brightest > 0) rows = rows.slice(0, brightest);
      if (!rows.length) throw new Error("No disks in the pattern shown");
      // A digital dark field keeps its match radius; a pixel image takes the disks' size.
      setDetector({ ...det, apertures: rows.map(([ky, kx]) => ({ x: kx, y: ky })), r: group.source === "vectors" ? det.r : found.r });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setFinding(false);
    }
  };

  // ---------- the groups ----------
  const add = (made: (groups: DetectorGroup[]) => DetectorGroup[]) => {
    const groups = [...st.groups];
    for (const g of made(groups)) groups.push(g);
    patch({ groups, selectedGroup: groups[groups.length - 1].id, shown: "live" });
  };
  const bf = (found: Disk): Detector => ({ ...defaultDetector(w, h, "circle"), cx: found.cx, cy: found.cy, r: found.r });
  const adf = (found: Disk): Detector => ({
    ...defaultDetector(w, h, "annulus"),
    cx: found.cx,
    cy: found.cy,
    r_inner: found.r,
    // Out to the nearest edge of the pattern.
    r: Math.max(found.r + 1, Math.min(found.cx, found.cy, w - found.cx, h - found.cy)),
  });
  const preset = async (kind: "bf" | "adf") => {
    const found = await findDisk();
    if (!found) return;
    add((gs) => [kind === "bf" ? newGroup(gs, bf(found), "BF") : newGroup(gs, adf(found), "ADF")]);
  };
  const addItems: MenuItem[] = [
    ...SHAPES.map((s) => ({
      label: s.label,
      title: s.title,
      // Apertures start empty; every shape starts centred on the pattern.
      onSelect: () => add((gs) => [newGroup(gs, defaultDetector(w, h, s.id))]),
    })),
    { label: "BF", title: "A disc on the bright-field disk of the mean pattern", onSelect: () => void preset("bf") },
    { label: "ADF", title: "An annulus from the bright-field disk to the pattern's edge", onSelect: () => void preset("adf") },
    { label: `Duplicate ${group.name}`, onSelect: () => duplicate(group) },
  ];
  const remove = (id: string) => {
    const groups = st.groups.filter((g) => g.id !== id);
    if (!groups.length) return;
    patch({ groups, ...(st.selectedGroup === id && { selectedGroup: groups[groups.length - 1].id }) });
  };
  const duplicate = (g: DetectorGroup) =>
    add((gs) => [{ ...newGroup(gs, g.detector, `${g.name} copy`, g.response), source: g.source, weight: g.weight }]);

  // ---------- merging: one group's apertures added to another's ----------
  // Apertures and circles (a circle: one aperture); each keeps its size.
  const MERGE_ONLY = "Only aperture and circle groups can be merged";
  const merge = (sourceId: string, targetId: string) => {
    const src = st.groups.find((g) => g.id === sourceId);
    const tgt = st.groups.find((g) => g.id === targetId);
    if (!src || !tgt || src.id === tgt.id) return;
    const detector = mergeDetectors(tgt.detector, src.detector, w, h);
    if (!detector) return setError(`${MERGE_ONLY} (${src.name}: ${src.detector.shape}, ${tgt.name}: ${tgt.detector.shape})`);
    setError(null);
    const groups = st.groups.filter((g) => g.id !== src.id).map((g) => (g.id === tgt.id ? { ...g, detector } : g));
    patch({ groups, selectedGroup: tgt.id, shown: "live" });
  };
  const [groupMenu, setGroupMenu] = useState<{ x: number; y: number; id: string } | null>(null);
  const groupItems = (g: DetectorGroup): MenuItem[] => {
    const others = st.groups.filter((o) => o.id !== g.id);
    return [
      { label: "Rename", onSelect: () => setRenaming(g.id) },
      ...(others.length
        ? [
            { label: `Merge ${g.name} into`, disabled: true },
            ...others.map((o) => {
              const ok = mergeable(g.detector) && mergeable(o.detector);
              return {
                label: o.name, color: o.color, indent: true, disabled: !ok,
                title: ok ? `${g.name}'s apertures join ${o.name}'s (each keeps its size); ${g.name} goes` : MERGE_ONLY,
                onSelect: () => merge(g.id, o.id),
              };
            }),
          ]
        : []),
      { label: "Duplicate", onSelect: () => duplicate(g) },
      { label: `Delete ${g.name}`, danger: true, disabled: st.groups.length < 2, onSelect: () => remove(g.id) },
    ];
  };
  // Drag a group onto another: merged into it.
  const GROUP_MIME = "application/x-si-viewer-group";
  const [dropOn, setDropOn] = useState<string | null>(null);

  // ---------- sizes: the radii (or a rectangle's sides), slider and box ----------
  // The box in the pattern's calibrated units, the slider in pixels. The
  // slider spans a narrow range for its shape; a size beyond it (typed, or
  // dragged on the pattern) widens it to a round number past the size. Per
  // group and control, it only widens (no jump while dragging back down).
  const ranges = useRef<Record<string, number>>({});
  const roundUp = (v: number) => {
    const p = 10 ** Math.floor(Math.log10(v));
    return ([1, 2, 5, 10].find((m) => m * p >= v) ?? 10) * p;
  };
  const sizeControl = (label: string, title: string, px: number, max: number, set: (px: number) => void) => {
    const key = `${group.id}:${label}`;
    let top = Math.max(max, ranges.current[key] ?? 0);
    if (px > top) top = ranges.current[key] = roundUp(px * 1.5);
    return (
    <label key={label} title={title}>
      <span>{label}</span>
      <span className="size-control">
        <input
          type="range"
          min={0.5}
          max={top}
          step={top > 50 ? 1 : 0.5}
          value={Math.min(px, top)}
          onChange={(e) => set(Number(e.target.value))}
        />
        <input
          key={px}
          type="number"
          className="cell-input"
          step="any"
          min={0}
          defaultValue={formatValue(px * cal.scale)}
          onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
          onBlur={(e) => {
            const v = Number(e.target.value) / cal.scale;
            if (Number.isFinite(v) && v > 0 && Math.abs(v - px) > 1e-9) set(v);
          }}
        />
        <span className="muted small">{cal.units || "px"}</span>
      </span>
    </label>
    );
  };
  // The sliders' starting ranges (pattern px): apertures sit on single
  // disks; a disc or annulus reaches a quarter of the pattern; a rectangle half.
  const small = Math.min(w, h);
  const apertureMax = roundUp(Math.max(4, small / 16));
  const discMax = roundUp(small / 4);
  const sideMax = roundUp(small / 2);
  const sizeControls = (() => {
    switch (det.shape) {
      case "circle":
        return [sizeControl("Radius", "The disc's radius", det.r, discMax, (r) => setDetector({ ...det, r: Math.max(0.5, r) }))];
      case "annulus":
        return [
          sizeControl("Inner", "The annulus's inner radius", det.r_inner, discMax, (r) =>
            setDetector({ ...det, r_inner: Math.max(0, Math.min(r, det.r - 0.5)) })),
          sizeControl("Outer", "The annulus's outer radius", det.r, discMax, (r) =>
            setDetector({ ...det, r: Math.max(r, det.r_inner + 0.5) })),
        ];
      case "apertures":
        return [sizeControl(
          "Radius",
          ddf ? "Every aperture's radius: the match tolerance of the digital dark field" : "Every aperture's radius (merged ones too)",
          det.r, apertureMax, (r) => setDetector(sizeApertures(det, Math.max(0.5, r))),
        )];
      case "rectangle": {
        const [cx, cy] = [(det.x0 + det.x1) / 2, (det.y0 + det.y1) / 2];
        return [
          sizeControl("Width", "The rectangle's width (about its centre)", Math.abs(det.x1 - det.x0), sideMax, (v) =>
            setDetector({ ...det, x0: cx - Math.max(0.5, v) / 2, x1: cx + Math.max(0.5, v) / 2 })),
          sizeControl("Height", "The rectangle's height (about its centre)", Math.abs(det.y1 - det.y0), sideMax, (v) =>
            setDetector({ ...det, y0: cy - Math.max(0.5, v) / 2, y1: cy + Math.max(0.5, v) / 2 })),
        ];
      }
      default:
        return [];
    }
  })();

  // ---------- saving ----------
  // A named group ("BF") keeps its name in the image's title.
  const titleOf = (g: DetectorGroup) =>
    /^Group \d+$/.test(g.name)
      ? undefined
      : `${isDdf(g) ? "DDF " : ""}${g.name}: ${detectorText(g.detector, cal.scale, cal.units)}${g.response === "sum" ? "" : `, ${responseLabel(g.response)}`}${isDdf(g) && g.weight === "count" ? " (count)" : ""}`;
  function isDdf(g: DetectorGroup) {
    return g.source === "vectors" && hasVectors;
  }
  const saveGroups = async (groups: DetectorGroup[]) => {
    setSaving(true);
    setError(null);
    try {
      let file = null;
      for (const g of groups) {
        const res = await fetch(`/api/datasets/${d.id}/4d/virtual/save`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...imageRequest(g, w, h, hasVectors, comLive), title: titleOf(g) }),
        });
        const body = await res.json();
        if (!res.ok) throw new Error(`${g.name}: ${body.detail ?? res.statusText}`);
        file = body.file;
      }
      // The new images join the dataset list; the 4D view stays shown.
      if (file) useDatasets.getState().replaceFile(file);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };
  const visible = st.groups.filter((g) => g.visible);

  return (
    <section className="q-form virtual-form">
      <ul className="group-list">
        {st.groups.map((g) => (
          <li
            key={g.id}
            className={`layer-row no-opacity${g.id === group.id ? " active" : ""}${g.visible ? "" : " hidden"}${dropOn === g.id ? " drop-target" : ""}`}
            onClick={() => g.id !== group.id && patch({ selectedGroup: g.id, shown: "live" })}
            onContextMenu={(e) => {
              e.preventDefault();
              setGroupMenu({ x: e.clientX, y: e.clientY, id: g.id });
            }}
            draggable={renaming !== g.id}
            onDragStart={(e) => {
              e.dataTransfer.setData(GROUP_MIME, g.id);
              e.dataTransfer.effectAllowed = "move";
            }}
            onDragOver={(e) => {
              if (!e.dataTransfer.types.includes(GROUP_MIME)) return;
              e.preventDefault();
              if (dropOn !== g.id) setDropOn(g.id);
            }}
            onDragLeave={() => dropOn === g.id && setDropOn(null)}
            onDragEnd={() => setDropOn(null)}
            onDrop={(e) => {
              e.preventDefault();
              setDropOn(null);
              merge(e.dataTransfer.getData(GROUP_MIME), g.id);
            }}
            title="Click to edit this group; double-click its name (or right-click) to rename it; drag it onto another group to merge them"
          >
            <button
              className="layer-eye"
              aria-pressed={g.visible}
              title={g.visible ? "Hide this group's detector" : "Show this group's detector"}
              onClick={(e) => {
                e.stopPropagation();
                patchGroup(g.id, { visible: !g.visible });
              }}
            >
              {g.visible ? <EyeIcon /> : <EyeOffIcon />}
            </button>
            <label className="layer-swatch" style={{ background: g.color }} title="Choose colour" onClick={(e) => e.stopPropagation()}>
              <input type="color" value={g.color} onChange={(e) => patchGroup(g.id, { color: e.target.value })} />
            </label>
            {renaming === g.id ? (
              <input
                className="cell-input"
                autoFocus
                defaultValue={g.name}
                onFocus={(e) => e.target.select()}
                onClick={(e) => e.stopPropagation()}
                onBlur={(e) => {
                  const name = e.target.value.trim();
                  if (name) patchGroup(g.id, { name });
                  setRenaming(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                  if (e.key === "Escape") setRenaming(null);
                  e.stopPropagation();
                }}
              />
            ) : (
              <span className="layer-label" onDoubleClick={() => setRenaming(g.id)}>{g.name}</span>
            )}
            <button
              className="icon-button"
              title={`Remove ${g.name}`}
              disabled={st.groups.length < 2}
              onClick={(e) => {
                e.stopPropagation();
                remove(g.id);
              }}
            >
              ×
            </button>
          </li>
        ))}
      </ul>
      {groupMenu && (() => {
        const g = st.groups.find((x) => x.id === groupMenu.id);
        return g ? <ContextMenu x={groupMenu.x} y={groupMenu.y} onClose={() => setGroupMenu(null)} items={groupItems(g)} /> : null;
      })()}
      <div className="virtual-row">
        <button
          className="secondary"
          aria-haspopup="menu"
          onMouseDown={(e) => addMenu && e.stopPropagation()}
          onClick={(e) => {
            if (addMenu) return setAddMenu(null);
            const r = e.currentTarget.getBoundingClientRect();
            setAddMenu({ x: r.left, y: r.bottom + 4 });
          }}
        >
          + Group ▾
        </button>
        {addMenu && <ContextMenu x={addMenu.x} y={addMenu.y} onClose={() => setAddMenu(null)} items={addItems} />}
      </div>

      <div className="virtual-row">
        {segmented("Image from", [
          { id: "pixels", label: "Pixels", title: "The patterns' pixels inside the detector" },
          {
            id: "vectors",
            label: "Bragg vectors",
            disabled: !vectorsDs,
            title: vectorsDs
              ? "Digital dark field: the disks found inside the detector, centred per position when the Centring is on; an aperture's radius is the match tolerance"
              : `Digital dark field: ${NO_VECTORS}`,
          },
        ], ddf ? "vectors" : "pixels", (v) => patchGroup(group.id, { source: v }))}
      </div>
      <div className="virtual-row">
        {segmented(
          "Response",
          // A centre of mass is of the pixels, not of the disks found.
          RESPONSES.map((r) => (ddf && isComResponse(r.id) ? { ...r, disabled: true, title: `${r.title} (not with Bragg vectors)` } : r)),
          ddf && isComResponse(group.response) ? "sum" : group.response,
          (v) => patchGroup(group.id, { response: v }),
        )}
        {ddf && segmented("Of", [
          { id: "intensity", label: "Intensity", title: "The disks' intensity (their correlation score)" },
          { id: "count", label: "Count", title: "The number of disks" },
        ], group.weight ?? "intensity", (v) => patchGroup(group.id, { weight: v }))}
        <span className="muted small fourd-readout">{detectorText(det, cal.scale, cal.units)}</span>
      </div>
      {sizeControls}
      {det.shape === "apertures" && (
        <>
          <div className="virtual-row">
            <button
              className="secondary"
              onClick={() => void fromDisks()}
              disabled={finding || !vectorsDs}
              title={vectorsDs ? "Place an aperture on each disk found at the cursor shown (or the last one used)" : NO_VECTORS}
            >
              {finding ? "Finding…" : "From the disks"}
            </button>
            <label className="field" title="Only the brightest disks (0: all)">
              Brightest
              <input
                type="number"
                className="cell-input"
                min={0}
                max={200}
                step={1}
                value={brightest}
                onChange={(e) => setBrightest(Math.max(0, Math.min(200, Math.round(Number(e.target.value) || 0))))}
              />
            </label>
            <label className="q-check-line small" title="Also an aperture on the direct beam">
              <input type="checkbox" checked={withBeam} onChange={(e) => setWithBeam(e.target.checked)} />
              Direct beam
            </label>
          </div>
          <div className="virtual-row">
            <button className="secondary" onClick={bfRadius} title="Size the apertures as the bright-field disk of the mean pattern">
              BF radius
            </button>
            <button
              className="secondary"
              onClick={() => setDetector({ ...det, apertures: [] })}
              disabled={!(det.apertures ?? []).length}
              title="Remove the apertures placed by ⌘/Ctrl-click"
            >
              Clear
            </button>
          </div>
          <label className="q-check-line small" title="Apertures at O + i g1 + j g2: drag O, g1 and g2 on the pattern">
            <input type="checkbox" checked={!!det.lattice} onChange={(e) => toggleLattice(e.target.checked)} />
            g-vector array
          </label>
          {det.lattice && (
            <>
              {LATTICE_FIELDS.map(([k, label, title]) => (
                <label key={k} title={title}>
                  <span>{label}</span>
                  <input
                    type="number"
                    step={k.startsWith("n") ? 1 : 0.5}
                    min={k.startsWith("n") ? 0 : -1}
                    max={k.startsWith("n") ? 20 : 1}
                    value={det.lattice![k]}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      if (!Number.isFinite(v)) return;
                      setLattice({ [k]: k.startsWith("n") ? Math.max(0, Math.min(20, Math.round(v))) : Math.max(-1, Math.min(1, v)) });
                    }}
                  />
                </label>
              ))}
              <label className="q-check-line small" title="Include the aperture at O itself">
                <input type="checkbox" checked={det.lattice.origin} onChange={(e) => setLattice({ origin: e.target.checked })} />
                Origin
              </label>
            </>
          )}
        </>
      )}
      {error && <p className="field-error">{error}</p>}
      <div className="process-actions">
        <button
          className="primary"
          onClick={() => void saveGroups([group])}
          disabled={saving}
          title={`Keep ${group.name}'s virtual image as an image dataset under the 4D-STEM data`}
        >
          {saving ? "Saving…" : "Save image"}
        </button>
        {st.groups.length > 1 && (
          <button
            className="secondary"
            onClick={() => void saveGroups(visible)}
            disabled={saving || !visible.length}
            title="Keep the virtual image of every visible group"
          >
            Save all ({visible.length})
          </button>
        )}
      </div>
    </section>
  );
}
