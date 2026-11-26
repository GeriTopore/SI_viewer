import { useRef, useState, type DragEvent } from "react";
import type { DatasetSummary } from "../api/client";
import { allDatasets, useDatasets } from "../state/datasets";
import type { Disk as FoundDisk } from "../state/disks";
import {
  circleDetector, defaultDetector, defaultLattice, imageRequest, isComResponse, mergeDetectors, mergeable, newGroup, responseLabel,
  sizeApertures, type Detector, type DetectorGroup, type DisksOptions, type Lattice, type Response,
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

/** Where a group dragged over another lands: before or after it, or merged into it. */
type DropAt = { id: string; at: "before" | "after" | "merge" };

/** The detector groups of a 4D-STEM dataset (each a virtual detector with its
 * response, name and colour, giving a virtual image): the list (dragged to
 * reorder or merge), the selected group's response and sizes, and keeping
 * the images as datasets. The detectors are drawn and dragged on the pattern. */
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
  const [disk, setDisk] = useState<Disk | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [addMenu, setAddMenu] = useState<{ x: number; y: number } | null>(null);

  // The bright-field disk of the mean pattern: sizes the apertures, places the array's origin.
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
    if (found) setDetector(sizeApertures(det, found.r));
  };
  const setLattice = (p: Partial<Lattice>) => det.lattice && setDetector({ ...det, lattice: { ...det.lattice, ...p } });

  // ---------- apertures on the disks found (the Bragg vectors) ----------
  // Those at the cursor shown (or the last one used): only once the disks
  // have been found (Disk detection), as is a digital dark field.
  const files = useDatasets((s) => s.files);
  const vectorsDs = vectorsOf(files);
  const NO_VECTORS = "Find the disks first (Disk detection)";
  const [finding, setFinding] = useState(false);
  const disksShown = async (): Promise<{ disks: FoundDisk[]; r: number }> => {
    if (!vectorsDs?.disks) throw new Error(NO_VECTORS);
    const cursor = st.cursors.find((c) => c.id === st.source) ?? st.cursors.find((c) => c.id === st.lastCursor) ?? st.cursors[0];
    const res = await fetch(`/api/datasets/${d.id}/4d/disks?x=${cursor.x}&y=${cursor.y}`);
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail ?? res.statusText);
    return { disks: body.disks, r: vectorsDs.disks.template.r };
  };
  /** ``base`` with an aperture on each disk kept; a digital dark field keeps
   * its radius (the match tolerance), a pixel image takes the disks' size. */
  const onDisks = async (base: Detector, opts: DisksOptions, keepR: boolean): Promise<Detector> => {
    const found = await disksShown();
    let rows = [...found.disks].sort((a, b) => b[2] - a[2]);
    // The direct beam: the disk on the bright-field disk of the mean pattern.
    if (!opts.beam) {
      const bfDisk = await findDisk();
      if (bfDisk) rows = rows.filter(([ky, kx]) => Math.hypot(kx - bfDisk.cx, ky - bfDisk.cy) > bfDisk.r);
    }
    if (opts.brightest > 0) rows = rows.slice(0, opts.brightest);
    if (!rows.length) throw new Error("No disks in the pattern shown");
    return { ...base, apertures: rows.map(([ky, kx]) => ({ x: kx, y: ky })), r: keepR ? base.r : found.r };
  };
  const finds = async (run: () => Promise<void>) => {
    setFinding(true);
    setError(null);
    try {
      await run();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setFinding(false);
    }
  };
  const addFromDisks = () =>
    finds(async () => {
      const opts: DisksOptions = { brightest: 0, beam: false };
      const detector = await onDisks(defaultDetector(w, h, "apertures"), opts, false);
      add((gs) => [{ ...newGroup(gs, detector), disks: opts }]);
    });
  const redoDisks = (opts: DisksOptions) => {
    const id = group.id;
    patchGroup(id, { disks: opts });
    void finds(async () => patchGroup(id, { detector: await onDisks(det, opts, group.source === "vectors"), disks: opts }));
  };

  // ---------- the groups ----------
  const add = (made: (groups: DetectorGroup[]) => DetectorGroup[]) => {
    const groups = [...st.groups];
    for (const g of made(groups)) groups.push(g);
    patch({ groups, selectedGroup: groups[groups.length - 1].id, shown: "live" });
  };
  const addArray = async () => {
    const found = await findDisk();
    const base = defaultDetector(w, h, "apertures");
    add((gs) => [newGroup(gs, { ...base, lattice: defaultLattice(found?.cx ?? w / 2, found?.cy ?? h / 2, w) })]);
  };
  const addItems: MenuItem[] = [
    {
      label: "Circle",
      title: "A disc (e.g. bright field); ⌘/Ctrl-click the pattern for more",
      onSelect: () => add((gs) => [newGroup(gs, circleDetector(w, h))]),
    },
    // Every shape starts centred on the pattern.
    ...SHAPES.map((s) => ({ label: s.label, title: s.title, onSelect: () => add((gs) => [newGroup(gs, defaultDetector(w, h, s.id))]) })),
    { label: "Aperture array", title: "Apertures at O + i g1 + j g2: drag O, g1 and g2 on the pattern", onSelect: () => void addArray() },
    {
      label: "From the disks",
      title: vectorsDs ? "An aperture on each disk found at the cursor shown" : NO_VECTORS,
      disabled: !vectorsDs,
      onSelect: () => void addFromDisks(),
    },
  ];
  const remove = (id: string) => {
    const groups = st.groups.filter((g) => g.id !== id);
    if (!groups.length) return;
    patch({ groups, ...(st.selectedGroup === id && { selectedGroup: groups[groups.length - 1].id }) });
  };
  const duplicate = (g: DetectorGroup) =>
    add((gs) => [{ ...newGroup(gs, g.detector, `${g.name} copy`, g.response), source: g.source, weight: g.weight, disks: g.disks }]);

  // ---------- merging: one group's apertures added to another's ----------
  // Apertures and circles (a circle: one aperture); each keeps its size.
  const MERGE_ONLY = "Only circle and array groups can be merged";
  const merge = (sourceId: string, targetId: string) => {
    const src = st.groups.find((g) => g.id === sourceId);
    const tgt = st.groups.find((g) => g.id === targetId);
    if (!src || !tgt || src.id === tgt.id) return;
    const detector = mergeDetectors(tgt.detector, src.detector, w, h);
    if (!detector) return setError(`${MERGE_ONLY} (${src.name}: ${src.detector.shape}, ${tgt.name}: ${tgt.detector.shape})`);
    setError(null);
    const groups = st.groups.filter((g) => g.id !== src.id).map((g) => (g.id === tgt.id ? { ...g, detector, disks: undefined } : g));
    patch({ groups, selectedGroup: tgt.id, shown: "live" });
  };
  /** Group ``id`` moved to before or after ``targetId``. */
  const reorder = (id: string, targetId: string, after: boolean) => {
    const moved = st.groups.find((g) => g.id === id);
    if (!moved || id === targetId) return;
    const rest = st.groups.filter((g) => g.id !== id);
    rest.splice(rest.findIndex((g) => g.id === targetId) + (after ? 1 : 0), 0, moved);
    patch({ groups: rest });
  };
  const [groupMenu, setGroupMenu] = useState<{ x: number; y: number; id: string } | null>(null);
  const groupItems = (g: DetectorGroup): MenuItem[] => {
    const others = st.groups.filter((o) => o.id !== g.id);
    return [
      { label: "Rename", onSelect: () => setRenaming(g.id) },
      { label: "Duplicate", onSelect: () => duplicate(g) },
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
      { label: `Delete ${g.name}`, danger: true, disabled: st.groups.length < 2, onSelect: () => remove(g.id) },
    ];
  };
  // Drag a group: onto another's middle, merged into it (circles and arrays);
  // onto its upper or lower part, moved before or after it.
  const GROUP_MIME = "application/x-si-viewer-group";
  const dragging = useRef<string | null>(null);
  const [dropOn, setDropOn] = useState<DropAt | null>(null);
  const dropAt = (e: DragEvent<HTMLLIElement>, g: DetectorGroup): DropAt["at"] => {
    const r = e.currentTarget.getBoundingClientRect();
    const f = (e.clientY - r.top) / r.height;
    const src = st.groups.find((x) => x.id === dragging.current);
    const canMerge = !!src && mergeable(src.detector) && mergeable(g.detector);
    if (canMerge) return f < 0.3 ? "before" : f > 0.7 ? "after" : "merge";
    return f < 0.5 ? "before" : "after";
  };

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
  const placed = (det.apertures ?? []).length;
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
          ddf ? "Every aperture's radius: the match tolerance of the digital dark field" : "Every circle's radius (merged ones too)",
          // One circle (no array) as a disc; several, as apertures on disks.
          det.r, placed <= 1 && !det.lattice ? discMax : apertureMax, (r) => setDetector(sizeApertures(det, Math.max(0.5, r))),
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
          body: JSON.stringify({ ...imageRequest(g, w, h, hasVectors), title: titleOf(g) }),
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
  // A centre of mass is of the pixels, not of the disks found.
  const response = ddf && isComResponse(group.response) ? "sum" : group.response;
  const disks = det.shape === "apertures" ? group.disks : undefined;

  return (
    <section className="q-form virtual-form">
      <ul className="group-list">
        {st.groups.map((g) => (
          <li
            key={g.id}
            className={`layer-row no-opacity${g.id === group.id ? " active" : ""}${g.visible ? "" : " hidden"}${dropOn?.id === g.id ? ` drop-${dropOn.at}` : ""}`}
            onClick={() => g.id !== group.id && patch({ selectedGroup: g.id, shown: "live" })}
            onContextMenu={(e) => {
              e.preventDefault();
              setGroupMenu({ x: e.clientX, y: e.clientY, id: g.id });
            }}
            draggable={renaming !== g.id}
            onDragStart={(e) => {
              dragging.current = g.id;
              e.dataTransfer.setData(GROUP_MIME, g.id);
              e.dataTransfer.effectAllowed = "move";
            }}
            onDragOver={(e) => {
              if (!e.dataTransfer.types.includes(GROUP_MIME) || dragging.current === g.id) return;
              e.preventDefault();
              const at = dropAt(e, g);
              if (dropOn?.id !== g.id || dropOn.at !== at) setDropOn({ id: g.id, at });
            }}
            onDragLeave={(e) => {
              if (!e.currentTarget.contains(e.relatedTarget as Node | null) && dropOn?.id === g.id) setDropOn(null);
            }}
            onDragEnd={() => {
              dragging.current = null;
              setDropOn(null);
            }}
            onDrop={(e) => {
              e.preventDefault();
              const id = e.dataTransfer.getData(GROUP_MIME);
              const at = dropAt(e, g);
              setDropOn(null);
              dragging.current = null;
              if (at === "merge") merge(id, g.id);
              else reorder(id, g.id, at === "after");
            }}
            title="Click to edit; double-click the name to rename; drag to reorder, or onto another circle or array group to merge; right-click for more"
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
          disabled={finding}
          onMouseDown={(e) => addMenu && e.stopPropagation()}
          onClick={(e) => {
            if (addMenu) return setAddMenu(null);
            const r = e.currentTarget.getBoundingClientRect();
            setAddMenu({ x: r.left, y: r.bottom + 4 });
          }}
        >
          {finding && !disks ? "Finding…" : "+ Group ▾"}
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
              ? "Digital dark field: the disks found inside the detector, centred per position while the descan is corrected; an aperture's radius is the match tolerance"
              : `Digital dark field: ${NO_VECTORS}`,
          },
        ], ddf ? "vectors" : "pixels", (v) => patchGroup(group.id, { source: v }))}
      </div>
      <label title={RESPONSES.find((r) => r.id === response)?.title}>
        <span>Response</span>
        <select value={response} onChange={(e) => patchGroup(group.id, { response: e.target.value as Response })}>
          {RESPONSES.map((r) => (
            <option key={r.id} value={r.id} disabled={ddf && isComResponse(r.id)}>
              {r.label}
            </option>
          ))}
        </select>
      </label>
      {ddf && (
        <div className="virtual-row">
          {segmented("Of", [
            { id: "intensity", label: "Intensity", title: "The disks' intensity (their correlation score)" },
            { id: "count", label: "Count", title: "The number of disks" },
          ], group.weight ?? "intensity", (v) => patchGroup(group.id, { weight: v }))}
        </div>
      )}
      {sizeControls}
      {disks && (
        <div className="virtual-row">
          <label className="field" title="Only the brightest disks (0: all)">
            Brightest
            <input
              type="number"
              className="cell-input"
              min={0}
              max={200}
              step={1}
              value={disks.brightest}
              onChange={(e) => redoDisks({ ...disks, brightest: Math.max(0, Math.min(200, Math.round(Number(e.target.value) || 0))) })}
            />
          </label>
          <label className="q-check-line small" title="Also an aperture on the direct beam">
            <input type="checkbox" checked={disks.beam} onChange={(e) => redoDisks({ ...disks, beam: e.target.checked })} />
            Direct beam
          </label>
          <button
            className="secondary"
            onClick={() => redoDisks(disks)}
            disabled={finding || !vectorsDs}
            title={vectorsDs ? "Place them again on the disks at the cursor shown (or the last one used)" : NO_VECTORS}
          >
            {finding ? "Finding…" : "At the cursor"}
          </button>
        </div>
      )}
      {det.shape === "apertures" && (
        <div className="virtual-row">
          <button className="secondary" onClick={bfRadius} title="Size the circles as the bright-field disk of the mean pattern">
            BF radius
          </button>
          {placed > 1 && (
            <button className="secondary" onClick={() => setDetector({ ...det, apertures: [] })} title="Remove the circles placed">
              Clear
            </button>
          )}
        </div>
      )}
      {det.lattice && det.shape === "apertures" && (
        <>
          <p className="process-heading small">Array</p>
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
