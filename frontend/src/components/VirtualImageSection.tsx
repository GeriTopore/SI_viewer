import { useState } from "react";
import type { DatasetSummary } from "../api/client";
import { useDatasets } from "../state/datasets";
import {
  defaultDetector, defaultLattice, detectorForRequest, newGroup, type Detector, type DetectorGroup, type Lattice,
} from "../state/fourd";
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
  const preset = async (kind: "bf" | "adf" | "both") => {
    const found = await findDisk();
    if (!found) return;
    add((gs) => {
      const out: DetectorGroup[] = [];
      if (kind !== "adf") out.push(newGroup(gs, bf(found), "BF"));
      if (kind !== "bf") out.push(newGroup([...gs, ...out], adf(found), "ADF"));
      return out;
    });
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
    { label: "BF + ADF", onSelect: () => void preset("both") },
    {
      label: `Duplicate ${group.name}`,
      onSelect: () => add((gs) => [{ ...newGroup(gs, det, `${group.name} copy`, group.response) }]),
    },
  ];
  const remove = (id: string) => {
    const groups = st.groups.filter((g) => g.id !== id);
    if (!groups.length) return;
    patch({ groups, ...(st.selectedGroup === id && { selectedGroup: groups[groups.length - 1].id }) });
  };

  // ---------- saving ----------
  // A named group ("BF") keeps its name in the image's title.
  const titleOf = (g: DetectorGroup) =>
    /^Group \d+$/.test(g.name) ? undefined : `${g.name}: ${detectorText(g.detector, cal.scale, cal.units)}${g.response === "sum" ? "" : `, ${g.response}`}`;
  const saveGroups = async (groups: DetectorGroup[]) => {
    setSaving(true);
    setError(null);
    try {
      let file = null;
      for (const g of groups) {
        const res = await fetch(`/api/datasets/${d.id}/4d/virtual/save`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ detector: detectorForRequest(g.detector, w, h), response: g.response, title: titleOf(g) }),
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
            className={`layer-row no-opacity${g.id === group.id ? " active" : ""}${g.visible ? "" : " hidden"}`}
            onClick={() => g.id !== group.id && patch({ selectedGroup: g.id, shown: "live" })}
            title="Click to edit this group; double-click its name to rename it"
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
        {segmented("Response", RESPONSES, group.response, (v) => patchGroup(group.id, { response: v }))}
        <span className="muted small fourd-readout">{detectorText(det, cal.scale, cal.units)}</span>
      </div>
      {det.shape === "apertures" && (
        <>
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
