import { useEffect, useState } from "react";
import { api, type DatasetSummary, type DpcCandidate, type DpcSettings } from "../api/client";
import { useDatasets } from "../state/datasets";

const LETTERS = ["A", "B", "C", "D"] as const;

/** The images DPC can be made from: segment images (and DPC images, to tune them). */
export const canMakeDpc = (d: DatasetSummary) =>
  (d.kind === "image" || d.kind === "stack") && (!!d.detector || d.origin === "dpc");

/** Process tab, images of a segmented detector: a DPC object (the direction
 * of A − C, B − D rotated into the image's frame, in a circular colour map
 * with the magnitude as brightness; DPC x and y under it) and an iDPC object
 * (their integral), made (again, in place) by Generate. */
export function DpcSection({ d }: { d: DatasetSummary }) {
  const [candidates, setCandidates] = useState<DpcCandidate[]>([]);
  const [s, setS] = useState<DpcSettings | null>(null);
  const [made, setMade] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [recorded, setRecorded] = useState<number | null>(null);
  const replaceFile = useDatasets((st) => st.replaceFile);

  useEffect(() => {
    let live = true;
    api
      .dpcInfo(d.id)
      .then((info) => {
        if (!live) return;
        setCandidates(info.candidates);
        setMade(!!info.settings);
        setRecorded(info.recorded);
        // As Velox makes its iDPC: the recorded rotation, the plain differences.
        setS(info.settings ?? { inputs: info.choice, normalise: false, theta: info.recorded ?? 0, mirror: false });
      })
      .catch((e) => live && setError((e as Error).message));
    return () => {
      live = false;
    };
  }, [d.id]);

  const complete = !!s && LETTERS.every((k) => s.inputs[k]);

  const make = async (settings: DpcSettings) => {
    setBusy(true);
    setError(null);
    try {
      const r = await api.makeDpc(settings);
      r.files.forEach(replaceFile);
      setMade(true);
      if (!made) useDatasets.getState().select(r.ids.colour);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const auto = async () => {
    if (!s || !complete) return;
    setBusy(true);
    setError(null);
    try {
      const r = await api.dpcRotation(s.inputs, s.normalise);
      setS({ ...s, theta: Math.round(r.theta * 10) / 10, mirror: r.mirror });
      setNote(`Least curl: ${Math.round(r.theta)}°${r.mirror ? ", mirrored" : ""}`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  if (!s) return <section className="q-form">{error && <p className="field-error">{error}</p>}</section>;
  const set = (patch: Partial<DpcSettings>) => setS({ ...s, ...patch });
  const title = (id?: string) => candidates.find((c) => c.id === id)?.title ?? "–";

  return (
    <section className="q-form">
      {LETTERS.map((k) => {
        const options = candidates.filter((c) => c.quadrant === k);
        return (
          <label key={k} title={`Quadrant ${k}: the segment image summed from it`}>
            <span>{k}</span>
            <select value={s.inputs[k] ?? ""} onChange={(e) => set({ inputs: { ...s.inputs, [k]: e.target.value } })}>
              {!s.inputs[k] && <option value="">None found</option>}
              {options.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.title} · segments {c.segments.join(" + ")}
                </option>
              ))}
            </select>
          </label>
        );
      })}
      <p className="muted small q-elements">
        x = {title(s.inputs.A)} − {title(s.inputs.C)} · y = {title(s.inputs.B)} − {title(s.inputs.D)}
      </p>
      <label title="Rotation of the detector's axes into the image's (degrees)">
        <span>Rotation (°)</span>
        <span className="q-inline process-slider">
          <input
            type="range"
            min={0}
            max={360}
            step={1}
            value={s.theta}
            onChange={(e) => set({ theta: Number(e.target.value) })}
          />
          <input
            type="number"
            step={1}
            value={s.theta}
            onChange={(e) => Number.isFinite(Number(e.target.value)) && set({ theta: Number(e.target.value) })}
          />
        </span>
      </label>
      <div className="process-actions">
        <label className="q-check" title="Flip the detector's y axis (its segments run the other way round)">
          <input type="checkbox" checked={s.mirror} onChange={(e) => set({ mirror: e.target.checked })} />
          Mirror
        </label>
        {recorded !== null && (
          <button
            onClick={() => {
              set({ theta: recorded, mirror: false });
              setNote(null);
            }}
            disabled={busy}
            title="The detector rotation recorded in the file (as Velox uses it)"
          >
            Recorded
          </button>
        )}
        <button onClick={auto} disabled={busy || !complete} title="The rotation leaving the least curl in the field">
          Auto
        </button>
      </div>
      {note && <p className="muted small q-elements">{note}</p>}
      <label className="q-check" title="Divide the differences by A + B + C + D">
        <input type="checkbox" checked={s.normalise} onChange={(e) => set({ normalise: e.target.checked })} />
        Normalise by the sum
      </label>
      {error && <p className="field-error">{error}</p>}
      <button className="primary" disabled={busy || !complete} onClick={() => make(s)}>
        {busy ? "Generating…" : "Generate"}
      </button>
    </section>
  );
}
