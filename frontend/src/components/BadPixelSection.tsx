import { useEffect, useState } from "react";
import type { DatasetSummary, FileSummary } from "../api/client";
import { useDatasets } from "../state/datasets";
import { useFourD, type BadPixels } from "../state/fourd";
import { InPlaceChoice, useInPlace } from "./InPlaceChoice";

const DEFAULTS = { hot: 5, dead: 0.2 };

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const out = await res.json();
  if (!res.ok) throw new Error(out.detail ?? res.statusText);
  return out as T;
}

/** Hot and dead detector pixels of 4D-STEM data, found in the mean pattern
 * (ringed on the pattern) and replaced in every pattern by the mean of the 8
 * around them; found again after, so the rings show what is left. */
export function BadPixelSection({ d }: { d: DatasetSummary }) {
  const [hot, setHot] = useState(DEFAULTS.hot);
  const [dead, setDead] = useState(DEFAULTS.dead);
  const found = useFourD((s) => s.bad[d.id]) ?? null;
  const setBad = useFourD((s) => s.setBad);
  const [inPlace, setInPlace] = useInPlace("repair");
  const [busy, setBusy] = useState<"find" | "repair" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [repaired, setRepaired] = useState<number | null>(null);

  const find = async (id = d.id) => {
    setBusy("find");
    setError(null);
    try {
      setBad(id, await postJson<BadPixels>(`/api/datasets/${id}/4d/bad-pixels`, { hot, dead }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };
  // Leaving the dataset (or this process) takes the rings off.
  useEffect(() => () => setBad(d.id, null), [d.id, setBad]);

  const repair = async () => {
    setBusy("repair");
    setError(null);
    try {
      const r = await postJson<{ file: FileSummary; id: string; pixels: number }>(
        `/api/datasets/${d.id}/4d/bad-pixels/repair`, { hot, dead, in_place: inPlace },
      );
      useDatasets.getState().replaceFile(r.file);
      setRepaired(r.pixels);
      if (r.id === d.id) await find(d.id);
      else setBad(d.id, null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const count = found ? found.hot.length + found.dead.length : 0;
  return (
    <section className="q-form">
      <label title="Hot: brighter than its neighbours by this × the mean (0: off)">
        <span>Hot (× mean)</span>
        <input type="number" min={0} step={1} value={hot} onChange={(e) => setHot(Math.max(0, Number(e.target.value) || 0))} />
      </label>
      <label title="Dead: dimmer than this × its neighbours (0: off)">
        <span>Dead (×)</span>
        <input type="number" min={0} max={1} step={0.05} value={dead} onChange={(e) => setDead(Math.min(1, Math.max(0, Number(e.target.value) || 0)))} />
      </label>
      {found && (
        <p className="small">
          {found.hot.length} hot · {found.dead.length} dead
          {found.border > 0 && <span className="muted"> · {found.border} on the border, left as they are</span>}
        </p>
      )}
      {repaired !== null && <p className="small muted">{repaired} repaired</p>}
      <InPlaceChoice inPlace={inPlace} onChange={setInPlace} disabled={!!busy} what={d.title} />
      <div className="rebin-warning" role="note" hidden={!inPlace}>
        Overwrites the loaded data; the file is unchanged.
      </div>
      {error && <p className="field-error">{error}</p>}
      <div className="process-actions">
        <button onClick={() => find()} disabled={!!busy} title="Find them in the mean pattern and ring them on the pattern">
          {busy === "find" ? "Finding…" : "Find"}
        </button>
        <button className="primary" onClick={repair} disabled={!!busy || !count} title="Replace them in every pattern by the mean of the 8 around">
          {busy === "repair" ? "Repairing…" : "Repair"}
        </button>
      </div>
    </section>
  );
}
