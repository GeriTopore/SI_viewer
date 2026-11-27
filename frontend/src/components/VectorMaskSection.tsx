import { useEffect, useState } from "react";
import { create } from "zustand";
import { api, dataVersion, type DatasetSummary, type JobSummary } from "../api/client";
import { allDatasets, useDatasets } from "../state/datasets";
import { useLayers } from "../state/layers";
import { runJob } from "../state/strain";
import { InPlaceChoice, useInPlace } from "./InPlaceChoice";
import { JobProgress } from "./StrainSection";

const ACTIVE = new Set(["queued", "running"]);

interface Magnitudes {
  k: number[];
  counts: number[];
  units: string;
  reflections: { structure: string; hkl: number[]; k: number; intensity: number }[];
}

/** The Structures open (opened .cif files). */
function useStructures(): DatasetSummary[] {
  const files = useDatasets((s) => s.files);
  return allDatasets(files).filter((x) => !!x.structure);
}

const hklText = (hkl: number[]) => hkl.map((h) => (h < 0 ? `${-h}̅` : `${h}`)).join("");

/** pyxem's histogram of the Bragg vectors' magnitudes, with a Structure's
 * reflections (diffsims' powder pattern) marked: does the phase fit, and the
 * calibration (its tutorial's check before template matching)? ``d``: the
 * 4D-STEM data; ``vectors``: its Bragg vectors. */
export function VectorMagnitudesSection({ d, vectors }: { d: DatasetSummary; vectors: DatasetSummary }) {
  const structures = useStructures();
  const [structure, setStructure] = useState<string>("");
  const [data, setData] = useState<Magnitudes | null>(null);
  const [error, setError] = useState<string | null>(null);
  const version = `${vectors.id}:${dataVersion(vectors)}:${vectors.disks?.count ?? 0}`;
  const chosen = structures.find((s) => s.id === structure) ? structure : "";

  useEffect(() => {
    let live = true;
    fetch(`/api/datasets/${d.id}/4d/disks/magnitudes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bins: 100, structures: chosen ? [chosen] : [] }),
    })
      .then(async (r) => {
        const body = await r.json();
        if (!r.ok) throw new Error(body.detail ?? r.statusText);
        if (live) setData(body as Magnitudes), setError(null);
      })
      .catch((e) => live && setError((e as Error).message));
    return () => {
      live = false;
    };
  }, [version, chosen]);

  return (
    <section className="q-form virtual-form">
      <label title="Mark its reflections (diffsims' powder pattern) on the histogram">
        <span>Structure</span>
        <select value={chosen} onChange={(e) => setStructure(e.target.value)}>
          <option value="">None</option>
          {structures.map((s) => <option key={s.id} value={s.id}>{s.title}</option>)}
        </select>
      </label>
      {error && <p className="field-error">{error}</p>}
      {data && <MagnitudePlot data={data} />}
      {chosen && data && !data.reflections.length && !error && (
        <p className="muted small">Calibrate the pattern's pixel size (1/Å or 1/nm) to mark the reflections.</p>
      )}
    </section>
  );
}

function MagnitudePlot({ data }: { data: Magnitudes }) {
  const W = 260, H = 130, L = 4, B = 18, T = 34;
  const n = data.k.length;
  if (!n) return <p className="muted small">No vectors.</p>;
  const step = n > 1 ? data.k[1] - data.k[0] : 1;
  const k0 = data.k[0] - step / 2, k1 = data.k[n - 1] + step / 2;
  const top = Math.max(1, ...data.counts);
  const x = (k: number) => L + ((k - k0) / (k1 - k0)) * (W - 2 * L);
  const y = (c: number) => H - B - (c / top) * (H - B - T);
  const bars = data.counts.map((c, i) => `M${x(data.k[i] - step / 2).toFixed(1)},${y(0)}V${y(c).toFixed(1)}H${x(data.k[i] + step / 2).toFixed(1)}V${y(0)}`).join("");
  const ticks = niceTicks(k0, k1, 4);
  return (
    <svg className="magnitude-plot" viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Histogram of the vectors' magnitudes">
      <path d={bars} className="magnitude-bars" />
      <line x1={L} x2={W - L} y1={y(0)} y2={y(0)} className="magnitude-axis" />
      {ticks.map((t) => (
        <text key={t} x={x(t)} y={H - 5} className="magnitude-tick" textAnchor="middle">{+t.toPrecision(3)}</text>
      ))}
      <text x={L} y={9} className="magnitude-tick">k ({data.units})</text>
      {data.reflections.filter((r) => r.k >= k0 && r.k <= k1).map((r) => (
        <g key={`${r.structure}${r.hkl.join(",")}`}>
          <title>{`${r.structure} ${hklText(r.hkl)}: ${r.k.toPrecision(4)} ${data.units}`}</title>
          <line x1={x(r.k)} x2={x(r.k)} y1={T - 2} y2={y(0)} className="magnitude-reflection" />
          <text x={x(r.k)} y={T - 4} className="magnitude-hkl" transform={`rotate(-90 ${x(r.k)} ${T - 4})`}>{hklText(r.hkl)}</text>
        </g>
      ))}
    </svg>
  );
}

function niceTicks(a: number, b: number, count: number): number[] {
  const raw = (b - a) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let t = Math.ceil(a / step) * step; t <= b; t += step) out.push(t);
  return out;
}

interface MaskState {
  radius: Record<string, number>;
  jobs: Record<string, JobSummary | undefined>;
  errors: Record<string, string | null>;
  run: (id: string, inPlace: boolean) => Promise<void>;
  cancel: (id: string) => Promise<void>;
}

export const DEFAULT_MASK_RADIUS = 2.1;

export const useVectorMask = create<MaskState>((set, get) => ({
  radius: {},
  jobs: {},
  errors: {},
  run: async (id, inPlace) => {
    set((s) => ({ errors: { ...s.errors, [id]: null } }));
    try {
      const job = await runJob(
        `/api/datasets/${id}/4d/disks/mask`,
        { disk_r: get().radius[id] ?? DEFAULT_MASK_RADIUS, in_place: inPlace },
        (j) => set((s) => ({ jobs: { ...s.jobs, [id]: j } })),
      );
      if (job.status === "done" && job.result) {
        useDatasets.getState().replaceFile(job.result.file);
        useLayers.getState().notify(job.result.id === id ? "Masked by its vectors." : "Masked by its vectors: a copy added under the data.");
      }
    } catch (e) {
      set((s) => ({ errors: { ...s.errors, [id]: (e as Error).message } }));
    }
  },
  cancel: async (id) => {
    const job = get().jobs[id];
    if (job) set((s) => ({ jobs: { ...s.jobs, [id]: undefined } })), await api.cancelJob(job.id);
  },
}));

/** The data masked by its Bragg vectors (pyxem's ``to_mask``: discs at the
 * vectors, × the patterns), as its orientation mapping tutorial does before
 * matching. ``d``: the 4D-STEM data. */
export function VectorMaskSection({ d }: { d: DatasetSummary }) {
  const radius = useVectorMask((s) => s.radius[d.id] ?? DEFAULT_MASK_RADIUS);
  const job = useVectorMask((s) => s.jobs[d.id]);
  const error = useVectorMask((s) => s.errors[d.id] ?? null);
  const [inPlace, setInPlace] = useInPlace("vector-mask");
  const running = !!job && ACTIVE.has(job.status);
  return (
    <section className="q-form virtual-form">
      <label title="Each vector's disc (pattern px): kept, the rest of the pattern set to 0">
        <span>Disc radius (px)</span>
        <input
          type="number" min={0.5} max={100} step={0.1} value={radius}
          onChange={(e) => {
            const v = Number(e.target.value);
            if (e.target.value !== "" && Number.isFinite(v) && v > 0) {
              useVectorMask.setState((s) => ({ radius: { ...s.radius, [d.id]: Math.min(100, v) } }));
            }
          }}
        />
      </label>
      <InPlaceChoice inPlace={inPlace} onChange={setInPlace} disabled={running} what={d.title} />
      <div className="rebin-warning" role="note" hidden={!inPlace}>
        Overwrites the loaded data; the file is unchanged.
      </div>
      {error && <p className="field-error">{error}</p>}
      {running ? (
        <JobProgress job={job!} cancel={() => void useVectorMask.getState().cancel(d.id)} />
      ) : (
        <div className="process-actions">
          <button className="primary" onClick={() => void useVectorMask.getState().run(d.id, inPlace)} title="Mask every pattern by its vectors">
            Mask
          </button>
        </div>
      )}
      {job?.status === "failed" && !running && <p className="field-error">{job.error}</p>}
    </section>
  );
}
