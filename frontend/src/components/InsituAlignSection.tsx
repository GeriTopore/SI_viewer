import { useEffect, useState } from "react";
import { api, type AlignmentSource, type DatasetSummary, type ShiftsCheck, type ShiftsChoice } from "../api/client";
import { useDatasets } from "../state/datasets";
import { reducing, useInsitu } from "../state/insitu";
import { formatValue } from "../format";

/** The bytes of ``file`` as base64. */
async function base64Of(file: File): Promise<string> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let text = "";
  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(text);
}

/** Process tab, in-situ spectrum-image series: register every frame with the
 * drift measured on the simultaneous ADF stack (its applied alignment, taken
 * by default) or with shifts imported from a .npy file, and sum them into a
 * new spectrum image. The shifts are checked against the series first. */
export function InsituAlignSection({ d }: { d: DatasetSummary }) {
  const [info, setInfo] = useState<AlignmentSource | null>(null);
  const [source, setSource] = useState<"adf" | "npy">("adf");
  const [file, setFile] = useState<{ name: string; data: string } | null>(null);
  const [units, setUnits] = useState<"adf" | "scan">("adf");
  const [check, setCheck] = useState<ShiftsCheck | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const job = useInsitu((s) => s.jobs[d.id]);
  const running = reducing(job);
  const summing = running && job?.kind === "aligned-sum";
  // The ADF's alignment changes in the dataset list: ask again when it does.
  const adfRevision = useDatasets((s) => {
    const adf = info?.adf ? s.files.flatMap((f) => f.datasets).find((x) => x.id === info.adf!.id) : undefined;
    return adf ? `${adf.revision}.${adf.filter_rev}.${adf.alignment ? 1 : 0}` : "";
  });

  useEffect(() => {
    api
      .insituAlignment(d.id)
      .then((i) => {
        setInfo(i);
        if (!i.adf) {
          setSource("npy");
          setUnits("scan");
        }
      })
      .catch((e) => setProblem((e as Error).message));
  }, [d.id, adfRevision]);

  const choice: ShiftsChoice | null =
    source === "adf" ? { source } : file ? { source, npy: file.data, units } : null;
  const choiceKey = JSON.stringify(choice && { ...choice, npy: file?.name });

  // Check the shifts whenever the choice changes.
  useEffect(() => {
    setCheck(null);
    setProblem(null);
    if (!choice || !info) return;
    let live = true;
    api
      .checkShifts(d.id, choice)
      .then((c) => live && setCheck(c))
      .catch((e) => live && setProblem((e as Error).message));
    return () => {
      live = false;
    };
  }, [choiceKey, info, d.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const adf = info?.adf ?? null;
  const progress = summing ? Math.round(job!.progress * 100) : 0;

  return (
    <section className="q-form">
      <div className="q-row">
        <span>Shifts</span>
        <div className="segmented process-choice" role="group" aria-label="Shifts">
          <button
            className={source === "adf" ? "active" : undefined}
            aria-pressed={source === "adf"}
            disabled={!adf || summing}
            onClick={() => setSource("adf")}
            title={adf ? `The alignment applied to ${adf.title}` : "No simultaneous ADF stack on this scan"}
          >
            ADF alignment
          </button>
          <button
            className={source === "npy" ? "active" : undefined}
            aria-pressed={source === "npy"}
            disabled={summing}
            onClick={() => setSource("npy")}
            title="Shifts from a .npy file: (frames, 2), the x and y shift that aligns each frame; NaN leaves a frame out"
          >
            Import .npy
          </button>
        </div>
      </div>
      {source === "adf" && adf && (
        <p className="muted small q-elements">
          {adf.title}
          {adf.k > 1 && ` · ${adf.k}× finer than the scan`}
        </p>
      )}
      {source === "npy" && (
        <>
          <label title="A .npy file of shifts">
            <span>File</span>
            <input
              type="file"
              accept=".npy"
              disabled={summing}
              onChange={async (e) => {
                const f = e.target.files?.[0];
                setFile(f ? { name: f.name, data: await base64Of(f) } : null);
              }}
            />
          </label>
          <label title="The pixels the shifts were measured in">
            <span>Pixels of</span>
            <select value={units} onChange={(e) => setUnits(e.target.value as "adf" | "scan")} disabled={summing}>
              {adf && (
                <option value="adf">
                  the ADF{adf.k > 1 ? ` (÷${adf.k})` : ""}
                </option>
              )}
              <option value="scan">the scan</option>
            </select>
          </label>
        </>
      )}
      {check && (
        <p className="muted small q-elements">
          {check.frames} of {info?.frames} frames{check.left_out ? ` (${check.left_out} left out)` : ""} · largest shift{" "}
          {formatValue(check.largest)} px · keeps {check.width} × {check.height} px
        </p>
      )}
      {problem && <p className="field-error">{problem}</p>}
      {summing ? (
        <div className="q-job">
          <div className="q-progress" role="progressbar" aria-valuenow={progress}>
            <div style={{ width: `${progress}%` }} />
          </div>
          <div className="q-job-row small">
            <span>
              {job!.message || "Starting…"} {progress}%
            </span>
            <button className="link-button" onClick={() => useInsitu.getState().cancel(d.id)}>
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <button
          className="primary"
          disabled={!check || running}
          onClick={() => choice && useInsitu.getState().alignedSum(d.id, choice)}
          title="Shift every frame into register and add them up (a sum, for the counts): a new spectrum image"
        >
          Align and sum
        </button>
      )}
    </section>
  );
}
