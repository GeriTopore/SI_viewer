import { useEffect, useMemo, useRef } from "react";
import { fetchCompareN, fetchCorrelation, fetchRegistrationFrame, type Frame } from "../api/arrays";
import type { DatasetSummary } from "../api/client";
import { ArrayPanel } from "../components/ArrayPanel";
import type { Transform } from "../components/ImageView";
import { ALIGN_STEPS, useAlignment, useStackAlignment, type Alignment } from "../state/alignment";
import { formatValue } from "../format";
import { useViewerImage } from "../state/viewerImage";
import { maskLabel } from "../viewer/fourierMask";
import { PANE_SIZES, useUi } from "../state/ui";
import { Splitter } from "../components/Splitter";
import { imageGeometry } from "./geometry";
import { useLatest } from "./useLatest";
import { MatricesStep } from "./AlignmentMatrices";
import { PeaksStep } from "./AlignmentPeaks";
import { ResultStep } from "./AlignmentResult";

/** Guided drift correction of a stack, filling the viewer: a step bar, then
 * the current step's diagnostics. Its parameters are set in the Process tab. */
export function AlignmentWorkspace({ ds }: { ds: DatasetSummary }) {
  const a = useStackAlignment(ds.id);
  const { patch, close } = useAlignment.getState();
  if (!a) return null;
  const index = ALIGN_STEPS.findIndex((s) => s.id === a.step);
  const go = (k: number) => {
    const step = ALIGN_STEPS[k];
    if (step?.ready) patch(ds.id, { step: step.id });
  };

  return (
    <div className="view align-workspace">
      <header className="align-header">
        <div className="align-title">
          <strong>Align frames</strong>
        </div>
        <ol className="align-steps">
          {ALIGN_STEPS.map((s, k) => (
            <li key={s.id}>
              <button
                className={s.id === a.step ? "active" : k < index ? "done" : undefined}
                disabled={!s.ready}
                onClick={() => go(k)}
                title={s.ready ? undefined : "Not available yet (next milestones)"}
              >
                <span className="align-step-number">{k + 1}</span>
                {s.label}
              </button>
            </li>
          ))}
        </ol>
        <div className="align-nav">
          <button className="secondary small-button" onClick={() => go(index - 1)} disabled={index === 0}>
            Back
          </button>
          <button
            className="primary small-button"
            onClick={() => go(index + 1)}
            disabled={!ALIGN_STEPS[index + 1]?.ready || !a.session}
          >
            Next
          </button>
          <button className="link-button" onClick={close} title="Back to the stack viewer (the registration is kept)">
            Close
          </button>
        </div>
      </header>
      {a.error && <div className="align-error">{a.error}</div>}
      {!a.session ? (
        <div className="placeholder">
          <p className="muted">
            <span className="spinner" aria-hidden /> Taking the FFTs of {a.last - a.first} frames…
          </p>
        </div>
      ) : a.step === "mask" ? (
        <MaskStep ds={ds} a={a} />
      ) : a.step === "peaks" ? (
        <PeaksStep ds={ds} a={a} />
      ) : a.step === "matrices" ? (
        <MatricesStep ds={ds} a={a} />
      ) : (
        <ResultStep ds={ds} a={a} />
      )}
      {a.starting && a.session && <div className="align-busy">Updating the frames…</div>}
    </div>
  );
}

// ---------- step 1: Fourier mask ----------

/** One frame of the stack, as loaded. It feeds the FFT panel (top right),
 * so the FFT of the frame in view is shown there, with the mask on it. */
function FramePanel({ ds, a, frame, title }: { ds: DatasetSummary; a: Alignment; frame: number; title: string }) {
  const g = imageGeometry(ds, "signal");
  const shown = useLatest(`${a.sessionStamp}:${frame}`, () => fetchRegistrationFrame(ds.id, frame));
  return (
    <ArrayPanel
      viewKey={`align:${ds.id}:${a.step}:frame`} // each step's own zoom (panel sizes differ)
      title={title}
      layers={[{ key: "frame", label: `Frame ${frame}`, frame: shown.value }]}
      version="frame" // one contrast for all frames, so they compare
      width={g.width}
      height={g.height}
      calX={g.calX}
      calY={g.calY}
      loading={shown.value ? null : "Loading frame…"}
      error={shown.error}
      publish
    />
  );
}

/** Position of a centred cross-correlation's maximum: the shift (px). */
function peakOf(cc: Frame | null): { dx: number; dy: number; x: number; y: number } | null {
  if (!cc) return null;
  let best = 0;
  for (let k = 1; k < cc.data.length; k++) if (cc.data[k] > cc.data[best]) best = k;
  const x = best % cc.width;
  const y = Math.floor(best / cc.width);
  return { x, y, dx: x - Math.floor(cc.width / 2), dy: y - Math.floor(cc.height / 2) };
}

/** The zero-shift centre (crosshair) and the peak (circle) of a cross-correlation. */
function peakMarker(peak: { x: number; y: number } | null, width: number, height: number) {
  return (t: Transform) => {
    const [cx, cy] = t.toScreen(Math.floor(width / 2) + 0.5, Math.floor(height / 2) + 0.5);
    return (
      <g className="align-peak">
        <path d={`M${cx - 6},${cy} h12 M${cx},${cy - 6} v12`} className="align-centre" />
        {peak &&
          (() => {
            const [px, py] = t.toScreen(peak.x + 0.5, peak.y + 0.5);
            return <circle cx={px} cy={py} r={7} />;
          })()}
      </g>
    );
  };
}

function CorrelationPanel({
  ds, viewKey, title, cc, version, loading, error,
}: {
  ds: DatasetSummary;
  viewKey: string;
  title: string;
  cc: Frame | null;
  version: string;
  loading?: boolean;
  error?: string | null;
}) {
  const g = imageGeometry(ds, "signal");
  const peak = useMemo(() => peakOf(cc), [cc]);
  return (
    <ArrayPanel
      viewKey={viewKey}
      title={title}
      layers={[{ key: "cc", label: "Cross-correlation", frame: cc, colormap: "viridis" }]}
      version={version}
      width={g.width}
      height={g.height}
      loading={cc && !loading ? null : "Computing…"}
      error={error}
      renderOverlay={peakMarker(peak, g.width, g.height)}
    />
  );
}

function MaskStep({ ds, a }: { ds: DatasetSummary; a: Alignment }) {
  const [i, j] = a.pair;
  const preview = useLatest(`${a.sessionStamp}:${i}:${j}:${JSON.stringify(a.mask)}`, () =>
    fetchCorrelation(ds.id, i, j, a.mask),
  );
  const compareKey = a.compare ? `${a.sessionStamp}:${i}:${j}:${JSON.stringify(a.mask)}:${a.compareNs.join(",")}` : null;
  const compare = useLatest(compareKey, () => fetchCompareN(ds.id, i, j, a.mask, a.compareNs));

  // The mask is drawn on the FFT panel (top right) while this step is shown.
  const setFftMask = useViewerImage((s) => s.setFftMask);
  useEffect(() => {
    setFftMask(a.mask);
  }, [a.mask, setFftMask]);
  useEffect(() => () => setFftMask(null), [setFftMask]);
  // Dragging the cutoff ring on the FFT changes the mask.
  const setEditFftMask = useViewerImage((s) => s.setEditFftMask);
  useEffect(() => {
    setEditFftMask((mask) => useAlignment.getState().patch(ds.id, { mask }));
    return () => setEditFftMask(null);
  }, [ds.id, setEditFftMask]);

  const compareFraction = useUi((s) => s.compareFraction);
  const setCompareFraction = useUi((s) => s.setCompareFraction);
  const bodyRef = useRef<HTMLDivElement>(null);

  return (
    <div ref={bodyRef} className="align-body align-mask">
      <div className="align-mask-row">
        <FramePanel ds={ds} a={a} frame={i} title={`Frame ${i}`} />
        <CorrelationPanel
          ds={ds}
          viewKey={`align:${ds.id}:mask-cc`}
          title={`Cross-correlation ${i} × ${j} (${maskLabel(a.mask)})`}
          cc={preview.value}
          version={preview.key ?? ""}
          loading={preview.loading && !preview.value}
          error={preview.error}
        />
      </div>
      {a.compare && (
        <Splitter
          direction="row"
          className="viewer-resizer"
          label="Resize the compared cross-correlations"
          onDrag={(y) => {
            const r = bodyRef.current!.getBoundingClientRect();
            setCompareFraction((r.bottom - y) / r.height);
          }}
          onReset={() => setCompareFraction(PANE_SIZES.compare.initial)}
        />
      )}
      {a.compare && (
        <div className="align-compare-row" style={{ flexBasis: `${compareFraction * 100}%` }}>
          {a.compareNs.map((cn, k) => (
            <CorrelationPanel
              key={`${cn}:${k}`}
              ds={ds}
              viewKey={`align:${ds.id}:compare:${k}`}
              title={`n = ${formatValue(cn)}`}
              cc={compare.value?.[k] ?? null}
              version={compare.key ?? ""}
              error={compare.error}
            />
          ))}
        </div>
      )}
    </div>
  );
}
