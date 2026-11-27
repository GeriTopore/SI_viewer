import type { AppliedAlignment, DatasetSummary } from "../api/client";
import { alignmentNpyUrl, alignmentShiftsUrl } from "../api/client";
import { downloadUrl } from "../download";
import { formatValue } from "../format";
import { maskLabel } from "../viewer/fourierMask";
import { DriftPlot } from "./DriftPlot";
import type { AxisCal } from "./ImageView";

const METHODS = { pixel: "brightest pixel", gf: "Gaussian fit", com: "centre of mass" };
const CORRELATIONS = { cc: "cross-correlation", mc: "mutual correlation", pc: "phase correlation" };

/** The Alignment tab under a drift-corrected stack: the measured drift and
 * how the correction was made. */
export function AlignmentSummary({ ds, calX }: { ds: DatasetSummary; calX: AxisCal }) {
  const a = ds.alignment as AppliedAlignment;
  const s = a.settings;
  const left = a.loaded_frames - a.frames.length;
  const outliers =
    s.outliers.method === "transitivity"
      ? `transitivity, threshold ${formatValue(s.outliers.threshold)} px, ${s.outliers.max_paths} paths`
      : s.outliers.method === "neighbours"
        ? `neighbours, max shift ${formatValue(s.outliers.max_shift)} px`
        : "none";
  const edits = s.edits ? { masked: s.edits.mask.length, kept: s.edits.keep.length } : null;
  return (
    <div className="alignment-summary">
      <DriftPlot drift={a.drift} calX={calX} />
      <dl className="alignment-facts">
        <dt>Frames</dt>
        <dd>
          {a.frames.length} of {a.loaded_frames}
          {left > 0 && <span className="muted"> ({left} left out: outside the range or bad)</span>}
        </dd>
        <dt>Crop</dt>
        <dd>
          {a.square[2]} × {a.square[3]} px at ({a.square[0]}, {a.square[1]})
        </dd>
        <dt>Fourier mask</dt>
        <dd>{maskLabel(s.mask)}</dd>
        <dt>Shifts</dt>
        <dd>
          {CORRELATIONS[s.peaks.correlation]}, {METHODS[s.peaks.method]}
        </dd>
        <dt>Outliers</dt>
        <dd>
          {outliers}
          {s.outliers.off_diagonal ? `; pairs at most ${s.outliers.off_diagonal} frames apart` : ""}
        </dd>
        {edits && edits.masked + edits.kept > 0 && (
          <>
            <dt>Manual edits</dt>
            <dd>
              {edits.masked} masked, {edits.kept} kept
            </dd>
          </>
        )}
        <dt>Save</dt>
        <dd className="align-save">
          <button onClick={() => downloadUrl(alignmentShiftsUrl(ds.id))}>Shifts (CSV)</button>
          <button
            onClick={() => downloadUrl(alignmentNpyUrl(ds.id))}
            title="(frames, 2) shifts in this stack's pixels; NaN: frame left out"
          >
            Shifts (.npy)
          </button>
        </dd>
      </dl>
    </div>
  );
}
