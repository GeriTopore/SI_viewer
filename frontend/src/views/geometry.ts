import type { Axis, DatasetSummary } from "../api/client";
import type { AxisCal } from "../components/ImageView";

export interface ImageGeometry {
  width: number;
  height: number;
  calX: AxisCal;
  calY: AxisCal;
}

const cal = (a: Axis | undefined): AxisCal => ({
  scale: a?.scale ?? 1,
  offset: a?.offset ?? 0,
  units: a?.units ?? "",
});

/** Size and calibration of the 2D plane shown for a dataset.
 *
 * Axes come in array order, so the last two of a group are (y, x). Images and
 * stacks show their signal axes; spectrum images show their navigation axes.
 */
export function imageGeometry(d: DatasetSummary, plane: "signal" | "navigation"): ImageGeometry {
  const axes = d.axes.filter((a) => a.navigate === (plane === "navigation"));
  const [y, x] = axes.slice(-2);
  return { width: x?.size ?? 0, height: y?.size ?? 0, calX: cal(x), calY: cal(y) };
}

/** Energy (or other) axis of a spectrum or spectrum image. */
export function spectralAxis(d: DatasetSummary): Axis | undefined {
  return d.axes.filter((a) => !a.navigate).at(-1);
}

export const isAbort = (e: unknown) => (e as Error)?.name === "AbortError";
