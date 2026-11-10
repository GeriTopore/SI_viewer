import type { Frame } from "../../api/arrays";
import type { Interaction, Pt, Transform } from "../../components/ImageView";
import type { ImageGeometry } from "../geometry";

/**
 * Navigators finer than the scan (Gatan "sub-scan": an ADF with k × k image
 * pixels per spectrum-image pixel, over the same field of view).
 *
 * The navigator view works in the navigator image's own pixels. Everything
 * about spectra (cursor, regions, spectrum requests, stored state) stays in
 * spectrum-image ("SI") pixels. The one bridge is the factor ``k``:
 *
 *   SI coordinate = image coordinate / k        (k = 1: the identity)
 *
 * applied to pointer positions on the way in (:func:`scanInteraction`) and to
 * the drawing transform on the way out (:func:`scanTransform`), so the cursor
 * and region code never sees image pixels. The maps drawn over the navigator
 * (SI-sized) are stretched by k, nearest neighbour, by giving their frames
 * ``bin = k`` (:func:`stretch`), which the renderer already draws that way.
 */

/** SI pixel (index) containing image coordinate ``v``. */
export const scanPixel = (v: number, k: number) => Math.floor(v / k);

/** A point in image pixels as a point in SI pixels. */
export const toScan = (p: Pt, k: number): Pt => (k === 1 ? p : { x: p.x / k, y: p.y / k });

/** The drawing transform in SI pixels: ``toScreen(x, y)`` takes SI
 * coordinates, ``zoom`` is screen px per SI pixel. */
export function scanTransform(t: Transform, k: number): Transform {
  if (k === 1) return t;
  return { zoom: t.zoom * k, toScreen: (x, y) => t.toScreen(x * k, y * k) };
}

/** ``i`` working in SI pixels while the view reports image pixels. */
export function scanInteraction(i: Interaction, k: number): Interaction {
  if (k === 1) return i;
  return {
    ...i,
    onDown: i.onDown && ((p, e, t) => i.onDown!(toScan(p, k), e, scanTransform(t, k))),
    onMove: i.onMove && ((p, e, t) => i.onMove!(toScan(p, k), e, scanTransform(t, k))),
    onUp: i.onUp && ((p, e, t) => i.onUp!(toScan(p, k), e, scanTransform(t, k))),
    onHover: i.onHover && ((p, t) => i.onHover!(p && toScan(p, k), t && scanTransform(t, k))),
    onDoubleClick: i.onDoubleClick && ((p) => i.onDoubleClick!(toScan(p, k))),
  };
}

// A stretched frame is the same object for the same frame, so the renderer
// does not upload it again at every render.
const stretched = new WeakMap<Frame, Map<number, Frame>>();

/** ``frame`` (SI-sized) drawn over a navigator k times finer: each of its
 * pixels covers k × k navigator pixels. The data is shared, not resampled. */
export function stretch(frame: Frame, k: number): Frame {
  if (k === 1) return frame;
  let byK = stretched.get(frame);
  if (!byK) stretched.set(frame, (byK = new Map()));
  let out = byK.get(k);
  if (!out) byK.set(k, (out = { ...frame, bin: frame.bin * k }));
  return out;
}

/** Size and calibration of the navigator view: the navigator image's own
 * (``image``), else the scan's refined k times. */
export function navigatorGeometry(scan: ImageGeometry, k: number, image?: ImageGeometry): ImageGeometry {
  if (k === 1) return scan;
  if (image && image.width === scan.width * k && image.height === scan.height * k) return image;
  const finer = (c: ImageGeometry["calX"]) => ({ ...c, scale: c.scale / k });
  return { width: scan.width * k, height: scan.height * k, calX: finer(scan.calX), calY: finer(scan.calY) };
}

/** A polygon in image pixels as one in SI pixels. */
export const polygonToScan = (points: [number, number][], k: number): [number, number][] =>
  k === 1 ? points : points.map(([x, y]) => [x / k, y / k]);
