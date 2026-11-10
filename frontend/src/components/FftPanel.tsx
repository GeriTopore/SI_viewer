import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useUi, type FftScale } from "../state/ui";
import { useViewerImage, type ViewerImage } from "../state/viewerImage";
import type { FftRequest, FftResult } from "../viewer/fft.worker";
import { formatValue, inAngstrom } from "../format";
import type { MaskSettings } from "../api/client";
import type { Frame } from "../api/arrays";
import { bandpassValue, maskCutoff, maskValue, type Bandpass } from "../viewer/fourierMask";
import { applyFilters, useFilterTarget } from "../state/filters";
import { colormapLut, isColour } from "../viewer/colormaps";
import { percentile, sortedSample } from "../viewer/contrast";
import type { DisplayParams } from "../viewer/renderer";
import { ContrastControls } from "./ContrastControls";
import { useFftBoxes, useViewFftBoxes, type FftBox } from "../state/fftBoxes";

// Largest patch transformed; a bigger visible area uses its centre.
const MAX_PATCH = 2048;
const MIN_PATCH = 16;
const ZOOMS = [1, 2, 4, 8];
// Wait for panning/zooming to pause this long before a new FFT.
const DELAY_MS = 120;

/** What was transformed: a square patch of the viewer's image. */
interface Patch {
  label: string;
  /** Patch size in full-resolution pixels, and the padded FFT size. */
  size: number;
  n: number;
  /** Pixel size of the patch along x and y (Å for lengths, else its units or px). */
  px: number;
  py: number;
  units: string;
  /** Frame pixels per patch pixel (a binned preview frame). */
  bin: number;
  /** The patch was cut from a larger area. */
  cropped: boolean;
  /** The FFT box it was cut from (null: the part of the image in view). */
  region: string | null;
}

interface Shown {
  patch: Patch;
  result: FftResult;
}

const nextPow2 = (v: number) => 2 ** Math.ceil(Math.log2(v));

/** The centred square of the visible area (or of an FFT box), as frame pixels. */
function cutPatch(img: ViewerImage, box: FftBox | null): { data: Float32Array; patch: Patch } | null {
  const { frame, calX, calY } = img;
  const b = frame.bin;
  const area = box ? { x0: box.x, y0: box.y, x1: box.x + box.size, y1: box.y + box.size } : img;
  const fx0 = Math.max(0, Math.floor(area.x0 / b));
  const fy0 = Math.max(0, Math.floor(area.y0 / b));
  const fx1 = Math.min(frame.width, Math.ceil(area.x1 / b));
  const fy1 = Math.min(frame.height, Math.ceil(area.y1 / b));
  const side = Math.min(fx1 - fx0, fy1 - fy0);
  const size = Math.min(side, MAX_PATCH);
  if (size < MIN_PATCH) return null;
  const sx = fx0 + Math.floor((fx1 - fx0 - size) / 2);
  const sy = fy0 + Math.floor((fy1 - fy0 - size) / 2);
  const data = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    const row = (sy + y) * frame.width + sx;
    data.set(frame.data.subarray(row, row + size), y * size);
  }
  const calibrated = !!calX.units;
  const x = calibrated ? inAngstrom(calX.scale, calX.units) : { size: 1, units: "px" };
  const y = calibrated ? inAngstrom(calY.scale, calY.units || calX.units) : { size: 1, units: "px" };
  return {
    data,
    patch: {
      label: box ? `${img.label} · ${box.name}` : img.label,
      size,
      n: nextPow2(size),
      px: x.size * b,
      py: y.size * b,
      units: x.units,
      bin: b,
      cropped: fx1 - fx0 > size || fy1 - fy0 > size,
      region: box?.name ?? null,
    },
  };
}

/** Power spectrum (FFT) of the visible part of the viewer's image. Clicking
 * marks a spatial frequency and gives the matching real-space spacing; the
 * mark stays until moved or hidden. */
export function FftPanel() {
  const open = useUi((s) => s.fftOpen);
  const setOpen = useUi((s) => s.setFftOpen);
  const image = useViewerImage((s) => s.image);
  const fftMask = useViewerImage((s) => s.fftMask);
  const showMask = useViewerImage((s) => s.showFftMask);
  const setShowMask = useViewerImage((s) => s.setShowFftMask);
  const measure = useUi((s) => s.fftMeasure);
  const setMeasure = useUi((s) => s.setFftMeasure);
  const hann = useUi((s) => s.fftHann);
  const setHann = useUi((s) => s.setFftHann);
  const [zoom, setZoom] = useState(1);
  // FFT boxes drawn on the viewer's image: the FFT is of one of them, or of the view.
  const fftView = image?.fftView;
  const { boxes, source } = useViewFftBoxes(fftView);
  const box = boxes.find((b) => b.id === source) ?? null;
  const compare = useViewerImage((s) => s.compare);
  const setCompare = useViewerImage((s) => s.setCompare);
  const setSource = (id: string | null) => fftView && useFftBoxes.getState().patch(fftView, { source: id, selected: id });
  // The focused dataset's bandpass filter, drawn like the alignment's mask
  // (which takes its place while an alignment is shown).
  const target = useFilterTarget();
  const filters = target?.filters;
  const small = filters?.spatial === "bandpass" ? filters.small : null;
  const large = filters?.spatial === "bandpass" ? filters.large : null;
  const band = useMemo(() => (small !== null && large !== null ? { small, large } : null), [small, large]);
  const overlay = fftMask ?? band;
  // Dragging a ring changes the mask (an alignment's) or the bandpass.
  const onMask = useViewerImage((s) => s.editFftMask);
  const targetId = target?.id;
  const onBand = useMemo(
    () =>
      targetId && filters?.spatial === "bandpass"
        ? (b: Bandpass) => applyFilters([targetId], { ...filters, ...b })
        : null,
    [targetId, filters],
  );

  return (
    <section className={`fft-panel${open ? "" : " closed"}`}>
      <div className="fft-header">
        <button
          className="link-button fft-toggle"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          title={open ? "Hide the FFT" : "Show the FFT of the image in the viewer"}
        >
          <span className="fft-chevron" aria-hidden>
            {open ? "▾" : "▸"}
          </span>
          FFT
        </button>
        {open && (
          <button
            className={`fft-mask-toggle${measure ? " active" : ""}`}
            aria-pressed={measure}
            onClick={() => setMeasure(!measure)}
            title={measure ? "Hide the spacing mark (clicks no longer place it)" : "Click the FFT to measure a spacing"}
          >
            Measure
          </button>
        )}
        {open && (
          <button
            className={`fft-mask-toggle${hann ? " active" : ""}`}
            aria-pressed={hann}
            onClick={() => setHann(!hann)}
            title={hann ? "Hann window on (no edge streaks)" : "Hann window off"}
          >
            Hann
          </button>
        )}
        {open && overlay && (
          <button
            className={`fft-mask-toggle${showMask ? " active" : ""}`}
            aria-pressed={showMask}
            onClick={() => setShowMask(!showMask)}
            title={fftMask ? "The alignment's Fourier mask" : "The bandpass filter's cutoffs"}
          >
            {fftMask ? "Mask" : "Bandpass"}
          </button>
        )}
        {open && (
          <div className="segmented fft-zoom" role="group" aria-label="FFT zoom">
            {ZOOMS.map((z) => (
              <button
                key={z}
                className={zoom === z ? "active" : undefined}
                aria-pressed={zoom === z}
                onClick={() => setZoom(z)}
                title={z === 1 ? "Whole FFT (up to the Nyquist frequency)" : `Centre of the FFT, ×${z}`}
              >
                ×{z}
              </button>
            ))}
          </div>
        )}
      </div>
      {open && image && (
        <div className="fft-regions-row">
          {boxes.length > 0 && (
            <div className="segmented fft-regions" role="group" aria-label="FFT region">
              <button
                className={box ? undefined : "active"}
                aria-pressed={!box}
                onClick={() => setSource(null)}
                title="The part of the image in view"
              >
                View
              </button>
              {boxes.map((b) => (
                <button
                  key={b.id}
                  className={b.id === box?.id ? "active" : undefined}
                  aria-pressed={b.id === box?.id}
                  onClick={() => setSource(b.id)}
                  title={`${b.size}×${b.size} px at (${b.x}, ${b.y})`}
                >
                  <span className="fft-region-swatch" style={{ background: b.color }} />
                  {b.name}
                </button>
              ))}
            </div>
          )}
          <button
            className={`fft-mask-toggle${compare ? " active" : ""}`}
            aria-pressed={compare}
            onClick={() => setCompare(!compare)}
            title="The FFTs of the view and of each FFT box, side by side beside the image"
          >
            View in main window
          </button>
        </div>
      )}
      {open && (
        <FftView
          image={image}
          box={box}
          zoom={zoom}
          mask={fftMask && showMask ? fftMask : null}
          band={!fftMask && showMask ? band : null}
          measure={measure}
          onMask={fftMask ? onMask : null}
          onBand={fftMask ? null : onBand}
        />
      )}
    </section>
  );
}

// Colour of the Fourier mask's tint, and its strength where the mask is 1.
const MASK_RGB = [255, 150, 30];
const MASK_ALPHA = 0.5;

// A ring is grabbed within this many screen pixels.
const GRAB_PX = 6;
// Sizes (px) and cutoffs snap to this step while dragging.
const SNAP = 0.1;
// (toFixed drops the float noise, e.g. 101 × 0.1 = 10.100000000000001.)
const snap = (v: number) => Number((Math.round(v / SNAP) * SNAP).toFixed(1));
const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), Math.max(hi, lo));
// The alignment's cutoff n ranges as its slider does; the backend limits the bandpass sizes.
const N_MIN = 1;
const N_MAX = 30;
const BAND_MAX = 10000;

/** Radius of the ellipse (semi-axes rx, ry; first axis at ``angle`` degrees) in the direction (dx, dy). */
function ellipseRadius(rx: number, ry: number, angle: number, dx: number, dy: number): number {
  const t = (angle * Math.PI) / 180;
  const a = dx * Math.cos(t) + dy * Math.sin(t);
  const b = -dx * Math.sin(t) + dy * Math.cos(t);
  const phi = Math.atan2(b, a);
  return (rx * ry) / Math.hypot(ry * Math.cos(phi), rx * Math.sin(phi));
}

/** A resize cursor along the direction (dx, dy) from the centre. */
function resizeCursor(dx: number, dy: number): string {
  const octant = (((Math.round(Math.atan2(dy, dx) / (Math.PI / 4)) % 4) + 4) % 4);
  return ["ew-resize", "nwse-resize", "ns-resize", "nesw-resize"][octant];
}

type Drag =
  | { id: "mask"; start: MaskSettings; rx: number; ry: number; angle: number }
  | { id: "small" | "large" };

export function FftView({
  image, box, zoom, mask, band: bandProp, measure, onMask, onBand, compact = false,
}: {
  image: ViewerImage | null;
  box: FftBox | null;
  /** Without the contrast controls (the FFTs beside the image). */
  compact?: boolean;
  zoom: number;
  mask: MaskSettings | null;
  band: Bandpass | null;
  measure: boolean;
  onMask: ((mask: MaskSettings) => void) | null;
  onBand: ((band: Bandpass) => Promise<void>) | null;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  // The bandpass while its ring is dragged, until the filter has been applied.
  const [liveBand, setLiveBand] = useState<Bandpass | null>(null);
  const liveRef = useRef<Bandpass | null>(null);
  const band = liveBand ?? bandProp;
  const [hover, setHover] = useState<{ id: string; cursor: string } | null>(null);
  const drag = useRef<Drag | null>(null);
  const suppressClick = useRef(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [side, setSide] = useState(0);
  const [shown, setShown] = useState<Shown | null>(null);
  const [tooSmall, setTooSmall] = useState(false);
  const [busy, setBusy] = useState(false);

  // Display: the scale, gamma and colormap persist. The range follows each
  // FFT (median up to the 99.95th percentile) until set by hand; a manual
  // range holds for this image and scale only.
  const hann = useUi((s) => s.fftHann);
  const scale = useUi((s) => s.fftScale);
  const setScale = useUi((s) => s.setFftScale);
  const gamma = useUi((s) => s.fftGamma);
  const setGamma = useUi((s) => s.setFftGamma);
  const colormap = useUi((s) => s.fftColormap);
  const setColormap = useUi((s) => s.setFftColormap);
  const contrastOpen = useUi((s) => s.fftContrastOpen);
  const setContrastOpen = useUi((s) => s.setFftContrastOpen);
  const label = image ? `${image.label}${box ? ` · ${box.id}` : ""}` : null;
  const [manual, setManual] = useState<{ lo: number; hi: number; scale: FftScale; label: string | null } | null>(null);
  const range = manual && manual.scale === scale && manual.label === label ? manual : null;
  useEffect(() => setManual(null), [scale, label]);

  // The displayed values (n × n) and their automatic range; the FFT itself
  // is not redone when only the scale changes.
  const view = useMemo(() => {
    if (!shown) return null;
    const { n, amp } = shown.result;
    let data = amp;
    if (scale === "log") {
      data = new Float32Array(amp.length);
      for (let i = 0; i < amp.length; i++) data[i] = Math.log1p(amp[i] * amp[i]);
    }
    const sample = sortedSample(data);
    const lo = percentile(sample, 50);
    const hi = percentile(sample, 99.95);
    const frame: Frame = { data, width: n, height: n, bin: 1 };
    return { frame, auto: [lo, hi > lo ? hi : lo + 1] as [number, number] };
  }, [shown, scale]);

  const display = useMemo(
    (): DisplayParams => ({
      lo: range?.lo ?? view?.auto[0] ?? 0,
      hi: range?.hi ?? view?.auto[1] ?? 1,
      gamma,
      colormap,
    }),
    [range?.lo, range?.hi, view, gamma, colormap],
  );
  const onDisplay = (patch: Partial<DisplayParams>) => {
    if (patch.lo !== undefined || patch.hi !== undefined) {
      setManual({ lo: patch.lo ?? display.lo, hi: patch.hi ?? display.hi, scale, label });
    }
    if (patch.gamma !== undefined) setGamma(patch.gamma);
    if (patch.colormap !== undefined) setColormap(patch.colormap);
  };
  // The clicked spatial frequency (units⁻¹, x right, y down): kept as a
  // frequency so it stays put as the FFT is redone (pan, zoom, another
  // frame). Cleared when hidden or when the units change.
  const [mark, setMark] = useState<{ kx: number; ky: number } | null>(null);
  const units = shown?.patch.units;
  useEffect(() => setMark(null), [units, measure]);

  // One worker; only the latest request's result is shown.
  const worker = useRef<Worker | null>(null);
  const pending = useRef<{ id: number; patch: Patch } | null>(null);
  const nextId = useRef(0);
  useEffect(() => {
    const w = new Worker(new URL("../viewer/fft.worker.ts", import.meta.url), { type: "module" });
    w.onmessage = (e: MessageEvent<FftResult>) => {
      const p = pending.current;
      if (!p || p.id !== e.data.id) return;
      pending.current = null;
      setBusy(false);
      setShown({ patch: p.patch, result: e.data });
    };
    worker.current = w;
    return () => w.terminate();
  }, []);

  useEffect(() => {
    if (!image) {
      setShown(null);
      setTooSmall(false);
      return;
    }
    const t = setTimeout(() => {
      const cut = cutPatch(image, box);
      setTooSmall(!cut);
      if (!cut) {
        setShown(null);
        return;
      }
      const id = ++nextId.current;
      pending.current = { id, patch: cut.patch };
      setBusy(true);
      const req: FftRequest = { id, patch: cut.data, size: cut.patch.size, n: cut.patch.n, hann };
      worker.current?.postMessage(req, [cut.data.buffer]);
    }, DELAY_MS);
    return () => clearTimeout(t);
  }, [image, hann, box?.x, box?.y, box?.size, box?.name]); // eslint-disable-line react-hooks/exhaustive-deps

  // Square canvas filling the panel width.
  useLayoutEffect(() => {
    const el = wrapRef.current!;
    const ro = new ResizeObserver(() => setSide(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Draw: the central n/zoom square, max-pooled down to the canvas so
  // isolated spots stay visible.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !side) return;
    const dpr = window.devicePixelRatio || 1;
    const d = Math.round(side * dpr);
    canvas.width = d;
    canvas.height = d;
    const ctx = canvas.getContext("2d")!;
    if (!shown || !view) {
      ctx.clearRect(0, 0, d, d);
      return;
    }
    const { n } = shown.result;
    const power = view.frame.data;
    const { lo, hi, gamma: g0, colormap: cmap } = display;
    const lut = colormapLut(cmap);
    const span = n / zoom;
    const start = (n - span) / 2;
    const img = ctx.createImageData(d, d);
    const gain = 1 / Math.max(hi - lo, 1e-30);
    // Frequency (cycles per frame pixel) per FFT pixel from the centre.
    const perPixel = 1 / (n * shown.patch.bin);
    for (let cy = 0; cy < d; cy++) {
      const ya = Math.floor(start + (cy * span) / d);
      const yb = Math.max(ya + 1, Math.floor(start + ((cy + 1) * span) / d));
      for (let cx = 0; cx < d; cx++) {
        const xa = Math.floor(start + (cx * span) / d);
        const xb = Math.max(xa + 1, Math.floor(start + ((cx + 1) * span) / d));
        let m = -Infinity;
        for (let y = ya; y < yb; y++) {
          const row = y * n;
          for (let x = xa; x < xb; x++) if (power[row + x] > m) m = power[row + x];
        }
        let t = Math.min(1, Math.max(0, (m - lo) * gain));
        if (g0 !== 1) t **= g0;
        const k = Math.round(t * 255) * 4;
        const o = (cy * d + cx) * 4;
        if (mask || band) {
          // Tint by the mask's (or bandpass's) weight at this frequency.
          const u = start + ((cx + 0.5) * span) / d - n / 2;
          const v = start + ((cy + 0.5) * span) / d - n / 2;
          const a =
            MASK_ALPHA *
            (mask ? maskValue(mask, u * perPixel, v * perPixel) : bandpassValue(band!, Math.hypot(u, v) * perPixel));
          img.data[o] = lut[k] * (1 - a) + MASK_RGB[0] * a;
          img.data[o + 1] = lut[k + 1] * (1 - a) + MASK_RGB[1] * a;
          img.data[o + 2] = lut[k + 2] * (1 - a) + MASK_RGB[2] * a;
        } else {
          img.data[o] = lut[k];
          img.data[o + 1] = lut[k + 1];
          img.data[o + 2] = lut[k + 2];
        }
        img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [shown, view, display, side, zoom, mask, band]);

  const p = shown?.patch;
  // Click: mark that frequency. FFT pixel u (from the centre) is u/n cycles
  // per patch pixel, i.e. u / (n · pixel size) per unit length.
  const onClick = (e: React.MouseEvent) => {
    if (suppressClick.current) {
      suppressClick.current = false;
      return;
    }
    if (!measure || !p || !shown || !side) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const span = shown.result.n / zoom;
    const u = ((e.clientX - rect.left) / rect.width - 0.5) * span;
    const v = ((e.clientY - rect.top) / rect.height - 0.5) * span;
    setMark({ kx: u / (p.n * p.px), ky: v / (p.n * p.py) });
  };

  const freq = measure && p ? mark : null;
  const k = freq && Math.hypot(freq.kx, freq.ky);
  // The mark in FFT pixels, then on screen (for the spots and the ring).
  const at = p && freq ? { u: freq.kx * p.n * p.px, v: freq.ky * p.n * p.py } : null;
  const toScreen = (u: number) => (shown ? (u / (shown.result.n / zoom) + 0.5) * side : 0);
  // The mask's cutoff ring (an ellipse for an elliptical mask): f cycles
  // per frame pixel is f·n·bin FFT pixels, i.e. f·bin·zoom·side on screen.
  const fc = mask && maskCutoff(mask);
  const ring = p && fc ? { rx: fc.r1 * p.bin * zoom * side, ry: fc.r2 * p.bin * zoom * side, angle: fc.angle } : null;
  const ringR = ring ? Math.max(ring.rx, ring.ry) : null;
  const ringLabel = !p || !mask || !fc
    ? ""
    : mask.type === "elliptical"
      ? `n1 = ${formatValue(mask.n)} · n2 = ${formatValue(mask.n2)} · θ = ${formatValue(mask.theta)}°`
      : `n = ${formatValue(mask.n)} · ${formatValue(2 * mask.n)} px${
          p.units !== "px" ? ` (${formatValue((2 * mask.n * p.px) / p.bin)} ${p.units})` : ""
        }`;

  // The bandpass's cutoffs: a period of s px is 1/s cycles per pixel.
  const sizeLabel = (s: number) =>
    `${formatValue(s)} px${p && p.units !== "px" ? ` (${formatValue((s * p.px) / p.bin)} ${p.units})` : ""}`;
  const bandRings = p && band
    ? [
        { id: "small", r: (p.bin * zoom * side) / band.small, label: sizeLabel(band.small), up: true },
        { id: "large", r: (p.bin * zoom * side) / band.large, label: sizeLabel(band.large), up: false },
      ]
    : [];

  // Dragging a ring. A ring is grabbed within GRAB_PX of its line; the radius
  // then follows the pointer's distance from the centre. The mask's n (and n2
  // for an ellipse, keeping the ratio) scales as 1/radius; the bandpass sizes
  // are 1/radius per ring. The mask is updated as the pointer moves; the
  // bandpass only on release (it is shown meanwhile from local state).
  const scaleK = p ? p.bin * zoom * side : 0;
  const pointer = (e: React.PointerEvent) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const dx = e.clientX - rect.left - side / 2;
    const dy = e.clientY - rect.top - side / 2;
    return { dx, dy, d: Math.hypot(dx, dy) };
  };
  const hitRing = (dx: number, dy: number, d: number): string | null => {
    if (!p || !side) return null;
    let best: string | null = null;
    let bestGap = GRAB_PX;
    if (onMask && mask && ring) {
      const gap = Math.abs(d - ellipseRadius(ring.rx, ring.ry, ring.angle, dx, dy));
      if (gap <= bestGap) [best, bestGap] = ["mask", gap];
    }
    if (onBand) {
      for (const b of bandRings) {
        const gap = Math.abs(d - b.r);
        if (gap <= bestGap) [best, bestGap] = [b.id, gap];
      }
    }
    return best;
  };
  const dragTo = (g: Drag, dx: number, dy: number, d: number) => {
    if (!scaleK) return;
    if (g.id === "mask") {
      const s = d / ellipseRadius(g.rx, g.ry, g.angle, dx, dy);
      if (!(s > 0) || !onMask) return;
      const { n, n2 } = g.start;
      const elliptical = g.start.type === "elliptical";
      const lo = N_MIN / Math.min(n, elliptical ? n2 : n);
      const hi = N_MAX / Math.max(n, elliptical ? n2 : n);
      const k = clamp(1 / s, lo, hi);
      const next = { ...g.start, n: clamp(snap(n * k), N_MIN, N_MAX) };
      if (elliptical) next.n2 = clamp(snap(n2 * k), N_MIN, N_MAX);
      if (next.n !== mask?.n || next.n2 !== mask?.n2) onMask(next);
    } else if (bandProp) {
      const v = snap(scaleK / Math.max(d, 1));
      const next =
        g.id === "small"
          ? { small: snap(clamp(v, SNAP, band!.large - SNAP)), large: band!.large }
          : { small: band!.small, large: snap(clamp(v, band!.small + SNAP, BAND_MAX)) };
      liveRef.current = next;
      setLiveBand(next);
    }
  };
  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    const { dx, dy, d } = pointer(e);
    const id = hitRing(dx, dy, d);
    if (!id) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    suppressClick.current = true;
    drag.current =
      id === "mask" && mask && ring
        ? { id, start: mask, rx: ring.rx, ry: ring.ry, angle: ring.angle }
        : { id: id as "small" | "large" };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const { dx, dy, d } = pointer(e);
    if (drag.current) {
      dragTo(drag.current, dx, dy, d);
      return;
    }
    const id = hitRing(dx, dy, d);
    const cursor = id ? resizeCursor(dx, dy) : "";
    if (id !== (hover?.id ?? null) || cursor !== (hover?.cursor ?? "")) setHover(id ? { id, cursor } : null);
  };
  const onPointerUp = (e: React.PointerEvent) => {
    const g = drag.current;
    if (!g) return;
    drag.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
    // A click that follows a drag is not a measurement.
    setTimeout(() => (suppressClick.current = false), 0);
    const b = liveRef.current;
    liveRef.current = null;
    if (g.id !== "mask" && b && onBand && (b.small !== bandProp?.small || b.large !== bandProp?.large)) {
      onBand(b)
        .catch(() => {})
        .finally(() => setLiveBand(null));
    } else setLiveBand(null);
    const { dx, dy, d } = pointer(e);
    const id = hitRing(dx, dy, d);
    setHover(id ? { id, cursor: resizeCursor(dx, dy) } : null);
  };

  let readout: React.ReactNode;
  if (!image) readout = <span className="muted">No image in the viewer</span>;
  else if (tooSmall)
    readout = <span className="muted">{box ? `Too few pixels in ${box.name}` : "Too few pixels in view: zoom out"}</span>;
  else if (p && freq && k) {
    const angle = (Math.atan2(-freq.ky, freq.kx) * 180) / Math.PI;
    readout = (
      <>
        <b>d = {formatValue(1 / k)} {p.units}</b>
        <span className="muted">
          {" "}
          · {formatValue(k)} {p.units}⁻¹ · {formatValue(Math.round(angle))}°
        </span>
      </>
    );
  } else if (p) {
    readout = (
      <span className="muted">
        {p.size}×{p.size} px
        {p.region ? (p.cropped ? ` (centre of ${p.region})` : ` in ${p.region}`) : p.cropped ? " (centre of view)" : " in view"}
        {" "}· Nyquist d ={" "}
        {formatValue(2 * Math.max(p.px, p.py))} {p.units}
      </span>
    );
  } else readout = <span className="muted">Computing…</span>;

  return (
    <div className="fft-body">
      <div
        ref={wrapRef}
        className={`fft-canvas-wrap${measure && p ? " measuring" : ""}`}
        onClick={onClick}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerLeave={() => !drag.current && setHover(null)}
        style={{ cursor: hover?.cursor || undefined, touchAction: (onMask && ring) || (onBand && band) ? "none" : undefined }}
        title={
          hover
            ? "Drag to resize"
            : p
              ? `FFT of ${p.label}${measure ? ": click to measure a spacing" : ""}`
              : undefined
        }
      >
        <canvas ref={canvasRef} style={{ width: side, height: side }} />
        {ring && ringR !== null && ringR > 0 && (
          <svg className="fft-overlay" width={side} height={side}>
            <g className={`align-ring${hover?.id === "mask" ? " hot" : ""}`}>
              <ellipse
                cx={side / 2}
                cy={side / 2}
                rx={ring.rx}
                ry={ring.ry}
                transform={`rotate(${ring.angle} ${side / 2} ${side / 2})`}
              />
              <text x={Math.min(side - 4, side / 2 + ringR * 0.71 + 4)} y={Math.max(12, side / 2 - ringR * 0.71 - 4)} textAnchor={side / 2 + ringR * 0.71 + 120 > side ? "end" : "start"}>
                {ringLabel}
              </text>
            </g>
          </svg>
        )}
        {bandRings.length > 0 && (
          <svg className="fft-overlay" width={side} height={side}>
            <g>
              {bandRings.map(({ id, r, label, up }) => {
                const c = side / 2;
                const x = Math.min(side - 4, c + r * 0.71 + 4);
                return (
                  <g key={id} className={`align-ring${hover?.id === id ? " hot" : ""}`}>
                    <circle cx={c} cy={c} r={r} />
                    <text
                      x={x}
                      y={up ? Math.max(12, c - r * 0.71 - 4) : Math.min(side - 4, c + r * 0.71 + 12)}
                      textAnchor={c + r * 0.71 + 90 > side ? "end" : "start"}
                    >
                      {label}
                    </text>
                  </g>
                );
              })}
            </g>
          </svg>
        )}
        {shown && at && k ? (
          <svg className="fft-overlay" width={side} height={side}>
            <circle
              cx={side / 2}
              cy={side / 2}
              r={Math.hypot(toScreen(at.u) - side / 2, toScreen(at.v) - side / 2)}
              className="fft-ring"
            />
            <circle cx={toScreen(at.u)} cy={toScreen(at.v)} r={4} className="fft-spot" />
            <circle cx={toScreen(-at.u)} cy={toScreen(-at.v)} r={4} className="fft-spot" />
          </svg>
        ) : null}
        {busy && <span className="spinner fft-busy" aria-hidden />}
      </div>
      <div className="fft-readout">{readout}</div>
      {!compact && (
        <div
          className={`fft-contrast${contrastOpen ? "" : " collapsed"}`}
          onDoubleClick={(e) => {
            if (e.target instanceof HTMLCanvasElement) setManual(null);
          }}
        >
          <h3 className="fft-contrast-title">
            <button
              className="process-group-toggle"
              aria-expanded={contrastOpen}
              onClick={() => setContrastOpen(!contrastOpen)}
              title={contrastOpen ? "Hide the contrast controls" : "Show the contrast controls"}
            >
              <span className="process-chevron" aria-hidden="true" />
              Contrast
            </button>
            {!contrastOpen && (
              <span className="muted small">
                {scale === "log" ? "Log" : "Linear"} · {isColour(colormap) ? "solid" : colormap}
              </span>
            )}
          </h3>
          {contrastOpen && (
            <>
              <div className="fft-contrast-head">
                <div className="segmented fft-zoom" role="group" aria-label="FFT scale">
                  <button
                    className={scale === "linear" ? "active" : undefined}
                    aria-pressed={scale === "linear"}
                    onClick={() => setScale("linear")}
                    title="Amplitude |F|"
                  >
                    Linear
                  </button>
                  <button
                    className={scale === "log" ? "active" : undefined}
                    aria-pressed={scale === "log"}
                    onClick={() => setScale("log")}
                    title="log(1 + |F|²)"
                  >
                    Log
                  </button>
                </div>
                {range && (
                  <button className="link-button" onClick={() => setManual(null)} title="Automatic range (or double-click the histogram)">
                    Reset
                  </button>
                )}
              </div>
              <ContrastControls frame={view?.frame ?? null} display={display} onChange={onDisplay} />
            </>
          )}
        </div>
      )}
    </div>
  );
}
