import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useUi } from "../state/ui";
import { useViewerImage, type ViewerImage } from "../state/viewerImage";
import type { FftRequest, FftResult } from "../viewer/fft.worker";
import { formatValue, inAngstrom } from "../format";
import type { MaskSettings } from "../api/client";
import { maskCutoff, maskValue } from "../viewer/fourierMask";

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
  /** The patch was cut from a larger visible area. */
  cropped: boolean;
}

interface Shown {
  patch: Patch;
  result: FftResult;
}

const nextPow2 = (v: number) => 2 ** Math.ceil(Math.log2(v));

/** The centred square of the visible area, as frame pixels. */
function cutPatch(img: ViewerImage): { data: Float32Array; patch: Patch } | null {
  const { frame, calX, calY } = img;
  const b = frame.bin;
  const fx0 = Math.floor(img.x0 / b);
  const fy0 = Math.floor(img.y0 / b);
  const fx1 = Math.min(frame.width, Math.ceil(img.x1 / b));
  const fy1 = Math.min(frame.height, Math.ceil(img.y1 / b));
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
      label: img.label,
      size,
      n: nextPow2(size),
      px: x.size * b,
      py: y.size * b,
      units: x.units,
      bin: b,
      cropped: fx1 - fx0 > size || fy1 - fy0 > size,
    },
  };
}

/** Power spectrum (FFT) of the visible part of the viewer's image. Hovering
 * gives the spatial frequency and the matching real-space spacing. */
export function FftPanel() {
  const open = useUi((s) => s.fftOpen);
  const setOpen = useUi((s) => s.setFftOpen);
  const image = useViewerImage((s) => s.image);
  const fftMask = useViewerImage((s) => s.fftMask);
  const showMask = useViewerImage((s) => s.showFftMask);
  const setShowMask = useViewerImage((s) => s.setShowFftMask);
  const [zoom, setZoom] = useState(1);

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
        {open && fftMask && (
          <button
            className={`fft-mask-toggle${showMask ? " active" : ""}`}
            aria-pressed={showMask}
            onClick={() => setShowMask(!showMask)}
            title={showMask ? "Hide the alignment's Fourier mask to see the whole FFT" : "Show the alignment's Fourier mask"}
          >
            Mask
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
      {open && <FftView image={image} zoom={zoom} mask={fftMask && showMask ? fftMask : null} />}
    </section>
  );
}

// Colour of the Fourier mask's tint, and its strength where the mask is 1.
const MASK_RGB = [255, 150, 30];
const MASK_ALPHA = 0.5;

function FftView({ image, zoom, mask }: { image: ViewerImage | null; zoom: number; mask: MaskSettings | null }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [side, setSide] = useState(0);
  const [shown, setShown] = useState<Shown | null>(null);
  const [tooSmall, setTooSmall] = useState(false);
  const [busy, setBusy] = useState(false);
  const [hover, setHover] = useState<{ u: number; v: number } | null>(null);

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
      const cut = cutPatch(image);
      setTooSmall(!cut);
      if (!cut) {
        setShown(null);
        return;
      }
      const id = ++nextId.current;
      pending.current = { id, patch: cut.patch };
      setBusy(true);
      const req: FftRequest = { id, patch: cut.data, size: cut.patch.size, n: cut.patch.n };
      worker.current?.postMessage(req, [cut.data.buffer]);
    }, DELAY_MS);
    return () => clearTimeout(t);
  }, [image]);

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
    if (!shown) {
      ctx.clearRect(0, 0, d, d);
      return;
    }
    const { n, power, lo, hi } = shown.result;
    const span = n / zoom;
    const start = (n - span) / 2;
    const img = ctx.createImageData(d, d);
    const scale = 255 / (hi - lo);
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
        const g = Math.max(0, Math.min(255, (m - lo) * scale));
        const o = (cy * d + cx) * 4;
        if (mask) {
          // Tint by the mask's weight at this frequency.
          const u = start + ((cx + 0.5) * span) / d - n / 2;
          const v = start + ((cy + 0.5) * span) / d - n / 2;
          const a = MASK_ALPHA * maskValue(mask, Math.hypot(u, v) * perPixel);
          img.data[o] = g * (1 - a) + MASK_RGB[0] * a;
          img.data[o + 1] = g * (1 - a) + MASK_RGB[1] * a;
          img.data[o + 2] = g * (1 - a) + MASK_RGB[2] * a;
        } else img.data[o] = img.data[o + 1] = img.data[o + 2] = g;
        img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [shown, side, zoom, mask]);

  // Hover position in FFT pixels from the centre (u right, v down).
  const onMove = (e: React.PointerEvent) => {
    if (!shown || !side) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const span = shown.result.n / zoom;
    setHover({
      u: ((e.clientX - rect.left) / rect.width - 0.5) * span,
      v: ((e.clientY - rect.top) / rect.height - 0.5) * span,
    });
  };

  const p = shown?.patch;
  // Spatial frequency: FFT pixel u is u/n cycles per pixel.
  const freq = p && hover && { kx: hover.u / (p.n * p.px), ky: hover.v / (p.n * p.py) };
  const k = freq && Math.hypot(freq.kx, freq.ky);
  // Screen radius and position of the hovered frequency (for the ring).
  const toScreen = (u: number) => (shown ? (u / (shown.result.n / zoom) + 0.5) * side : 0);
  // The mask's cutoff ring: f_c cycles per frame pixel is f_c·n·bin FFT pixels.
  const fc = mask && maskCutoff(mask);
  const ringR = p && fc ? toScreen(fc * shown!.result.n * p.bin) - side / 2 : null;
  const ringLabel =
    p && mask && fc
      ? `n = ${formatValue(mask.n)} · ${formatValue(2 * mask.n)} px${
          p.units !== "px" ? ` (${formatValue((2 * mask.n * p.px) / p.bin)} ${p.units})` : ""
        }`
      : "";

  let readout: React.ReactNode;
  if (!image) readout = <span className="muted">No image in the viewer</span>;
  else if (tooSmall) readout = <span className="muted">Too few pixels in view: zoom out</span>;
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
        {p.size}×{p.size} px{p.cropped ? " (centre of view)" : " in view"} · Nyquist d ={" "}
        {formatValue(2 * Math.max(p.px, p.py))} {p.units}
      </span>
    );
  } else readout = <span className="muted">Computing…</span>;

  return (
    <div className="fft-body">
      <div
        ref={wrapRef}
        className="fft-canvas-wrap"
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
        title={p ? `FFT of ${p.label}` : undefined}
      >
        <canvas ref={canvasRef} style={{ width: side, height: side }} />
        {ringR !== null && ringR > 0 && (
          <svg className="fft-overlay" width={side} height={side}>
            <g className="align-ring">
              <circle cx={side / 2} cy={side / 2} r={ringR} />
              <text x={Math.min(side - 4, side / 2 + ringR * 0.71 + 4)} y={Math.max(12, side / 2 - ringR * 0.71 - 4)} textAnchor={side / 2 + ringR * 0.71 + 120 > side ? "end" : "start"}>
                {ringLabel}
              </text>
            </g>
          </svg>
        )}
        {shown && hover && k ? (
          <svg className="fft-overlay" width={side} height={side}>
            <circle
              cx={side / 2}
              cy={side / 2}
              r={Math.hypot(toScreen(hover.u) - side / 2, toScreen(hover.v) - side / 2)}
              className="fft-ring"
            />
            <circle cx={toScreen(hover.u)} cy={toScreen(hover.v)} r={4} className="fft-spot" />
            <circle cx={toScreen(-hover.u)} cy={toScreen(-hover.v)} r={4} className="fft-spot" />
          </svg>
        ) : null}
        {busy && <span className="spinner fft-busy" aria-hidden />}
      </div>
      <div className="fft-readout">{readout}</div>
    </div>
  );
}
