import { create } from "zustand";
import type { Frame } from "../api/arrays";
import type { MaskSettings } from "../api/client";
import type { AxisCal } from "../components/ImageView";

/** The image the viewer shows (its active layer) and the part of it in view,
 * published for analysis such as the FFT panel. */
export interface ViewerImage {
  label: string;
  frame: Frame;
  calX: AxisCal;
  calY: AxisCal;
  /** Visible part of the image, in full-resolution pixels (clipped to it). */
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

interface ViewerImageState {
  image: ViewerImage | null;
  /** Which view published ``image`` (so an old view leaving doesn't clear a new one's). */
  owner: string | null;
  publish: (owner: string, image: ViewerImage | null) => void;
  clear: (owner: string) => void;
  /** A Fourier mask drawn on the FFT panel (drift correction), and whether
   * it is shown. */
  fftMask: MaskSettings | null;
  showFftMask: boolean;
  setFftMask: (mask: MaskSettings | null) => void;
  setShowFftMask: (show: boolean) => void;
}

export const useViewerImage = create<ViewerImageState>((set, get) => ({
  image: null,
  owner: null,
  publish: (owner, image) => set({ owner, image }),
  clear: (owner) => get().owner === owner && set({ owner: null, image: null }),
  fftMask: null,
  showFftMask: true,
  setFftMask: (fftMask) => set({ fftMask }),
  setShowFftMask: (showFftMask) => set({ showFftMask }),
}));
