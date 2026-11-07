import { useEffect, type ReactNode } from "react";
import type { Frame } from "../api/arrays";
import { useDisplay } from "../state/display";
import type { DisplayParams } from "../viewer/renderer";
import { autoContrast } from "../views/useImageDisplay";
import { ImageView, type AxisCal, type Transform, type ViewLayer } from "./ImageView";

export interface PanelLayer {
  key: string;
  label: string;
  frame: Frame | null;
  colormap?: string;
  opacity?: number;
  /** Fixed contrast (e.g. 0–1 for a mask) instead of automatic. */
  range?: [number, number];
}

const NO_CAL: AxisCal = { scale: 1, offset: 0, units: "" };

interface Props {
  /** Pan/zoom and contrast key of the panel. */
  viewKey: string;
  title: ReactNode;
  /** Extra line under the title (e.g. a readout). */
  subtitle?: ReactNode;
  layers: PanelLayer[];
  /** Changes when the data does: contrast is redone. */
  version: string;
  width: number;
  height: number;
  calX?: AxisCal;
  calY?: AxisCal;
  loading?: string | null;
  error?: string | null;
  renderOverlay?: (t: Transform) => ReactNode;
  /** Publish to the FFT panel (the workspace's main image only). */
  publish?: boolean;
}

/** A titled image view for diagnostics (e.g. an FFT or a cross-correlation),
 * auto-contrasted whenever its data changes. */
export function ArrayPanel({
  viewKey, title, subtitle, layers, version, width, height, calX = NO_CAL, calY = NO_CAL, loading, error,
  renderOverlay, publish = false,
}: Props) {
  const displays = useDisplay((s) => s.display);
  useEffect(() => {
    for (const l of layers) {
      if (!l.frame) continue;
      const key = `${viewKey}:${l.key}`;
      if (l.range) {
        const { display, initDisplay, setDisplay } = useDisplay.getState();
        const fixed: DisplayParams = { lo: l.range[0], hi: l.range[1], gamma: 1, colormap: l.colormap ?? "gray" };
        const cur = display[key];
        if (!cur) initDisplay(key, fixed);
        else if (cur.lo !== fixed.lo || cur.hi !== fixed.hi) setDisplay(key, { lo: fixed.lo, hi: fixed.hi });
      } else autoContrast(key, l.frame, version, () => l.colormap ?? "gray");
    }
  }, [layers, version, viewKey]);

  const viewLayers: ViewLayer[] = layers.map((l) => ({
    key: l.key,
    label: l.label,
    frame: l.frame,
    display: displays[`${viewKey}:${l.key}`],
    opacity: l.opacity ?? 1,
    visible: true,
  }));

  return (
    <section className="array-panel">
      <header className="array-panel-title">
        <span>{title}</span>
        {subtitle && <span className="muted">{subtitle}</span>}
      </header>
      <ImageView
        viewKey={viewKey}
        layers={viewLayers}
        fullWidth={width}
        fullHeight={height}
        calX={calX}
        calY={calY}
        loading={loading}
        error={error}
        renderOverlay={renderOverlay}
        publish={publish}
        scaleBar={!!calX.units} // e.g. an FFT: no meaningful scale bar
        readout={false}
      />
    </section>
  );
}
