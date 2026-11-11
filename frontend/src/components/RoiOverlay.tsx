import type { Pt, Transform } from "./ImageView";
import { CURSOR_COLOR, shapeBounds, type Roi, type Shape, type SiState } from "../state/si";

interface Props {
  t: Transform;
  state: SiState;
  cursorSize: number;
  draft: Shape | null;
  hover: Pt | null;
  /** Regions hidden (the cursor and a region being drawn still show). */
  hideRois?: boolean;
}

/** SVG drawing of the spectrum cursor, regions and the region being drawn. */
export function RoiOverlay({ t, state, cursorSize, draft, hover, hideRois }: Props) {
  return (
    <g className="roi-overlay">
      {state.rois
        .filter((r) => r.visible && !hideRois)
        .map((r) => (
          <RoiShape key={r.id} t={t} roi={r} selected={r.id === state.selected} />
        ))}
      {draft && <DraftShape t={t} shape={draft} hover={hover} />}
      {state.cursor && state.showCursor && <CursorBox t={t} cursor={state.cursor} size={cursorSize} />}
    </g>
  );
}

function shapeElement(t: Transform, shape: Shape, props: React.SVGProps<SVGElement>) {
  if (shape.kind === "polygon") {
    const pts = shape.points.map(([x, y]) => t.toScreen(x, y).join(",")).join(" ");
    return <polygon points={pts} {...(props as React.SVGProps<SVGPolygonElement>)} />;
  }
  const [x0, y0, x1, y1] = shapeBounds(shape);
  const [sx0, sy0] = t.toScreen(x0, y0);
  const [sx1, sy1] = t.toScreen(x1, y1);
  if (shape.kind === "rect") {
    return <rect x={sx0} y={sy0} width={sx1 - sx0} height={sy1 - sy0} {...(props as React.SVGProps<SVGRectElement>)} />;
  }
  return (
    <ellipse
      cx={(sx0 + sx1) / 2}
      cy={(sy0 + sy1) / 2}
      rx={(sx1 - sx0) / 2}
      ry={(sy1 - sy0) / 2}
      {...(props as React.SVGProps<SVGEllipseElement>)}
    />
  );
}

function RoiShape({ t, roi, selected }: { t: Transform; roi: Roi; selected: boolean }) {
  const [x0, y0, x1, y1] = shapeBounds(roi.shape);
  const [lx, ly] = t.toScreen(x0, y0);
  const corners =
    selected && roi.shape.kind !== "polygon"
      ? [t.toScreen(x0, y0), t.toScreen(x1, y0), t.toScreen(x0, y1), t.toScreen(x1, y1)]
      : [];
  return (
    <g>
      {shapeElement(t, roi.shape, {
        fill: roi.color,
        fillOpacity: selected ? 0.18 : 0.1,
        stroke: roi.color,
        strokeWidth: selected ? 2 : 1.5,
      })}
      <text x={lx + 3} y={ly - 4} className="roi-label" fill={roi.color}>
        {roi.name}
      </text>
      {corners.map(([cx, cy], i) => (
        <rect key={i} x={cx - 4} y={cy - 4} width={8} height={8} className="roi-handle" stroke={roi.color} />
      ))}
    </g>
  );
}

function DraftShape({ t, shape, hover }: { t: Transform; shape: Shape; hover: Pt | null }) {
  if (shape.kind !== "polygon") {
    return shapeElement(t, shape, { fill: "#ffffff", fillOpacity: 0.08, stroke: "#ffffff", strokeDasharray: "4 3" });
  }
  const pts = [...shape.points, ...(hover ? [[hover.x, hover.y] as [number, number]] : [])];
  const [fx, fy] = t.toScreen(...shape.points[0]);
  return (
    <g>
      <polyline
        points={pts.map(([x, y]) => t.toScreen(x, y).join(",")).join(" ")}
        fill="none"
        stroke="#ffffff"
        strokeDasharray="4 3"
      />
      <circle cx={fx} cy={fy} r={5} className="roi-handle" stroke="#ffffff" />
    </g>
  );
}

function CursorBox({ t, cursor, size }: { t: Transform; cursor: { x: number; y: number }; size: number }) {
  const half = Math.floor((size - 1) / 2);
  const [sx0, sy0] = t.toScreen(cursor.x - half, cursor.y - half);
  const [sx1, sy1] = t.toScreen(cursor.x - half + size, cursor.y - half + size);
  // Keep the box visible when zoomed out.
  const pad = Math.max(0, (6 - (sx1 - sx0)) / 2);
  const cx = (sx0 + sx1) / 2;
  const cy = (sy0 + sy1) / 2;
  const arm = 7;
  const box = { x0: sx0 - pad, y0: sy0 - pad, x1: sx1 + pad, y1: sy1 + pad };
  return (
    <g stroke={CURSOR_COLOR} strokeWidth={1.5} className="cursor-box">
      <rect x={box.x0} y={box.y0} width={box.x1 - box.x0} height={box.y1 - box.y0} fill="none" />
      <line x1={cx} y1={box.y0 - arm} x2={cx} y2={box.y0} />
      <line x1={cx} y1={box.y1} x2={cx} y2={box.y1 + arm} />
      <line x1={box.x0 - arm} y1={cy} x2={box.x0} y2={cy} />
      <line x1={box.x1} y1={cy} x2={box.x1 + arm} y2={cy} />
    </g>
  );
}
