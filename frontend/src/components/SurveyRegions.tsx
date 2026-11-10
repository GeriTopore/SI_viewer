import type { SurveyRegion } from "../api/client";
import type { Transform } from "./ImageView";

/** The rectangles GMS drew on a survey image: where the spectrum image was
 * taken, and the spatial drift reference. */
export function SurveyRegions({ t, regions }: { t: Transform; regions: SurveyRegion[] }) {
  return (
    <g className="survey-regions">
      {regions.map((r, i) => {
        const [x0, y0] = t.toScreen(r.x, r.y);
        const [x1, y1] = t.toScreen(r.x + r.width, r.y + r.height);
        return (
          <g key={i} className={`survey-region ${r.kind}`}>
            <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} />
            {r.label && (
              <text className="roi-label" x={x0 + 3} y={y0 - 4}>
                {r.label}
              </text>
            )}
          </g>
        );
      })}
    </g>
  );
}
