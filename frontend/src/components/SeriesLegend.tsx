import type { PixelsSpectrum } from "../api/arrays";
import { CURSOR_COLOR, useSi, type SiState } from "../state/si";
import { CURSOR_CHANNEL } from "../views/si/useSiSpectra";

interface Props {
  si: string;
  state: SiState;
  spectra: Record<string, PixelsSpectrum>;
  totalPixels: number;
  wholeColor: string;
}

const CURSOR_SIZES = [1, 3, 5, 7, 9, 15];

/** Legend chips for the SI spectra: all pixels, the cursor and each region. */
export function SeriesLegend({ si, state, spectra, totalPixels, wholeColor }: Props) {
  const { patch, updateRoi, removeRoi, cursorSize, setCursorSize } = useSi();
  const cursor = state.cursor;

  return (
    <div className="series-legend">
      <label className={`series-chip${state.showWhole ? "" : " off"}`} title="Spectrum of every pixel">
        <input type="checkbox" checked={state.showWhole} onChange={(e) => patch(si, { showWhole: e.target.checked })} />
        <span className="series-swatch" style={{ background: wholeColor }} />
        All pixels <span className="muted">{totalPixels.toLocaleString()} px</span>
      </label>

      <label className={`series-chip${state.showCursor ? "" : " off"}`} title="Spectrum at the cursor">
        <input type="checkbox" checked={state.showCursor} onChange={(e) => patch(si, { showCursor: e.target.checked })} />
        <span className="series-swatch" style={{ background: CURSOR_COLOR }} />
        Cursor
        <span className="muted">{cursor ? `${cursor.x}, ${cursor.y}` : "–"}</span>
        <select
          value={cursorSize}
          onChange={(e) => setCursorSize(Number(e.target.value))}
          onClick={(e) => e.stopPropagation()}
          title="Average over an n × n block of pixels"
        >
          {CURSOR_SIZES.map((n) => (
            <option key={n} value={n}>
              {n}×{n}
            </option>
          ))}
        </select>
        {spectra[CURSOR_CHANNEL] && cursorSize > 1 && (
          <span className="muted">{spectra[CURSOR_CHANNEL].pixels} px</span>
        )}
      </label>

      {state.rois.map((r) => (
        <span
          key={r.id}
          className={`series-chip roi-chip${r.visible ? "" : " off"}${state.selected === r.id ? " selected" : ""}`}
          onClick={() => patch(si, { selected: r.id })}
          title="Click to select (drag its corners in the navigator to resize, Delete to remove)"
        >
          <input
            type="checkbox"
            checked={r.visible}
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => updateRoi(si, r.id, { visible: e.target.checked })}
            aria-label={`Show ${r.name}`}
          />
          <span className="series-swatch" style={{ background: r.color }} />
          {r.name}
          <span className="muted">{spectra[r.id] ? `${spectra[r.id].pixels.toLocaleString()} px` : "…"}</span>
          <button
            className="icon-button"
            title={`Remove ${r.name}`}
            onClick={(e) => {
              e.stopPropagation();
              removeRoi(si, r.id);
            }}
          >
            ×
          </button>
        </span>
      ))}
    </div>
  );
}
