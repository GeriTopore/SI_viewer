import { CURSOR2_COLOR, CURSOR_COLOR, CURSOR_SIZES, useSi, type SiState } from "../state/si";
import { ColourPick } from "./ColourPick";

interface Props {
  si: string;
  state: SiState;
  wholeColor: string;
}

/** Legend chips for the SI spectra: all pixels, the cursor and each region. */
export function SeriesLegend({ si, state, wholeColor }: Props) {
  const { patch, updateRoi, deleteRois, cursorSize, setCursorSize } = useSi();
  const cursor = state.cursor;

  return (
    <div className="series-legend">
      <label className={`series-chip${state.showWhole ? "" : " off"}`} title="Spectrum of every pixel">
        <input type="checkbox" checked={state.showWhole} onChange={(e) => patch(si, { showWhole: e.target.checked })} />
        <span className="series-swatch" style={{ background: wholeColor }} />
        All pixels
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
      </label>

      {state.cursor2 && (
        <span className={`series-chip${state.showCursor2 ? "" : " off"}`} title="Spectrum at the second cursor (⌘-click to place it)">
          <input
            type="checkbox"
            checked={state.showCursor2}
            onChange={(e) => patch(si, { showCursor2: e.target.checked })}
            aria-label="Show cursor 2"
          />
          <span className="series-swatch" style={{ background: CURSOR2_COLOR }} />
          Cursor 2
          <span className="muted">
            {state.cursor2.x}, {state.cursor2.y}
          </span>
          <button className="icon-button" title="Remove cursor 2" onClick={() => patch(si, { cursor2: null })}>
            ×
          </button>
        </span>
      )}

      {state.rois.map((r) => (
        <span
          key={r.id}
          className={`series-chip roi-chip${r.visible ? "" : " off"}${state.selected === r.id ? " selected" : ""}`}
          onClick={() => patch(si, { selected: r.id })}
          title="Click to select (Delete to remove)"
        >
          <input
            type="checkbox"
            checked={r.visible}
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => updateRoi(si, r.id, { visible: e.target.checked })}
            aria-label={`Show ${r.name}`}
          />
          <ColourPick color={r.color} label={r.name} onChange={(color) => updateRoi(si, r.id, { color })}>
            <span className="series-swatch" style={{ background: r.color }} />
            {r.name}
          </ColourPick>
          <button
            className="icon-button"
            title={`Remove ${r.name}`}
            onClick={(e) => {
              e.stopPropagation();
              deleteRois(si, [r.id]);
            }}
          >
            ×
          </button>
        </span>
      ))}
    </div>
  );
}
