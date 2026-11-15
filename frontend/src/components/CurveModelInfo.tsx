import type { DatasetSummary } from "../api/client";
import { formatValue } from "../format";
import { modeOf } from "../state/curveFit";
import { toggleFolded, toggleShown, useModelAtCursor, useModelColours, useModelView } from "../state/curveModel";

const MODE_LABEL = { free: "Free", limited: "Limit", fixed: "Fixed", twin: "Twin" } as const;

/** Info tab of a curve model (origin "curve-model"): its components with
 * their parameters at the cursor, read only. A component's checkbox shows
 * it on the plot; its header folds its table. */
export function CurveModelInfo({ d }: { d: DatasetSummary }) {
  const view = useModelView(d);
  const colours = useModelColours(d);
  const { result, cursor } = useModelAtCursor(d);
  const model = d.curve_model?.model;
  if (!model) return null;
  return (
    <section className="curve-fit curve-model-info">
      <p className="muted small curve-fit-goodness">
        {cursor
          ? [
              `${cursor.size > 1 ? `${cursor.size}×${cursor.size} px at` : "Pixel"} (${cursor.x}, ${cursor.y})`,
              result?.red_chisq != null ? `reduced χ² ${formatValue(result.red_chisq)}` : null,
              result && result.pixels !== cursor.size * cursor.size && result.pixels > 0 ? `${result.pixels} px fitted` : null,
            ].filter(Boolean).join(" · ")
          : "Place the cursor to see the values there"}
        {model.bg_lo != null && model.bg_hi != null && (
          <>
            <br />
            Background window {formatValue(model.bg_lo)}–{formatValue(model.bg_hi)}
          </>
        )}
      </p>
      {model.components.map((c) => {
        const shown = !view.hidden.includes(c.name);
        const folded = view.folded.includes(c.name);
        const at = result?.components.find((x) => x.name === c.name);
        return (
          <section key={c.name} className={`curve-component${c.active === false ? " off" : ""}`}>
            <div className="section-header curve-component-header" onClick={() => toggleFolded(d, c.name)} title={folded ? "Show its parameters" : "Hide its parameters"}>
              <span className="curve-component-title">
                <input
                  type="checkbox"
                  checked={shown && c.active !== false}
                  disabled={c.active === false}
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => toggleShown(d, c.name, e.target.checked)}
                  aria-label={`Show ${c.name}`}
                  title={c.active === false ? "Switched off in the fit" : "Show on the spectrum"}
                />
                <span className="series-swatch" style={{ background: colours[c.name] }} />
                <span>{c.name}</span>
                {c.name !== c.kind && !c.name.startsWith(`${c.kind} `) && <span className="muted small">{c.kind}</span>}
              </span>
              <span className="muted small">{folded ? "▸" : "▾"}</span>
            </div>
            {!folded && (
              <table className="axes curve-params">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Value</th>
                    <th>Fit</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(c.params).map(([name, p]) => {
                    const v = at?.params.find((q) => q.name === name)?.value;
                    const mode = modeOf({ ...p, twin: p.twin ?? null });
                    const how =
                      mode === "twin" && p.twin
                        ? `= ${p.twin.expr === "x" ? "" : `${p.twin.expr} with x = `}${p.twin.component} ${p.twin.param}`
                        : mode === "limited"
                          ? `${p.bmin != null ? formatValue(p.bmin) : "−∞"} to ${p.bmax != null ? formatValue(p.bmax) : "∞"}`
                          : undefined;
                    return (
                      <tr key={name}>
                        <td>{name}</td>
                        <td className="num">{v !== undefined ? formatValue(v) : "–"}</td>
                        <td className="muted" title={how}>{MODE_LABEL[mode]}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </section>
        );
      })}
    </section>
  );
}
