import { Fragment, useState, type ReactNode } from "react";
import { isColour, swatchBackground } from "../viewer/colormaps";
import type { DisplayParams } from "../viewer/renderer";

export interface PanelLayer {
  id: string; // "base" or a dataset id
  label: string;
  visible: boolean;
  opacity: number;
  display: DisplayParams | undefined;
  removable: boolean;
  /** A shape layer's colour (else the image's colormap). */
  colour?: string;
  /** A shape layer's mark size (× its default). */
  size?: { value: number; onChange: (value: number) => void; label?: string };
  /** What a shape layer is coloured by. */
  choice?: { value: string; options: { value: string; label: string }[]; title: string; onChange: (value: string) => void };
  /** More of its properties (e.g. a vector's shape). */
  properties?: { label: string; control: ReactNode }[];
}

export interface LayersPanelProps {
  layers: PanelLayer[];
  active: string;
  onActivate: (id: string) => void;
  onChange: (id: string, patch: { visible?: boolean; opacity?: number }) => void;
  onColour: (id: string, colour: string) => void;
  onRemove: (id: string) => void;
  /** Show the opacity sliders (not in a grid, where nothing is blended). */
  opacity?: boolean;
}

/** List of image layers: visibility, colour, opacity, remove. Floating over
 * the image, or ``docked`` in the Layers tab (no header to fold it), where a
 * layer with properties (a shape layer's size, colour-by, a vector's shape
 * …) opens them nested under it (▸). */
export function LayersPanel({
  layers, active, onActivate, onChange, onColour, onRemove, opacity = true, docked = false,
}: LayersPanelProps & { docked?: boolean }) {
  const [collapsed, setCollapsed] = useState(false);
  const [open, setOpen] = useState<Set<string>>(() => new Set());
  const toggleOpen = (id: string) =>
    setOpen((s) => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  const propsOf = (l: PanelLayer) => [
    ...(l.size
      ? [{
          label: l.size.label ?? "Size",
          control: (
            <input
              type="range"
              className="layer-size"
              min={0.5}
              max={4}
              step={0.25}
              value={l.size.value}
              title={`×${l.size.value}`}
              onChange={(e) => l.size!.onChange(Number(e.target.value))}
            />
          ),
        }]
      : []),
    ...(l.choice
      ? [{
          label: "Colour by",
          control: (
            <select className="layer-choice" value={l.choice.value} title={l.choice.title} onChange={(e) => l.choice!.onChange(e.target.value)}>
              {l.choice.options.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          ),
        }]
      : []),
    ...(l.properties ?? []),
  ];

  return (
    <div className={docked ? "layers-panel docked" : "layers-panel"}>
      {!docked && (
        <button className="layers-header" onClick={() => setCollapsed(!collapsed)} aria-expanded={!collapsed}>
          <span>{collapsed ? "▸" : "▾"} Layers</span>
          <span className="muted">{layers.length}</span>
        </button>
      )}
      {!collapsed && (
        <ul>
          {layers.map((l) => {
            const cm = l.colour ?? l.display?.colormap ?? "gray";
            const props = docked ? propsOf(l) : [];
            const expanded = props.length > 0 && open.has(l.id);
            return (
              <Fragment key={l.id}>
              <li
                className={`layer-row${l.id === active ? " active" : ""}${l.visible ? "" : " hidden"}${opacity ? "" : " no-opacity"}`}
                onClick={() => onActivate(l.id)}
                title="Click to edit this layer's contrast and colour"
              >
                <button
                  className="layer-eye"
                  aria-pressed={l.visible}
                  title={opacity ? (l.visible ? "Hide layer" : "Show layer") : l.visible ? "Hide panel" : "Show panel"}
                  onClick={(e) => {
                    e.stopPropagation();
                    onChange(l.id, { visible: !l.visible });
                  }}
                >
                  {l.visible ? <EyeIcon /> : <EyeOffIcon />}
                </button>
                <label
                  className="layer-swatch"
                  style={{ background: swatchBackground(cm) }}
                  title="Choose colour"
                  onClick={(e) => e.stopPropagation()}
                >
                  <input
                    type="color"
                    value={isColour(cm) ? cm : "#ffffff"}
                    onChange={(e) => onColour(l.id, e.target.value)}
                  />
                </label>
                <span className="layer-main">
                  {props.length > 0 && (
                    <button
                      className="layer-expand"
                      aria-expanded={expanded}
                      title={expanded ? "Hide its properties" : "Its properties"}
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleOpen(l.id);
                      }}
                    >
                      {expanded ? "▾" : "▸"}
                    </button>
                  )}
                  <span className="layer-label" title={l.label}>{l.label}</span>
                </span>
                {opacity && <input
                  type="range"
                  className="layer-opacity"
                  min={0}
                  max={1}
                  step={0.01}
                  value={l.opacity}
                  title={`Opacity ${Math.round(l.opacity * 100)}%`}
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => onChange(l.id, { opacity: Number(e.target.value) })}
                />}
                {l.removable ? (
                  <button
                    className="icon-button"
                    title="Remove layer"
                    onClick={(e) => {
                      e.stopPropagation();
                      onRemove(l.id);
                    }}
                  >
                    ×
                  </button>
                ) : (
                  <span className="layer-remove-spacer" />
                )}
              </li>
              {expanded && (
                <li className="layer-props" onClick={(e) => e.stopPropagation()}>
                  {props.map((q) => (
                    <label key={q.label}>
                      <span>{q.label}</span>
                      {q.control}
                    </label>
                  ))}
                </li>
              )}
              </Fragment>
            );
          })}
        </ul>
      )}
    </div>
  );
}

export const EyeIcon = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden>
    <path d="M1 8s2.5-5 7-5 7 5 7 5-2.5 5-7 5-7-5-7-5z" fill="none" stroke="currentColor" strokeWidth="1.4" />
    <circle cx="8" cy="8" r="2.2" fill="currentColor" />
  </svg>
);

export const EyeOffIcon = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden>
    <path d="M1 8s2.5-5 7-5 7 5 7 5-2.5 5-7 5-7-5-7-5z" fill="none" stroke="currentColor" strokeWidth="1.4" opacity="0.5" />
    <path d="M2 14L14 2" stroke="currentColor" strokeWidth="1.4" />
  </svg>
);
