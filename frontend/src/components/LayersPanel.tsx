import { useState } from "react";
import { isColour, swatchBackground } from "../viewer/colormaps";
import type { DisplayParams } from "../viewer/renderer";

export interface PanelLayer {
  id: string; // "base" or a dataset id
  label: string;
  visible: boolean;
  opacity: number;
  display: DisplayParams | undefined;
  removable: boolean;
}

interface Props {
  layers: PanelLayer[];
  active: string;
  onActivate: (id: string) => void;
  onChange: (id: string, patch: { visible?: boolean; opacity?: number }) => void;
  onColour: (id: string, colour: string) => void;
  onRemove: (id: string) => void;
}

/** Floating list of image layers: visibility, colour, opacity, remove. */
export function LayersPanel({ layers, active, onActivate, onChange, onColour, onRemove }: Props) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="layers-panel">
      <button className="layers-header" onClick={() => setCollapsed(!collapsed)} aria-expanded={!collapsed}>
        <span>{collapsed ? "▸" : "▾"} Layers</span>
        <span className="muted">{layers.length}</span>
      </button>
      {!collapsed && (
        <ul>
          {layers.map((l) => {
            const cm = l.display?.colormap ?? "gray";
            return (
              <li
                key={l.id}
                className={`layer-row${l.id === active ? " active" : ""}${l.visible ? "" : " hidden"}`}
                onClick={() => onActivate(l.id)}
                title="Click to edit this layer's contrast and colour"
              >
                <button
                  className="layer-eye"
                  aria-pressed={l.visible}
                  title={l.visible ? "Hide layer" : "Show layer"}
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
                <span className="layer-label">{l.label}</span>
                <input
                  type="range"
                  className="layer-opacity"
                  min={0}
                  max={1}
                  step={0.01}
                  value={l.opacity}
                  title={`Opacity ${Math.round(l.opacity * 100)}%`}
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => onChange(l.id, { opacity: Number(e.target.value) })}
                />
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
            );
          })}
        </ul>
      )}
    </div>
  );
}

const EyeIcon = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden>
    <path d="M1 8s2.5-5 7-5 7 5 7 5-2.5 5-7 5-7-5-7-5z" fill="none" stroke="currentColor" strokeWidth="1.4" />
    <circle cx="8" cy="8" r="2.2" fill="currentColor" />
  </svg>
);

const EyeOffIcon = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden>
    <path d="M1 8s2.5-5 7-5 7 5 7 5-2.5 5-7 5-7-5-7-5z" fill="none" stroke="currentColor" strokeWidth="1.4" opacity="0.5" />
    <path d="M2 14L14 2" stroke="currentColor" strokeWidth="1.4" />
  </svg>
);
