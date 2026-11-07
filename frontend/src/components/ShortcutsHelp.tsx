import { useEffect, useState } from "react";
import { SHORTCUTS, useShortcuts } from "../keyboard";

/** "?" button in the top bar, and the shortcut list it (or the ? key) opens. */
export function ShortcutsHelp() {
  const [open, setOpen] = useState(false);
  useShortcuts({ "?": () => setOpen((o) => !o) });
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <>
      <button className="theme-toggle" onClick={() => setOpen(true)} title="Keyboard shortcuts (?)" aria-label="Keyboard shortcuts">
        ?
      </button>
      {open && (
        <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && setOpen(false)}>
          <div className="modal shortcuts-dialog" role="dialog" aria-label="Keyboard shortcuts">
            <h2>Keyboard shortcuts</h2>
            <div className="shortcuts-groups">
              {SHORTCUTS.map((g) => (
                <section key={g.group}>
                  <h3>{g.group}</h3>
                  <dl>
                    {g.keys.map(([k, what]) => (
                      <div key={k} className="shortcut-row">
                        <dt>
                          <kbd>{k}</kbd>
                        </dt>
                        <dd>{what}</dd>
                      </div>
                    ))}
                  </dl>
                </section>
              ))}
            </div>
            <div className="rebin-actions">
              <button className="primary" onClick={() => setOpen(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
