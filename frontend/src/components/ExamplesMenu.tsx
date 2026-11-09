import { useEffect, useRef, useState } from "react";
import { api, type ExampleInfo } from "../api/client";
import { useDatasets } from "../state/datasets";

/** "Examples" button: bundled datasets for trying features (and tutorials). */
export function ExamplesMenu() {
  const [open, setOpen] = useState(false);
  const [examples, setExamples] = useState<ExampleInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const progress = useDatasets((s) => s.progress);
  const openExample = useDatasets((s) => s.openExample);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open || examples) return;
    api.listExamples().then(setExamples).catch((e) => setError((e as Error).message));
  }, [open, examples]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="examples-menu" ref={ref}>
      <button
        className={`secondary${open ? " active" : ""}`}
        onClick={() => setOpen(!open)}
        disabled={!!progress}
        aria-expanded={open}
        title="Open example data to try the app's features"
      >
        Examples ▾
      </button>
      {open && (
        <div className="examples-popover" role="menu">
          {error && <p className="field-error">{error}</p>}
          {!examples && !error && <p className="muted small">Loading…</p>}
          {examples?.map((ex) => (
            <button
              key={ex.id}
              role="menuitem"
              className="example-item"
              onClick={() => {
                setOpen(false);
                openExample(ex.id, ex.title);
              }}
            >
              <span className="example-title">{ex.title}</span>
              <span className="example-desc">{ex.description}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
