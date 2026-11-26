import { useEffect, useRef, useState } from "react";
import { api, type ExampleInfo } from "../api/client";
import { useDatasets } from "../state/datasets";

const OPEN_GROUP_KEY = "si-viewer.examples.group";

function storedGroup(): string | null {
  try {
    return localStorage.getItem(OPEN_GROUP_KEY);
  } catch {
    return null;
  }
}

/** "Examples" button: example datasets for trying features (and tutorials),
 * grouped by use; one group open at a time. */
export function ExamplesMenu() {
  const [open, setOpen] = useState(false);
  const [examples, setExamples] = useState<ExampleInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [group, setGroup] = useState<string | null>(storedGroup);
  const progress = useDatasets((s) => s.progress);
  const openExample = useDatasets((s) => s.openExample);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    // Refetched on each opening: a downloaded dataset loses its size.
    api.listExamples().then(setExamples).catch((e) => setError((e as Error).message));
  }, [open]);

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

  const toggleGroup = (g: string) => {
    const next = group === g ? null : g;
    setGroup(next);
    try {
      if (next) localStorage.setItem(OPEN_GROUP_KEY, next);
      else localStorage.removeItem(OPEN_GROUP_KEY);
    } catch {
      /* not remembered */
    }
  };

  const groups: [string, ExampleInfo[]][] = [];
  for (const ex of examples ?? []) {
    const last = groups[groups.length - 1];
    if (last && last[0] === ex.group) last[1].push(ex);
    else groups.push([ex.group, [ex]]);
  }

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
          {groups.map(([g, items]) => (
            <div key={g} className="example-group">
              <button
                className={`example-group-head${group === g ? " open" : ""}`}
                aria-expanded={group === g}
                onClick={() => toggleGroup(g)}
              >
                <span className="example-group-caret">{group === g ? "▾" : "▸"}</span>
                <span className="example-group-name">{g}</span>
                <span className="muted small">{items.length}</span>
              </button>
              {group === g && items.map((ex) => (
                <button
                  key={ex.id}
                  role="menuitem"
                  className="example-item"
                  title={ex.download_mb ? `Downloaded from Zenodo on first use (${ex.download_mb} MB)` : undefined}
                  onClick={() => {
                    setOpen(false);
                    openExample(ex.id, ex.title);
                  }}
                >
                  <span className="example-title">
                    {ex.title}
                    {ex.download_mb != null && <span className="example-download"> ↓ {ex.download_mb} MB</span>}
                  </span>
                  <span className="example-desc">{ex.description}</span>
                </button>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
