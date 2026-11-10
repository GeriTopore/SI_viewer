import { useEffect, useRef, useState } from "react";
import { api, type DatasetSummary } from "../api/client";
import { useDatasets } from "../state/datasets";
import { useLayers } from "../state/layers";

/** A field renaming ``d`` in place: Enter or leaving it saves, Escape cancels.
 * ``onDone`` is called either way (the caller goes back to showing the title). */
export function RenameInput({ d, onDone }: { d: DatasetSummary; onDone: () => void }) {
  const [value, setValue] = useState(d.title);
  const ref = useRef<HTMLInputElement>(null);
  const finished = useRef(false);
  useEffect(() => {
    ref.current?.focus();
    ref.current?.select();
  }, []);

  const finish = async (save: boolean) => {
    if (finished.current) return;
    finished.current = true;
    onDone();
    const title = value.trim();
    if (!save || !title || title === d.title) return;
    try {
      useDatasets.getState().updateDataset(await api.rename(d.id, title));
    } catch (e) {
      useLayers.getState().notify(`Could not rename ${d.title}: ${(e as Error).message}`);
    }
  };

  return (
    <input
      ref={ref}
      className="rename-input"
      value={value}
      aria-label="Name"
      onChange={(e) => setValue(e.target.value)}
      onBlur={() => finish(true)}
      // Keys and clicks stay in the field (not the row's Enter/Delete or the
      // viewer's shortcuts; not selecting the row).
      onKeyDown={(e) => {
        e.stopPropagation();
        if (e.key === "Enter") finish(true);
        else if (e.key === "Escape") finish(false);
      }}
      onClick={(e) => e.stopPropagation()}
      onDoubleClick={(e) => e.stopPropagation()}
      onPointerDown={(e) => e.stopPropagation()}
    />
  );
}

/** A dataset's title that turns into a rename field when clicked. */
export function EditableTitle({ d }: { d: DatasetSummary }) {
  const [editing, setEditing] = useState(false);
  if (editing) return <RenameInput d={d} onDone={() => setEditing(false)} />;
  return (
    <span className="editable-title" role="button" tabIndex={0} title="Rename" onClick={() => setEditing(true)}
      onKeyDown={(e) => (e.key === "Enter" || e.key === "F2") && setEditing(true)}>
      {d.title}
    </span>
  );
}
