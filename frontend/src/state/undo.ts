import { create } from "zustand";
import { useLayers } from "./layers";

/** A deletion that can be undone: what it was (for the notices), and how to bring it back. */
interface UndoEntry {
  label: string;
  restore: () => void | Promise<void>;
}

// Deletions kept (the backend keeps its last 20).
const LIMIT = 50;

interface UndoStore {
  stack: UndoEntry[];
  /** Record a deletion, and say it can be undone. */
  push: (label: string, restore: UndoEntry["restore"]) => void;
  /** Undo the last deletion (⌘/Ctrl Z). */
  undo: () => Promise<void>;
}

export const useUndo = create<UndoStore>((set, get) => ({
  stack: [],
  push: (label, restore) => {
    set({ stack: [...get().stack, { label, restore }].slice(-LIMIT) });
    useLayers.getState().notify(`Deleted ${label} · ⌘/Ctrl Z to undo`);
  },
  undo: async () => {
    const entry = get().stack.at(-1);
    const notify = useLayers.getState().notify;
    if (!entry) {
      notify("Nothing to undo");
      return;
    }
    set({ stack: get().stack.slice(0, -1) });
    try {
      await entry.restore();
      notify(`Restored ${entry.label}`);
    } catch (e) {
      notify(`Could not restore ${entry.label}: ${(e as Error).message}`);
    }
  },
}));

/** ``items`` with ``item`` put back at ``index`` (or at the end), unless it is there already. */
export function putBack<T extends { id: string }>(items: T[], item: T, index: number): T[] {
  if (items.some((x) => x.id === item.id)) return items;
  const out = [...items];
  out.splice(Math.min(index, out.length), 0, item);
  return out;
}
