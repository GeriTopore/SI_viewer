import type { ReactNode } from "react";
import { useUi, type BottomTab } from "../state/ui";

/** The plot under the image that goes with a process. */
const PROCESS_TABS: Record<string, BottomTab> = { neighbourhoods: "neighbourhood", sublattices: "cell" };

/** Working in a process brings its plot forward under the image (when shown). */
export function followProcess(id: string | null | undefined) {
  const tab = id ? PROCESS_TABS[id] : undefined;
  const ui = useUi.getState();
  if (tab && ui.bottomTabsShown.includes(tab) && ui.bottomTab !== tab) ui.setBottomTab(tab);
}

/** Open process ``to`` beside the one asking for it (``from``, if any):
 * split off in the Process tab (pinned to ``image``); or, when ``from`` is
 * the split-off process itself, in the tab's own view, so both stay in sight.
 * A process already showing is just brought forward. */
export function openProcess(to: string, from?: string, image?: string | null) {
  const ui = useUi.getState();
  ui.setRightTab("process");
  const split = ui.processSplit;
  if (split?.id === to) {
    followProcess(to);
    return;
  }
  if (from && split?.id === from) {
    ui.setProcessOpen(to);
  } else {
    ui.setProcessSplit({ id: to, side: split?.side ?? "bottom", image: image ?? null });
    if (ui.processOpen === to) ui.setProcessOpen(null);
  }
  followProcess(to);
}

/** A link in a process's text to another process it needs (opened beside it). */
export function ProcessLink({ to, from, image, children }: { to: string; from?: string; image?: string | null; children: ReactNode }) {
  return (
    <button type="button" className="link-button process-link" onClick={() => openProcess(to, from, image)}>
      {children}
    </button>
  );
}
