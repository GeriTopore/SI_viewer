import { useEffect, useRef, useState, type DragEvent, type ReactNode, type RefObject } from "react";
import { useShortcuts } from "../keyboard";
import { PANE_SIZES, useUi, type BottomTab, type SplitSide } from "../state/ui";
import { Splitter } from "./Splitter";

/** Drag-and-drop type for the pane's tabs. */
const TAB_MIME = "application/x-si-viewer-tab";

export interface PaneTab {
  id: BottomTab;
  label: string;
  content: ReactNode;
}

interface Props {
  tabs: PaneTab[];
  /** The viewer holding the image above: the pane's height is a fraction of it. */
  viewerRef: RefObject<HTMLElement>;
  /** Label of the handle between the image and the pane. */
  resizeLabel: string;
}

/** The pane under the image: tabs, any two of which can be shown side by side
 * or stacked (the Split button, or dragging a tab onto the pane). */
export function BottomPane({ tabs, viewerRef, resizeLabel }: Props) {
  const bottomTab = useUi((s) => s.bottomTab);
  const bottomSplit = useUi((s) => s.bottomSplit);
  const bottomFraction = useUi((s) => s.bottomFraction);
  const splitDir = useUi((s) => s.splitDir);
  const splitFraction = useUi((s) => s.splitFraction);
  const splitRowFraction = useUi((s) => s.splitRowFraction);
  const setBottomTab = useUi((s) => s.setBottomTab);
  const focusBottomTab = useUi((s) => s.focusBottomTab);
  const splitBottom = useUi((s) => s.splitBottom);
  const unsplitBottom = useUi((s) => s.unsplitBottom);
  const placeBottomTab = useUi((s) => s.placeBottomTab);
  const setBottomFraction = useUi((s) => s.setBottomFraction);
  const setSplitFraction = useUi((s) => s.setSplitFraction);
  const setSplitRowFraction = useUi((s) => s.setSplitRowFraction);

  // Split only if the view has both tabs; the stored split is kept otherwise.
  const ids = tabs.map((t) => t.id);
  const split =
    bottomSplit && bottomSplit[0] !== bottomSplit[1] && bottomSplit.every((id) => ids.includes(id))
      ? bottomSplit
      : null;
  const focused: BottomTab = split
    ? split.includes(bottomTab)
      ? bottomTab
      : split[0]
    : ids.includes(bottomTab)
      ? bottomTab
      : (bottomSplit?.find((id) => ids.includes(id)) ?? ids[0]);
  const next = (id: BottomTab, among = ids) => among[(among.indexOf(id) + 1) % among.length];
  const content = (id: BottomTab) => tabs.find((t) => t.id === id)?.content;

  // Keep the stored focus on a shown tab, so setBottomTab replaces the right half.
  useEffect(() => {
    if (split && !split.includes(bottomTab)) focusBottomTab(split[0]);
  }, [split, bottomTab, focusBottomTab]);

  const focus = (id: BottomTab) => {
    if (split) setBottomTab(id);
    else focusBottomTab(id);
  };

  // T cycles the focused half's tab, skipping the one in the other half.
  useShortcuts(
    {
      t: () => {
        const other = split?.find((id) => id !== focused);
        focus(next(focused, ids.filter((id) => id !== other)));
      },
    },
    tabs.length > 1,
  );

  // Dragging a tab: over the body to split, over the tab bar to unsplit.
  const [dragTab, setDragTab] = useState<BottomTab | null>(null);
  const [hover, setHover] = useState<SplitSide | "bar" | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const tabDrag = (e: DragEvent) => dragTab !== null && e.dataTransfer.types.includes(TAB_MIME);
  // Left / right third: two columns; middle third: two rows (upper / lower half).
  const sideOf = (e: DragEvent): SplitSide => {
    const r = bodyRef.current!.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width;
    if (x < 1 / 3) return "left";
    if (x > 2 / 3) return "right";
    return e.clientY < r.top + r.height / 2 ? "top" : "bottom";
  };
  const endDrag = () => {
    setDragTab(null);
    setHover(null);
  };
  const dropOnBody = (e: DragEvent) => {
    if (!tabDrag(e)) return;
    e.preventDefault();
    const side = sideOf(e);
    if (split) placeBottomTab(dragTab!, side);
    else {
      const partner = focused !== dragTab ? focused : next(dragTab!);
      const dir = side === "left" || side === "right" ? "cols" : "rows";
      if (side === "left" || side === "top") splitBottom(dragTab!, partner, dragTab!, dir);
      else splitBottom(partner, dragTab!, dragTab!, dir);
    }
    endDrag();
  };
  const dropOnBar = (e: DragEvent) => {
    if (!split || !tabDrag(e)) return;
    e.preventDefault();
    unsplitBottom(split.includes(dragTab!) ? split.find((id) => id !== dragTab)! : dragTab!);
    endDrag();
  };

  const halves = split ?? [focused];
  const rows = splitDir === "rows";
  return (
    <>
      <Splitter
        direction="row"
        className="viewer-resizer"
        label={resizeLabel}
        onDrag={(y) => {
          const r = viewerRef.current!.getBoundingClientRect();
          setBottomFraction((r.bottom - y) / r.height);
        }}
        onReset={() => setBottomFraction(PANE_SIZES.bottom.initial)}
      />
      <div className="viewer-bottom" style={{ flexBasis: `${bottomFraction * 100}%` }}>
        <div
          className={`viewer-tabs${hover === "bar" ? " drop-target" : ""}`}
          role="tablist"
          onDragOver={(e) => {
            if (!split || !tabDrag(e)) return;
            e.preventDefault();
            e.dataTransfer.dropEffect = "move";
            setHover("bar");
          }}
          onDragLeave={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget as Node)) setHover(null);
          }}
          onDrop={dropOnBar}
        >
          {tabs.map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={halves.includes(t.id)}
              className={t.id === focused ? "active" : halves.includes(t.id) ? "shown" : undefined}
              draggable={tabs.length > 1}
              onDragStart={(e) => {
                e.dataTransfer.setData(TAB_MIME, t.id);
                e.dataTransfer.effectAllowed = "move";
                setDragTab(t.id);
              }}
              onDragEnd={endDrag}
              onClick={() => focus(t.id)}
            >
              {t.label}
            </button>
          ))}
          {tabs.length > 1 && (
            <button
              className="split-toggle"
              title="Show two tabs side by side (or drag a tab onto the pane)"
              onClick={() => (split ? unsplitBottom(focused) : splitBottom(focused, next(focused)))}
            >
              {split ? "Unsplit" : "Split"}
            </button>
          )}
        </div>
        <div ref={bodyRef} className={`viewer-tab-body${split ? ` split ${splitDir}` : ""}`}>
          {/* Keyed halves: splitting or unsplitting keeps the focused tab's plot (and its zoom). */}
          {halves.map((id, i) => [
            i === 1 && (
              <Splitter
                key="split-resizer"
                direction={rows ? "row" : "col"}
                className={rows ? "split-row-resizer" : "split-resizer"}
                label="Resize the two halves of the pane"
                onDrag={(pos) => {
                  const r = bodyRef.current!.getBoundingClientRect();
                  if (rows) setSplitRowFraction((pos - r.top) / r.height);
                  else setSplitFraction((pos - r.left) / r.width);
                }}
                onReset={() => (rows ? setSplitRowFraction : setSplitFraction)(PANE_SIZES[rows ? "splitRow" : "split"].initial)}
              />
            ),
            <div
              key={id}
              className={`viewer-half${i === 1 ? " second" : ""}${split && focused === id ? " focused" : ""}`}
              style={split && i === 0 ? { flexBasis: `${(rows ? splitRowFraction : splitFraction) * 100}%` } : undefined}
              onPointerDownCapture={() => focused !== id && focus(id)}
            >
              {content(id)}
            </div>,
          ])}
          {dragTab && tabs.length > 1 && (
            <div
              className="split-drop"
              onDragOver={(e) => {
                if (!tabDrag(e)) return;
                e.preventDefault();
                e.dataTransfer.dropEffect = "move";
                setHover(sideOf(e));
              }}
              onDragLeave={() => setHover(null)}
              onDrop={dropOnBody}
            >
              {hover && hover !== "bar" && <div className={`hot ${hover}`} />}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
