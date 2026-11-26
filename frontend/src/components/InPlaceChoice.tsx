import { useState } from "react";

/** Operations that change loaded data, each with its own remembered choice. */
export type InPlaceOp = "rebin" | "zlp" | "align" | "repair";

const key = (op: InPlaceOp) => `si-viewer:in-place:${op}`;

/** Whether ``op`` runs in place (else as a new dataset): the last choice made
 * in this browser, in place at first. */
export function useInPlace(op: InPlaceOp): [boolean, (v: boolean) => void] {
  const [value, setValue] = useState(() => {
    try {
      return localStorage.getItem(key(op)) !== "new";
    } catch {
      return true;
    }
  });
  const set = (v: boolean) => {
    setValue(v);
    try {
      localStorage.setItem(key(op), v ? "in-place" : "new");
    } catch {
      /* not kept: fine */
    }
  };
  return [value, set];
}

/** "In place" checkbox for an operation that changes loaded data: unticked,
 * the result is a new dataset and the source is left as it is. */
export function InPlaceChoice({ inPlace, onChange, disabled, what = "the data" }: {
  inPlace: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
  what?: string;
}) {
  return (
    <label
      className="q-check"
      title={inPlace ? `Changes ${what} itself` : `Leaves ${what} as it is and adds the result under it`}
    >
      <input type="checkbox" checked={inPlace} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      In place
    </label>
  );
}
