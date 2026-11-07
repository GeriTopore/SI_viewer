import { useEffect, useState } from "react";
import { api, type ExportFormat } from "./api/client";

/** A file name from a title, e.g. "Fe Kα (fit)" → "Fe_Kα_fit". */
export const safeFileName = (title: string) => title.replace(/[^\p{L}\p{N}.-]+/gu, "_").replace(/^_+|_+$/g, "") || "data";

/** Let the browser download ``url`` (the server names the file). */
export function downloadUrl(url: string) {
  const a = document.createElement("a");
  a.href = url;
  a.download = "";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

export function downloadBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Rows of cells as CSV (cells quoted when needed). */
export function toCsv(rows: (string | number | null | undefined)[][]): string {
  const cell = (v: string | number | null | undefined) => {
    const s = v === null || v === undefined || (typeof v === "number" && !Number.isFinite(v)) ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return rows.map((r) => r.map(cell).join(",")).join("\n");
}

export const downloadCsv = (rows: Parameters<typeof toCsv>[0], name: string) =>
  downloadBlob(new Blob([toCsv(rows)], { type: "text/csv" }), name);

let formats: Promise<Record<string, ExportFormat[]>> | null = null;

/** Formats each kind of dataset can be saved in (fetched once). */
export function useExportFormats(): Record<string, ExportFormat[]> {
  const [value, setValue] = useState<Record<string, ExportFormat[]>>({});
  useEffect(() => {
    formats ??= api.exportFormats().catch((e) => {
      formats = null;
      throw e;
    });
    formats.then(setValue).catch(() => {});
  }, []);
  return value;
}
