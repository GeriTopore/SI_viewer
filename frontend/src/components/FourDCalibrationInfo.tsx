import { useState } from "react";
import type { DatasetSummary } from "../api/client";
import { setCalibrationSwitches } from "../state/calibration";
import { allDatasets, useDatasets } from "../state/datasets";
import { formatValue } from "../format";

type Part = "origin" | "ellipse" | "scale" | "rotation";

/** The 4D-STEM data a dataset belongs to: itself, or the data its Bragg
 * vectors, their map, its origin or beam-shift maps were made from. */
export function fourDOf(d: DatasetSummary, all: DatasetSummary[]): DatasetSummary | null {
  let cur: DatasetSummary | undefined = d;
  for (let i = 0; cur && i < 4; i++) {
    if (cur.kind === "4dstem") return cur;
    cur = all.find((x) => x.id === cur!.parent_id);
  }
  return null;
}

const deg = (rad: number) => formatValue((rad * 180) / Math.PI);

/** 4D-STEM data's calibration, as py4DSTEM keeps it: the Q (pattern) and R
 * (scan) pixel sizes, each scan position's origin, the ellipse, the QR
 * rotation; each part applied to the Bragg vectors can be switched off and
 * on, the Bragg vector map following at once. */
export function FourDCalibrationInfo({ d }: { d: DatasetSummary }) {
  const files = useDatasets((s) => s.files);
  const cube = fourDOf(d, allDatasets(files));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!cube) return null;
  const cal = cube.calibration ?? {};
  const on = cal.enabled ?? {};
  const hasVectors = allDatasets(files).some((x) => x.origin === "disks" && x.parent_id === cube.id);
  const scan = cube.axes.filter((a) => a.navigate);

  const toggle = async (part: Part, value: boolean) => {
    setBusy(true);
    setError(null);
    try {
      await setCalibrationSwitches(cube.id, { enabled: { [part]: value } });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const sw = (part: Part, available: boolean, title: string) => (
    <input
      type="checkbox"
      className="calibration-switch"
      checked={available && !!on[part]}
      disabled={!available || busy || !hasVectors}
      title={hasVectors ? title : "Applied to the Bragg vectors: find the disks first (Disk detection)"}
      onChange={(e) => void toggle(part, e.target.checked)}
    />
  );

  return (
    <section className="calibration-info">
      <h4 className="process-heading">Calibration</h4>
      <dl className="props">
        <dt>Q pixel size</dt>
        <dd>
          {sw("scale", !!cal.scale, "The pattern's pixel size applied to the Bragg vectors")}
          {cal.scale ? (
            <>
              {formatValue(cal.scale)} {cal.units}
              {cal.how?.standard && <span className="muted"> · {cal.how.standard} {cal.how.hkl}</span>}
              {cal.how?.camera_length && <span className="muted"> · {cal.how.camera_length} mm, {cal.how.kv} kV</span>}
            </>
          ) : <span className="muted">not calibrated (Calibration process)</span>}
        </dd>
        <dt>R pixel size</dt>
        <dd>
          {scan.length ? `${formatValue(scan[0].scale)} ${scan[0].units || "px"}` : "–"}
          <span className="muted"> · the scan's axes</span>
        </dd>
        <dt>Centring</dt>
        <dd>
          {sw("origin", !!cal.origin, "Subtract each scan position's fitted origin from its Bragg vectors (descan)")}
          {cal.origin ? (
            <>
              {cal.origin.source === "vectors" ? "Bragg vectors" : "beam found"}
              {cal.origin.plane.mode !== "none" && `, ${cal.origin.plane.mode === "plane" ? "plane" : "corner plane"} (${formatValue(cal.origin.rms)} px RMS)`}
              <span className="muted"> · mean ({formatValue(cal.origin.mean[0])}, {formatValue(cal.origin.mean[1])}) px</span>
            </>
          ) : <span className="muted">not measured (Calibration process)</span>}
        </dd>
        <dt>Ellipse</dt>
        <dd>
          {sw("ellipse", !!cal.ring, "The ring's ellipse made circular in the Bragg vectors")}
          {cal.ring ? (
            <>
              a {formatValue(cal.ring.a)}, b {formatValue(cal.ring.b)} px, θ {deg(cal.ring.angle)}°
              <span className="muted"> · {formatValue(cal.ring.ellipticity * 100)} %</span>
            </>
          ) : <span className="muted">not fitted (Calibration process)</span>}
        </dd>
        <dt>QR rotation</dt>
        <dd>
          {sw("rotation", !!cal.rotation, "The pattern's axes turned into the scan's in the Bragg vectors")}
          {cal.rotation ? (
            <>
              {formatValue(cal.rotation.theta)}°{cal.rotation.mirror ? ", mirrored" : ""}
              <span className="muted">
                {" "}· {cal.rotation.how === "com" ? "least curl (DPC / iCOM)" : "entered (DPC / iCOM)"}
                {cal.rotation.curl_ratio != null && `, curl left ${formatValue(cal.rotation.curl_ratio * 100)} %`}
              </span>
            </>
          ) : <span className="muted">not measured (DPC / iCOM process)</span>}
        </dd>
      </dl>
      {error && <p className="field-error">{error}</p>}
    </section>
  );
}
