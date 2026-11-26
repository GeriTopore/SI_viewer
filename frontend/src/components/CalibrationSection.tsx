import { useEffect, useState } from "react";
import type { DatasetSummary } from "../api/client";
import { setQrRotation, useCalibration, type FittedRing, type RingSearch } from "../state/calibration";
import { done } from "./FourDCalibrationInfo";
import { allDatasets, useDatasets } from "../state/datasets";
import { formatValue } from "../format";
import { segmented } from "../views/FourDView";

type How = "ring" | "camera" | "entered";
type Units = "1/nm" | "1/Å";

const post = async (url: string, body: unknown) => {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const out = await res.json();
  if (!res.ok) throw new Error(out.detail ?? res.statusText);
  return out;
};

/** The relativistic electron wavelength (Å) at ``kv`` kV (as the backend's). */
function wavelength(kv: number): number {
  const h = 6.62607015e-34, m = 9.1093837015e-31, e = 1.602176634e-19, c = 299792458;
  const v = kv * 1e3;
  return (h / Math.sqrt(2 * m * e * v * (1 + (e * v) / (2 * m * c * c)))) * 1e10;
}

/** Calibrating 4D-STEM patterns, step by step: (1) each position's origin
 * and the descan, (2) the ellipticity: a ring of the Bragg vector map (or
 * the max / mean pattern) fitted as an ellipse (the annulus searched and the
 * ellipse drawn on the pattern), then kept, (3) the QR rotation (the least
 * curl of the deflection step 1 measured, or entered), (4) the pixel size
 * from the ring and a standard's reflection, from the camera length, or
 * entered, setting the pattern axes (the data isn't resampled). The order
 * applied to the vectors: origin, ellipse, rotation, pixel size. A step's
 * heading is ticked once its calibration is kept. */
export function CalibrationSection({ d }: { d: DatasetSummary }) {
  const [w] = [d.signal_shape[0] ?? 128];
  const search: RingSearch = useCalibration((s) => s.search[d.id]) ?? { r_min: Math.round(w * 0.15), r_max: Math.round(w * 0.25), source: "bvm" };
  const ring = useCalibration((s) => s.ring[d.id] ?? null);
  const { setSearch, setGuess, setRing } = useCalibration.getState();
  const [standards, setStandards] = useState<Record<string, Record<string, number>>>({});
  const [how, setHow] = useState<How>("ring");
  const [standard, setStandard] = useState("Au");
  const [hkl, setHkl] = useState("111");
  const [camera, setCamera] = useState({ length: 100, pixel: 15, kv: 200 });
  const [entered, setEntered] = useState<number | "">("");
  const [units, setUnits] = useState<Units>(d.axes.some((a) => a.units === "1/Å" || a.units === "1/A") ? "1/Å" : "1/nm");
  const [centre, setCentre] = useState(true);
  const [busy, setBusy] = useState<"fit" | "ellipse" | "apply" | null>(null);
  const files = useDatasets((s) => s.files);
  const hasVectors = allDatasets(files).some((x) => x.origin === "disks" && x.parent_id === d.id);
  const beamFound = allDatasets(files).some((x) => x.origin === "beam-shift" && x.parent_id === d.id);
  // The map's origin per position: as step 1 measured it, else the beam found, else the direct beam.
  const origin = search.origin ?? (d.calibration?.origin ? (d.calibration.origin.source === "beam" ? "beam" : "direct") : beamFound ? "beam" : "direct");
  const kept = d.calibration?.ring ?? null;
  // The pixel size from a standard: the ring kept (step 2), else the one just fitted.
  const sized = kept ?? ring;
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/api/datasets/${d.id}/4d/calibration/standards`).then((r) => r.json()).then(setStandards).catch(() => undefined);
    fetch(`/api/datasets/${d.id}/4d/bf-disk`).then((r) => (r.ok ? r.json() : null)).then((b) => b && setGuess(d.id, { cx: b.cx, cy: b.cy })).catch(() => undefined);
  }, [d.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const fit = async () => {
    setBusy("fit");
    setError(null);
    try {
      const r: FittedRing = await post(`/api/datasets/${d.id}/4d/calibration/ring`, {
        options: { r_min: search.r_min, r_max: search.r_max }, source: search.source,
        origin: { origin, plane: { mode: "plane" } },
      });
      setRing(d.id, r);
    } catch (e) {
      setError((e as Error).message);
      setRing(d.id, null);
    } finally {
      setBusy(null);
    }
  };

  const setEllipse = async () => {
    if (!ring) return;
    setBusy("ellipse");
    setError(null);
    try {
      const out = await post(`/api/datasets/${d.id}/4d/calibration/ellipse`, { ring });
      useDatasets.getState().replaceFile(out.file);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };
  const ellipseKept = !!ring && !!kept && (["cx", "cy", "a", "b", "angle"] as const).every((k) => Math.abs(ring[k] - kept[k]) < 1e-6);

  const d_spacing = standards[standard]?.[hkl];
  const per = units === "1/nm" ? 10 : 1; // 1/Å → the units chosen
  const scale =
    how === "ring" ? (sized && d_spacing ? (per / d_spacing) / sized.radius : null)
      : how === "camera" ? ((camera.pixel * 1e-3) / camera.length / wavelength(camera.kv)) * per
        : entered === "" ? null : entered;

  const apply = async () => {
    setBusy("apply");
    setError(null);
    try {
      const out = await post(`/api/datasets/${d.id}/4d/calibration`, {
        how, ring: sized, units, centre: centre && !!sized,
        ...(how === "ring" && { d: d_spacing, standard, hkl }),
        ...(how === "camera" && { camera_length: camera.length, pixel: camera.pixel, kv: camera.kv }),
        ...(how === "entered" && { scale: entered }),
      });
      useDatasets.getState().replaceFile(out.file);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const num = (label: string, title: string, value: number, set: (v: number) => void, step = 1) => (
    <label title={title}>
      <span>{label}</span>
      <input type="number" step={step} value={value}
        onChange={(e) => e.target.value !== "" && Number(e.target.value) > 0 && set(Number(e.target.value))} />
    </label>
  );

  return (
    <section className="q-form virtual-form">
      <OriginStep d={d} hasVectors={hasVectors} beamFound={beamFound} />
      <h4 className="process-heading">Step 2 · Ellipticity{done(!!kept)}</h4>
      <div className="virtual-row">
        {segmented("Pattern", [
          { id: "max", label: "Max", title: "The maximum over the scan: sparse spots (single crystals)" },
          { id: "mean", label: "Mean", title: "The mean over the scan: rings (polycrystals, amorphous)" },
          {
            id: "bvm", label: "BVM", disabled: !hasVectors,
            title: hasVectors ? "The Bragg vector map, each position moved by its origin: the sharpest rings" : "Find the disks first (Disk detection)",
          },
        ], search.source, (v) => setSearch(d.id, { ...search, source: v }))}
      </div>
      {search.source === "bvm" && (
        <div className="virtual-row">
          {segmented("Origin", [
            { id: "beam", label: "Beam", title: "Each position's beam found by Fix descan (step 1), descan plane", disabled: !beamFound },
            { id: "direct", label: "Direct beam", title: "Each position's own direct-beam disk, descan plane" },
          ], origin, (v) => setSearch(d.id, { ...search, origin: v }))}
        </div>
      )}
      {num("Inner radius (px)", "The annulus searched for the ring, from the bright-field disk's centre", search.r_min, (v) => setSearch(d.id, { ...search, r_min: v }))}
      {num("Outer radius (px)", "The annulus searched for the ring", search.r_max, (v) => setSearch(d.id, { ...search, r_max: v }))}
      <div className="process-actions">
        <button className={ring ? "secondary" : "primary"} disabled={!!busy} onClick={() => void fit()}
          title="Fit the ring in the annulus as an ellipse: its centre, radius and ellipticity, drawn on the pattern">
          {busy === "fit" ? "Fitting…" : ring ? "Fit again" : "Fit ring"}
        </button>
        <button className={ellipseKept ? "secondary" : "primary"} disabled={!ring || !!busy || ellipseKept} onClick={() => void setEllipse()}
          title="Keep the ellipse: the Bragg vectors made circular (before the rotation and pixel size)">
          {busy === "ellipse" ? "Setting…" : ellipseKept ? "Set" : "Set ellipse"}
        </button>
      </div>
      {ring && (
        <p className="muted small">
          r {formatValue(ring.radius)} px · ellipticity {formatValue(ring.ellipticity * 100)} % · θ {formatValue(Math.round((ring.angle * 180) / Math.PI * 10) / 10)}° · fit {formatValue(ring.rms)} px RMS
        </p>
      )}

      <RotationStep d={d} />

      <h4 className="process-heading">Step 4 · Pixel size{done(!!d.calibration?.scale)}</h4>
      <div className="virtual-row">
        {segmented("From", [
          { id: "ring", label: "Standard", title: "The ring is a standard's known reflection" },
          { id: "camera", label: "Camera", title: "From the camera length, detector pixel and voltage" },
          { id: "entered", label: "Entered", title: "Entered by hand" },
        ], how, setHow)}
      </div>
      {how === "ring" && (
        <>
          <label title="The calibration standard">
            <span>Standard</span>
            <select value={standard} onChange={(e) => { setStandard(e.target.value); setHkl(Object.keys(standards[e.target.value] ?? {})[0] ?? ""); }}>
              {Object.keys(standards).map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </label>
          <label title="The reflection the fitted ring is">
            <span>Reflection</span>
            <select value={hkl} onChange={(e) => setHkl(e.target.value)}>
              {Object.entries(standards[standard] ?? {}).map(([k, dd]) => (
                <option key={k} value={k}>{k} · d {dd} Å</option>
              ))}
            </select>
          </label>
        </>
      )}
      {how === "camera" && (
        <>
          {num("Camera length (mm)", "The camera length", camera.length, (v) => setCamera({ ...camera, length: v }), 10)}
          {num("Detector pixel (µm)", "The detector's pixel size (binned: times the binning)", camera.pixel, (v) => setCamera({ ...camera, pixel: v }), 1)}
          {num("Voltage (kV)", "The accelerating voltage", camera.kv, (v) => setCamera({ ...camera, kv: v }), 20)}
        </>
      )}
      {how === "entered" && (
        <label title="The pattern's pixel size">
          <span>Pixel size ({units})</span>
          <input type="number" step={0.001} value={entered}
            onChange={(e) => setEntered(e.target.value === "" ? "" : Math.max(0, Number(e.target.value)))} />
        </label>
      )}
      <div className="virtual-row">
        {segmented("Units", [{ id: "1/nm", label: "1/nm" }, { id: "1/Å", label: "1/Å" }], units, setUnits)}
        <span className="muted small fourd-readout">{scale ? `${formatValue(scale)} ${units} per px` : how === "ring" ? "Fit a ring first (step 2)" : ""}</span>
      </div>
      <label className="q-check-line small" title="The pattern axes read 0 at the ring's centre">
        <input type="checkbox" checked={centre} disabled={!sized} onChange={(e) => setCentre(e.target.checked)} />
        0 at the ring's centre
      </label>
      {d.calibration?.scale && (
        <p className="muted small">
          Calibrated: {formatValue(d.calibration.scale)} {d.calibration.units} per px
          {d.calibration.how?.standard ? ` (${d.calibration.how.standard} ${d.calibration.how.hkl})` : ""}
          {d.calibration.ring ? ` · ellipticity ${formatValue(d.calibration.ring.ellipticity * 100)} %` : ""}
        </p>
      )}
      {error && <p className="field-error">{error}</p>}
      <div className="process-actions">
        <button className="primary" disabled={!scale || !!busy} onClick={() => void apply()}
          title="Set the pattern axes' scale (and 0 at the ring's centre); the data isn't resampled">
          {busy === "apply" ? "Calibrating…" : "Calibrate"}
        </button>
      </div>
    </section>
  );
}

type OriginSource = "vectors" | "beam";
type PlaneMode = "none" | "plane" | "corners";

/** Step 1: each scan position's origin, py4DSTEM-style (measure_origin /
 * fit_origin): measured from the direct-beam disk among its Bragg vectors,
 * or the beam found by Centring; then a descan plane fitted (whole scan or
 * corners) or each position kept. "Origin x / y" and "(fit)" maps join the
 * dataset pane; the origin is applied to the Bragg vectors (Info tab). */
function OriginStep({ d, hasVectors, beamFound }: { d: DatasetSummary; hasVectors: boolean; beamFound: boolean }) {
  const made = d.calibration?.origin;
  const [source, setSource] = useState<OriginSource>(made?.source ?? (hasVectors ? "vectors" : "beam"));
  const [plane, setPlane] = useState<PlaneMode>(made?.plane.mode ?? "plane");
  const [corners, setCorners] = useState(made?.plane.corners ?? 0.1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const can = source === "vectors" ? hasVectors : beamFound;

  const measure = async () => {
    setBusy(true);
    setError(null);
    try {
      const out = await post(`/api/datasets/${d.id}/4d/calibration/origin`, { source, plane: { mode: plane, corners } });
      useDatasets.getState().replaceFile(out.file);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <h4 className="process-heading">Step 1 · Descan{done(!!made)}</h4>
      <div className="virtual-row">
        {segmented("Measured from", [
          {
            id: "vectors", label: "Bragg vectors", disabled: !hasVectors,
            title: hasVectors ? "Each position's direct-beam disk among its Bragg vectors" : "Find the disks first (Disk detection)",
          },
          {
            id: "beam", label: "Beam found", disabled: !beamFound,
            title: beamFound ? "The beam found by Fix descan (step 1)" : "Find the beam first (Fix descan, step 1)",
          },
        ], source, setSource)}
      </div>
      <div className="virtual-row">
        {segmented("Fit", [
          { id: "none", label: "Each position", title: "Each position's origin as measured" },
          { id: "plane", label: "Plane", title: "A linear plane fitted to the origins over the scan (descan)" },
          { id: "corners", label: "Corners", title: "A plane fitted to the scan's corners only" },
        ], plane, setPlane)}
      </div>
      {plane === "corners" && (
        <label title="The part of each side of the scan taken as its corners">
          <span>Corners (fraction)</span>
          <input type="number" min={0.02} max={0.5} step={0.05} value={corners}
            onChange={(e) => e.target.value !== "" && setCorners(Math.min(0.5, Math.max(0.02, Number(e.target.value))))} />
        </label>
      )}
      <div className="process-actions">
        <button className={made ? "secondary" : "primary"} disabled={!can || busy} onClick={() => void measure()}
          title="Measure (and fit) the origin per scan position: Origin x / y maps join the dataset pane; applied to the Bragg vectors">
          {busy ? "Measuring…" : made ? "Measure again" : "Measure origin"}
        </button>
      </div>
      {made && (
        <p className="muted small">
          Origin ({formatValue(made.mean[0])}, {formatValue(made.mean[1])}) px
          {made.plane.mode !== "none" ? ` · fit ${formatValue(made.rms)} px RMS` : ""}
        </p>
      )}
      {error && <p className="field-error">{error}</p>}
    </>
  );
}

interface FoundRotation {
  theta: number;
  mirror: boolean;
  curl_ratio: number;
}

/** Step 3: the QR rotation (the pattern's axes into the scan's) and
 * mirror: the ones leaving the least curl in the deflection step 1 measured
 * (each origin less the descan plane), or entered; kept in the calibration
 * (switched on), turning the Bragg vectors into the scan's frame. */
function RotationStep({ d }: { d: DatasetSummary }) {
  const made = d.calibration?.rotation;
  const measured = !!d.calibration?.origin;
  const [theta, setTheta] = useState<number>(made?.theta ?? 0);
  const [mirror, setMirror] = useState<boolean>(made?.mirror ?? false);
  const [found, setFound] = useState<FoundRotation | null>(null);
  const [busy, setBusy] = useState<"find" | "set" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const asFound = !!found && Math.abs(found.theta - theta) < 1e-6 && found.mirror === mirror;
  const kept = !!made && Math.abs(made.theta - (((theta % 360) + 360) % 360)) < 1e-6 && made.mirror === mirror;

  const find = async () => {
    setBusy("find");
    setError(null);
    try {
      const r: FoundRotation = await post(`/api/datasets/${d.id}/4d/calibration/rotation/find`, {});
      setFound(r);
      setTheta(r.theta);
      setMirror(r.mirror);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const keep = async () => {
    setBusy("set");
    setError(null);
    try {
      await setQrRotation(d.id, { theta, mirror, how: asFound ? "curl" : "entered", ...(asFound ? { curl_ratio: found!.curl_ratio } : {}) });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <>
      <h4 className="process-heading">Step 3 · QR rotation{done(!!made)}</h4>
      <label title="The rotation of the pattern's axes into the scan's (degrees, counter-clockwise)">
        <span>θ (°)</span>
        <input type="number" min={-360} max={360} step={0.5} value={Math.round(theta * 100) / 100}
          onChange={(e) => e.target.value !== "" && setTheta(Number(e.target.value))} />
      </label>
      <label className="q-check-line small" title="The pattern's y axis flipped first">
        <input type="checkbox" checked={mirror} onChange={(e) => setMirror(e.target.checked)} />
        Mirror
      </label>
      {found && (
        <p className="muted small">
          Least curl at {formatValue(Math.round(found.theta * 100) / 100)}°{found.mirror ? ", mirrored" : ""} · curl left {formatValue(found.curl_ratio * 100)} %
        </p>
      )}
      {error && <p className="field-error">{error}</p>}
      <div className="process-actions">
        <button className="secondary" disabled={!measured || !!busy} onClick={() => void find()}
          title={measured ? "The rotation and mirror leaving the least curl in the deflection: each origin less the descan plane (step 1)" : "Measure the origin first (step 1)"}>
          {busy === "find" ? "Finding…" : "Find"}
        </button>
        <button className={kept ? "secondary" : "primary"} disabled={!!busy || kept} onClick={() => void keep()}
          title="Keep θ and the mirror as the QR rotation: the Bragg vectors turned into the scan's frame">
          {busy === "set" ? "Setting…" : kept ? "Set" : "Set rotation"}
        </button>
      </div>
    </>
  );
}
