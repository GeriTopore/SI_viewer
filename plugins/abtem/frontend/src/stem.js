// STEM imaging of a Potential: the probe and scan form, detectors added to a
// run from its polar data, and post-processing of its images.

const API = "/api/plugins/abtem";
const ENERGY = { initial: 300, min: 20, max: 1000 };
const STEM = {
  semiangle: 20,
  // The cell a probe is given by default, so it doesn't wrap into itself.
  probeExtent: 20,
  maxTile: 64,
  minSampling: 0.01,
  step: 1,
  maxGpts: 4096,
  maxBytes: 2 * 1024 ** 3,
  batchBytes: 256 * 1024 ** 2,
};
// Detectors a run starts with: abTEM's scan and detect walkthrough (BF, MAADF,
// HAADF; the quickstart's BF is 0 to the convergence) and the usual ABF, half to
// all of the convergence. Outer angles are cut to what the sampling allows.
const DETECTORS = [
  { name: "BF", inner: () => 0, outer: (alpha) => alpha },
  { name: "ABF", inner: (alpha) => alpha / 2, outer: (alpha) => alpha },
  { name: "MAADF", inner: () => 50, outer: () => 120 },
  { name: "HAADF", inner: () => 90, outer: () => 200 },
];
const ANTIALIAS = 2 / 3;
const BYTES = ["B", "kB", "MB", "GB"];

// Badge colours of the STEM objects: apart from the app's and the plugin's.
export const STEM_BADGES = {
  stem: "#b2749a",
  bf: "#857b56",
  abf: "#47362e",
  adf: "#7a994c",
  maadf: "#644970",
  haadf: "#7c0899",
  detector: "#04470a",
};

const wavelength = (kv) => 12.39842 / Math.sqrt(kv * (2 * 510.99895 + kv));
const num = (x, digits = 4) => String(+Number(x).toFixed(digits));
const size = (bytes) => {
  let i = 0;
  while (bytes >= 1024 && i < BYTES.length - 1) {
    bytes /= 1024;
    i += 1;
  }
  return `${bytes < 10 && i > 0 ? bytes.toFixed(1) : Math.round(bytes)} ${BYTES[i]}`;
};

export function activateStem(host) {
  const { React } = host;
  const h = React.createElement;

  const row = (label, value, title, key = label) => [
    h("dt", { key: `${key}-t`, title }, label),
    h("dd", { key: `${key}-d` }, value),
  ];

  /** A probe scanned over a Potential, with annular detectors. */
  function StemSection({ d }) {
    const s = d.plugin.settings;
    const [cellX, cellY] = s.gpts.map((n, i) => n * s.sampling_xy[i]);
    const probeTile = [Math.ceil(STEM.probeExtent / cellX), Math.ceil(STEM.probeExtent / cellY)];
    const [engine, setEngine] = React.useState(null);
    const [energy, setEnergy] = React.useState(String(ENERGY.initial));
    const [semiangle, setSemiangle] = React.useState(String(STEM.semiangle));
    const [defocus, setDefocus] = React.useState("0");
    const [cs, setCs] = React.useState("0");
    const [tile, setTile] = React.useState(probeTile.map(String));
    const [from, setFrom] = React.useState(["0", "0"]);
    const [to, setTo] = React.useState(["1", "1"]);
    const [samplingText, setSamplingText] = React.useState("");
    const [step, setStep] = React.useState(String(STEM.step));
    // What was changed of each detector: {on, inner, outer}; the rest follows the convergence.
    const [edits, setEdits] = React.useState({});
    const [job, setJob] = React.useState(null);
    const [error, setError] = React.useState(null);
    const mounted = React.useRef(true);
    const active = job?.status === "queued" || job?.status === "running";

    const finish = React.useCallback((done) => {
      if (!mounted.current) return;
      setJob(done);
      if (done.status === "failed") setError(done.error);
      if (done.status === "done" && done.result?.file) host.showFile(done.result.file, done.result.dataset_id);
    }, []);

    React.useEffect(() => {
      mounted.current = true;
      let stale = false;
      host.request(`${API}/engine`).then((e) => { if (!stale) setEngine(e); }).catch((e) => { if (!stale) setError(e.message); });
      host.request(`${API}/potentials/${d.id}/stem`)
        .then((r) => {
          if (!stale && r.job) host.followJob(r.job, (j) => { if (mounted.current) setJob(j); }).then(finish).catch((e) => setError(e.message));
        })
        .catch((e) => { if (!stale) setError(e.message); });
      return () => {
        stale = true;
        mounted.current = false;
      };
    }, [d.id, finish]);

    const kv = Number(energy);
    const alpha = Number(semiangle);
    const energyOk = kv >= ENERGY.min && kv <= ENERGY.max;
    const limit = energyOk ? (wavelength(kv) / (2 * s.potential.sampling)) * ANTIALIAS * 1000 : null;
    const alphaOk = alpha > 0 && limit !== null && alpha <= limit;
    const lensOk = defocus.trim() !== "" && Number.isFinite(Number(defocus)) && cs.trim() !== "" && Number.isFinite(Number(cs));
    const tiles = tile.map(Number);
    const tileOk = tiles.every((n) => Number.isInteger(n) && n >= 1 && n <= STEM.maxTile);
    const a = from.map(Number);
    const b = to.map(Number);
    const regionOk = [0, 1].every((i) => from[i].trim() !== "" && to[i].trim() !== "" && a[i] >= 0 && a[i] < b[i] && b[i] <= 1);
    const nyquist = energyOk && alpha > 0 ? (wavelength(kv) * 1000) / (4 * alpha) : null;
    const given = samplingText.trim() !== "";
    const sampling = given ? Number(samplingText) : nyquist;
    const samplingOk = sampling !== null && Number.isFinite(sampling) && sampling >= STEM.minSampling && sampling <= 10;
    const stepValue = Number(step);
    const stepOk = stepValue >= 0.1 && stepValue <= 10;
    const positions = samplingOk && regionOk
      ? [0, 1].map((i) => Math.max(1, Math.ceil(Math.round(((b[i] - a[i]) * (i ? cellY : cellX) / sampling) * 1e9) / 1e9)))
      : null;
    const bins = limit !== null && stepOk ? Math.floor(limit / stepValue) : null;
    const polarBytes = positions && bins ? positions[0] * positions[1] * bins * 4 : null;

    const detectors = DETECTORS.map((base) => {
      const e = edits[base.name] ?? {};
      const outerDefault = limit !== null && alpha > 0 ? Math.min(base.outer(alpha), Math.floor(limit)) : 0;
      const innerDefault = alpha > 0 ? base.inner(alpha) : 0;
      const inner = e.inner ?? num(innerDefault, 2);
      const outer = e.outer ?? num(outerDefault, 2);
      const ok = inner.trim() !== "" && outer.trim() !== "" && Number(inner) >= 0 && Number(inner) < Number(outer);
      const on = e.on ?? Number(inner) < Number(outer);
      return { name: base.name, inner, outer, on, ok, beyond: on && limit !== null && Number(outer) > limit };
    });
    const chosen = detectors.filter((x) => x.on);
    const detectorsOk = chosen.length > 0 && chosen.every((x) => x.ok && !x.beyond);
    const edit = (name, change) => setEdits((all) => ({ ...all, [name]: { ...all[name], ...change } }));

    const tiledX = s.gpts[0] * tiles[0];
    const tiledY = s.gpts[1] * tiles[1];
    const tooLarge = tileOk && Math.max(tiledX, tiledY) > STEM.maxGpts;
    const tooMuchPolar = polarBytes !== null && polarBytes > STEM.maxBytes;
    const small = tileOk && Math.min(cellX * tiles[0], cellY * tiles[1]) < STEM.probeExtent;
    let memory = null;
    if (tileOk && positions && polarBytes !== null) {
      const threads = Math.max(1, (navigator.hardwareConcurrency || 4) - 2);
      const waves = Math.min(threads * STEM.batchBytes, positions[0] * positions[1] * tiledX * tiledY * 8 * 4);
      memory = tiledX * tiledY * s.slices * 4 + polarBytes + waves;
    }
    const valid = energyOk && alphaOk && lensOk && tileOk && regionOk && samplingOk && stepOk && detectorsOk && !tooLarge && !tooMuchPolar;

    const run = async () => {
      setError(null);
      try {
        finish(await host.runJob(`${API}/potentials/${d.id}/stem`, {
          energy_kv: kv,
          semiangle: alpha,
          defocus: Number(defocus),
          cs_um: Number(cs),
          tile: tiles,
          start: a,
          end: b,
          sampling: given ? sampling : null,
          step: stepValue,
          detectors: chosen.map((x) => ({ name: x.name, inner: Number(x.inner), outer: Number(x.outer) })),
        }, (j) => { if (mounted.current) setJob(j); }));
      } catch (e) {
        if (mounted.current) setError(e.message);
      }
    };
    const number = (label, value, set, extra = {}) => h("input", {
      className: "cell-input",
      type: "number",
      value,
      disabled: active,
      "aria-label": label,
      onChange: (e) => set(e.target.value),
      ...extra,
    });
    const pair = (label, values, set, extra = {}) => h("span", { className: "model-zone" },
      [0, 1].map((i) => number(`${label} ${"xy"[i]}`, values[i], (v) => set(values.map((old, j) => (j === i ? v : old))), extra)));
    return h("div", { className: "engine-form model-form" },
      h("dl", { className: "props" },
        row("Energy (kV)", number("STEM energy", energy, setEnergy, { min: ENERGY.min, max: ENERGY.max })),
        row("Convergence (mrad)", number("Probe convergence", semiangle, setSemiangle, { min: 0, max: limit ? Math.floor(limit) : undefined }), limit ? `Up to ${limit.toFixed(0)} mrad at this sampling` : undefined),
        row("Defocus (Å)", number("Probe defocus", defocus, setDefocus, { step: 1 }), "Positive focuses into the specimen"),
        row("Cs (µm)", number("Probe Cs", cs, setCs)),
        row("Tiling", pair("Tiling", tile, setTile, { min: 1, max: STEM.maxTile, step: 1 }), "Repeats of the potential along x and y, for the probe. The scan stays over one cell"),
        row("Scan from", pair("Scan from", from, setFrom, { min: 0, max: 1, step: 0.1 }), "Fractions of the cell"),
        row("Scan to", pair("Scan to", to, setTo, { min: 0, max: 1, step: 0.1 }), "Fractions of the cell"),
        row("Sampling (Å)", number("Scan sampling", samplingText, setSamplingText, {
          min: STEM.minSampling, step: 0.01, placeholder: nyquist ? num(nyquist, 3) : "", title: "Empty: Nyquist sampling of the probe",
        }), "Empty: Nyquist sampling of the probe"),
        row("Radial bin (mrad)", number("Radial bin", step, setStep, { min: 0.1, max: 10, step: 0.1 }), "Width of the bins the scattered beam is saved in, from which any annulus can be integrated later"),
        detectors.map((x) => row(
          h("label", { className: "model-zone" },
            h("input", { type: "checkbox", checked: x.on, disabled: active, "aria-label": `${x.name} detector`, onChange: (e) => edit(x.name, { on: e.target.checked }) }),
            x.name),
          h("span", { className: "model-zone" },
            number(`${x.name} inner`, x.inner, (v) => edit(x.name, { inner: v }), { min: 0, step: 1 }),
            number(`${x.name} outer`, x.outer, (v) => edit(x.name, { outer: v }), { min: 0, max: limit ? Math.floor(limit) : undefined, step: 1 })),
          `${x.name} inner and outer angle (mrad)`,
          x.name,
        )),
        tileOk && row("Cell", `${(cellX * tiles[0]).toFixed(1)} × ${(cellY * tiles[1]).toFixed(1)} Å`),
        tileOk && row("Grid", `${tiledX} × ${tiledY}`),
        positions && row("Positions", `${positions[0]} × ${positions[1]} = ${positions[0] * positions[1]}`),
        polarBytes !== null && row("Polar data", size(polarBytes), `${bins} bins`),
        memory !== null && row("Memory", `≈ ${size(memory)}`),
        row("Potential", `${s.potential.sampling} Å · ${s.potential.slice_thickness} Å slices`),
        active && row("Progress", `${Math.round((job.progress ?? 0) * 100)}%${job.message ? ` · ${job.message}` : ""}`)),
      !energyOk && h("p", { className: "model-warning small" }, `Energy: ${ENERGY.min}–${ENERGY.max} kV`),
      energyOk && !alphaOk && h("p", { className: "model-warning small" }, `Convergence: up to ${limit.toFixed(0)} mrad`),
      !tileOk && h("p", { className: "model-warning small" }, `Tiling: 1–${STEM.maxTile}`),
      !regionOk && h("p", { className: "model-warning small" }, "Scan: 0 ≤ from < to ≤ 1"),
      regionOk && !samplingOk && h("p", { className: "model-warning small" }, `Sampling: ${STEM.minSampling}–10 Å`),
      !stepOk && h("p", { className: "model-warning small" }, "Radial bin: 0.1–10 mrad"),
      detectors.filter((x) => x.on && !x.ok).map((x) => h("p", { key: x.name, className: "model-warning small" }, `${x.name}: inner below outer`)),
      detectors.filter((x) => x.beyond).map((x) => h("p", { key: x.name, className: "model-warning small" }, `${x.name}: outer up to ${Math.floor(limit)} mrad at this sampling`)),
      !chosen.length && h("p", { className: "model-warning small" }, "Choose a detector"),
      tooLarge && h("p", { className: "model-warning small" }, `Limit: ${STEM.maxGpts} pixels per side`),
      tooMuchPolar && h("p", { className: "model-warning small" }, "Limit: 2 GB of polar data"),
      small && h("p", { className: "model-warning small" }, "Small cell: the probe can wrap into itself"),
      engine && !engine.installed && h("p", { className: "model-warning small" }, "Install abTEM first"),
      error && h("p", { className: "model-warning small" }, error),
      active
        ? h("button", { className: "secondary", disabled: !job.id, onClick: () => host.cancelJob(job.id) }, "Cancel")
        : h("button", { className: "primary", disabled: !valid || !engine?.installed, onClick: run }, "Scan"));
  }

  /** One more annulus of a STEM run, integrated from its saved polar data. */
  function AddDetector({ d }) {
    const s = d.plugin.settings;
    const top = Math.floor(s.max_angle);
    const [name, setName] = React.useState("ADF");
    const [inner, setInner] = React.useState(String(Math.min(top - 1, Math.round(s.semiangle))));
    const [outer, setOuter] = React.useState(String(top));
    const [busy, setBusy] = React.useState(false);
    const [error, setError] = React.useState(null);
    const lo = Number(inner);
    const hi = Number(outer);
    const valid = name.trim() !== "" && inner.trim() !== "" && outer.trim() !== "" && lo >= 0 && lo < hi && hi <= s.max_angle;
    const add = async () => {
      setBusy(true);
      setError(null);
      try {
        const { file, dataset_id } = await host.request(`${API}/stems/${d.id}/detectors`, {
          method: "POST", body: JSON.stringify({ name: name.trim(), inner: lo, outer: hi }),
        });
        host.showFile(file, dataset_id);
      } catch (e) {
        setError(e.message);
      } finally {
        setBusy(false);
      }
    };
    const number = (label, value, set) => h("input", {
      className: "cell-input", type: "number", min: 0, max: top, step: 1, value, disabled: busy, "aria-label": label,
      onChange: (e) => set(e.target.value),
    });
    return h("div", { className: "engine-form model-form" },
      h("dl", { className: "props" },
        row("Name", h("input", {
          className: "cell-input", list: "abtem-stem-detectors", value: name, disabled: busy, "aria-label": "Detector name",
          maxLength: 24, onChange: (e) => setName(e.target.value),
        })),
        row("Inner (mrad)", number("Inner angle", inner, setInner)),
        row("Outer (mrad)", number("Outer angle", outer, setOuter), `The run holds up to ${top} mrad`)),
      h("datalist", { id: "abtem-stem-detectors" }, ["BF", "ABF", "ADF", "MAADF", "HAADF"].map((n) => h("option", { key: n, value: n }))),
      !valid && h("p", { className: "model-warning small" }, `0 ≤ inner < outer ≤ ${top} mrad`),
      error && h("p", { className: "model-warning small" }, error),
      h("button", { className: "primary", disabled: busy || !valid, onClick: add }, busy ? "Adding…" : "Add detector"));
  }

  /** A STEM image interpolated, blurred by the source size and given shot noise. */
  function PostProcess({ d }) {
    const sampling = d.axes?.find((a) => !a.navigate)?.scale;
    const [source, setSource] = React.useState("");
    const [interpolate, setInterpolate] = React.useState("");
    const [dose, setDose] = React.useState("");
    const [seed, setSeed] = React.useState("");
    const [busy, setBusy] = React.useState(false);
    const [error, setError] = React.useState(null);
    const given = (v) => v.trim() !== "" && Number(v) > 0;
    const valid = (given(source) || given(interpolate) || given(dose))
      && [source, interpolate, dose].every((v) => v.trim() === "" || Number(v) >= 0)
      && (seed.trim() === "" || (Number.isInteger(Number(seed)) && Number(seed) >= 0));
    const apply = async () => {
      setBusy(true);
      setError(null);
      try {
        const { file, dataset_id } = await host.request(`${API}/stem-images/${d.id}/post`, {
          method: "POST",
          body: JSON.stringify({
            interpolate: given(interpolate) ? Number(interpolate) : null,
            source_size: given(source) ? Number(source) : 0,
            dose: given(dose) ? Number(dose) : null,
            seed: seed.trim() === "" ? null : Number(seed),
          }),
        });
        host.showFile(file, dataset_id);
      } catch (e) {
        setError(e.message);
      } finally {
        setBusy(false);
      }
    };
    const number = (label, value, set, extra = {}) => h("input", {
      className: "cell-input", type: "number", min: 0, value, disabled: busy, "aria-label": label,
      onChange: (e) => set(e.target.value), ...extra,
    });
    return h("div", { className: "engine-form model-form" },
      h("dl", { className: "props" },
        row("Interpolate (Å)", number("Interpolated sampling", interpolate, setInterpolate, { step: 0.01, placeholder: sampling ? num(sampling, 3) : "Off" }), "A finer sampling, before the blur"),
        row("Source size (Å)", number("Source size", source, setSource, { step: 0.05, placeholder: "Off" }), "Gaussian σ of the blur"),
        row("Dose (e⁻/Å²)", number("Dose", dose, setDose, { step: "any", placeholder: "Off" }), "Poisson noise: the image then counts electrons per pixel"),
        dose.trim() !== "" && row("Seed", number("Noise seed", seed, setSeed, { step: 1, placeholder: "Random" }))),
      error && h("p", { className: "model-warning small" }, error),
      h("button", { className: "primary", disabled: busy || !valid, onClick: apply }, busy ? "Processing…" : "Apply"));
  }

  function StemInfo({ d }) {
    const s = d.plugin.settings;
    const [lx, ly] = s.scan_extent;
    return h("div", { className: "run-info" },
      h("dl", { className: "props" },
        row("Energy", `${s.energy_kv} kV`),
        row("Convergence", `${s.semiangle} mrad`),
        row("Defocus", `${s.defocus} Å`),
        row("Cs", `${s.cs_um} µm`),
        row("Tiling", s.tile.join(" × ")),
        row("Scan", `${num(s.start[0], 3)}–${num(s.end[0], 3)} × ${num(s.start[1], 3)}–${num(s.end[1], 3)} of the cell`),
        row("Sampling", `${s.scan_sampling.map((x) => num(x, 3)).join(" × ")} Å`),
        row("Positions", `${s.gpts.join(" × ")} · ${num(lx, 2)} × ${num(ly, 2)} Å`),
        row("Polar data", `${s.bins} bins of ${s.step} mrad`),
        row("Detectors", s.detectors.map((x) => `${x.name} ${x.inner}–${x.outer}`).join(", ")),
        row("Potential", `${s.potential.sampling} Å · ${s.potential.slice_thickness} Å slices`),
        row("abTEM", s.abtem),
        row("Folder", s.run)));
  }

  host.addProcess({
    id: "stem", title: "STEM", tip: "A probe scanned over this potential, with annular detectors", section: "Simulation",
    applies: (d) => d.plugin?.id === "abtem" && d.plugin.label === "Potential",
    render: (d) => h(StemSection, { key: d.id, d }),
  });

  host.addProcess({
    id: "stem-detector", title: "Add detector", tip: "Another annulus, integrated from this run's saved scattering", section: "Simulation",
    applies: (d) => d.plugin?.id === "abtem" && d.plugin.item === "stem",
    render: (d) => h(AddDetector, { key: d.id, d }),
  });

  host.addProcess({
    id: "stem-post", title: "Source size and noise", tip: "Interpolate, blur by the source size and add shot noise to this STEM image", section: "Simulation",
    applies: (d) => d.plugin?.id === "abtem" && d.plugin.settings?.result === "stem-image",
    render: (d) => h(PostProcess, { key: d.id, d }),
  });

  host.addItemView({
    item: "stem",
    label: "STEM",
    render: (d) => h(StemInfo, { key: d.id, d }),
  });
}
