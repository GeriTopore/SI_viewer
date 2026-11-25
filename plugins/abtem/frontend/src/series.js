// Thickness series (a plane wave kept every n slices) and defocus series (one
// CTF at a range of defocus values) of HRTEM.

const MAX_BYTES = 2 * 1024 ** 3;
const MAX_DEFOCI = 100;
// A defocus series opens on the CTF's defocus ± this, in steps of this many Å.
const DEFOCUS_RANGE = 100;
const DEFOCUS_STEP = 20;

/** The defocus values from ``from`` to ``to`` in ``step`` (Å): how many (null: not a series). */
export const defocusCount = (from, to, step) => {
  if (![from, to, step].every(Number.isFinite) || step <= 0) return null;
  const n = Math.floor((to - from) / step + 1e-9) + 1;
  return n >= 2 ? n : null;
};

/** Registers the series processes and the series item's view. ``helpers`` are
 * the plugin's own: API, CtfChoice, useProjectCtfs, wavelength, ANTIALIAS,
 * HRTEM (energy limits and default), num, potentialLabel. */
export function registerSeries(host, helpers) {
  const { React } = host;
  const h = React.createElement;
  const { API, CtfChoice, useProjectCtfs, wavelength, ANTIALIAS, HRTEM, num, potentialLabel } = helpers;

  const row = (label, value, title) => [
    h("dt", { key: `${label}-t`, title }, label),
    h("dd", { key: `${label}-d` }, value),
  ];

  /** A number to type, in a row of the form. */
  const numberInput = (label, value, set, disabled, extra = {}) => h("input", {
    className: "cell-input", type: "number", value, disabled, "aria-label": label,
    onChange: (e) => set(e.target.value), ...extra,
  });

  /** The plane wave through a Potential, kept every n slices, and the image a
   * CTF of the project forms of each. */
  function ThicknessSection({ d }) {
    const s = d.plugin.settings;
    const [engine, setEngine] = React.useState(null);
    const ctfs = useProjectCtfs(d.file_id, d.id);
    const [ctfId, setCtfId] = React.useState(undefined);
    const [coherence, setCoherence] = React.useState("envelope");
    const [energy, setEnergy] = React.useState(String(HRTEM.energy));
    const [step, setStep] = React.useState("1");
    const [job, setJob] = React.useState(null);
    const [error, setError] = React.useState(null);
    const mounted = React.useRef(true);
    const active = job?.status === "queued" || job?.status === "running";

    React.useEffect(() => {
      if (ctfId === undefined && ctfs) setCtfId(ctfs[0]?.id ?? null);
    }, [ctfs, ctfId]);

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
      host.request(`${API}/potentials/${d.id}/thickness-series`)
        .then((r) => {
          if (!stale && r.job) host.followJob(r.job, (j) => { if (mounted.current) setJob(j); }).then(finish).catch((e) => setError(e.message));
        })
        .catch((e) => { if (!stale) setError(e.message); });
      return () => {
        stale = true;
        mounted.current = false;
      };
    }, [d.id, finish]);

    const potential = s.potential;
    const chosen = ctfs?.find((c) => c.id === ctfId);
    const kv = chosen ? chosen.settings.energy_kv : Number(energy);
    const energyOk = kv >= HRTEM.minEnergy && kv <= HRTEM.maxEnergy;
    const limit = energyOk ? (wavelength(kv) / (2 * potential.sampling)) * ANTIALIAS * 1000 : null;
    const apertureOk = !chosen || chosen.settings.semiangle === null || (limit !== null && chosen.settings.semiangle <= limit);
    const stepN = Number(step);
    const stepOk = Number.isInteger(stepN) && stepN >= 1 && stepN <= s.slices;
    const kept = stepOk ? Math.floor(s.slices / stepN) : null;
    const configs = Number(potential.phonon_configs) || 0;
    const [nx, ny] = s.gpts;
    // Complex values: two floats per pixel of every plane (and configuration).
    const tooMuchMemory = stepOk && 8 * nx * ny * (kept + 1) * (configs || 1) > MAX_BYTES;
    const valid = energyOk && apertureOk && ctfId !== undefined && stepOk && !tooMuchMemory;
    const run = async () => {
      setError(null);
      try {
        finish(await host.runJob(`${API}/potentials/${d.id}/thickness-series`, {
          energy_kv: kv, ctf: ctfId ?? null, coherence, step: stepN,
        }, (j) => { if (mounted.current) setJob(j); }));
      } catch (e) {
        if (mounted.current) setError(e.message);
      }
    };
    return h("div", { className: "engine-form model-form" },
      h("dl", { className: "props" },
        CtfChoice({ ctfs, ctfId, setCtfId, coherence, setCoherence, disabled: active, allowNone: true }),
        chosen
          ? row("Energy", `${kv} kV`, "The CTF's")
          : row("Energy (kV)", numberInput("Series energy", energy, setEnergy, active, { min: HRTEM.minEnergy, max: HRTEM.maxEnergy })),
        row("Step (slices)", numberInput("Slices between planes", step, setStep, active, { min: 1, max: s.slices, step: 1 }),
          "The exit wave is kept after every this many slices"),
        stepOk && row("Thicknesses", kept + 1, "Planes kept, the incident wave included"),
        stepOk && row("Maximum", `${num(kept * stepN * s.slice_thickness, 2)} Å`, "Thickness of the last plane"),
        active && row("Progress", `${Math.round((job.progress ?? 0) * 100)}%${job.message ? ` · ${job.message}` : ""}`)),
      !stepOk && h("p", { className: "model-warning small" }, `Step: 1 to ${s.slices} slices`),
      tooMuchMemory && h("p", { className: "model-warning small" }, "Limit: 2 GB of exit waves"),
      !energyOk && h("p", { className: "model-warning small" }, `Energy: ${HRTEM.minEnergy}–${HRTEM.maxEnergy} kV`),
      energyOk && !apertureOk && h("p", { className: "model-warning small" }, `The CTF's aperture: up to ${limit.toFixed(0)} mrad at this sampling`),
      engine && !engine.installed && h("p", { className: "model-warning small" }, "Install abTEM first"),
      error && h("p", { className: "model-warning small" }, error),
      active
        ? h("button", { className: "secondary", disabled: !job.id, onClick: () => host.cancelJob(job.id) }, "Cancel")
        : h("button", { className: "primary", disabled: !valid || !engine?.installed, onClick: run }, "Run series"));
  }

  /** One CTF of the project at a range of defocus values, applied to an HRTEM
  * run's exit wave: a stack of images through defocus. */
  function DefocusSection({ d }) {
    const s = d.plugin.settings;
    const ctfs = useProjectCtfs(d.file_id, d.id);
    const usable = (ctfs ?? []).filter((c) => Math.abs(c.settings.energy_kv - s.energy_kv) < 1e-6);
    const [ctfId, setCtfId] = React.useState(undefined);
    const [coherence, setCoherence] = React.useState("envelope");
    const [from, setFrom] = React.useState("");
    const [to, setTo] = React.useState("");
    const [step, setStep] = React.useState(String(DEFOCUS_STEP));
    const [busy, setBusy] = React.useState(false);
    const [error, setError] = React.useState(null);
    React.useEffect(() => {
      if (ctfId === undefined && ctfs) setCtfId(usable[0]?.id ?? null);
    }, [ctfs]); // eslint-disable-line react-hooks/exhaustive-deps
    const chosen = ctfs?.find((c) => c.id === ctfId);
    // The range follows the CTF chosen: its own defocus ± 100 Å.
    React.useEffect(() => {
      if (!chosen) return;
      setFrom(String(+(chosen.settings.defocus - DEFOCUS_RANGE).toFixed(2)));
      setTo(String(+(chosen.settings.defocus + DEFOCUS_RANGE).toFixed(2)));
    }, [chosen?.id]); // eslint-disable-line react-hooks/exhaustive-deps
    const [a, b, c] = [from, to, step].map((x) => (x.trim() === "" ? NaN : Number(x)));
    const count = defocusCount(a, b, c);
    const tooMany = count !== null && count > MAX_DEFOCI;
    const nx = s.gpts[s.gpts.length - 2];
    const ny = s.gpts[s.gpts.length - 1];
    const tooMuchMemory = count !== null && 4 * nx * ny * count > MAX_BYTES;
    const valid = !!chosen && count !== null && !tooMany && !tooMuchMemory;
    const run = async () => {
      setBusy(true);
      setError(null);
      try {
        const { file, dataset_id } = await host.request(`${API}/runs/${d.id}/defocus-series`, {
          method: "POST", body: JSON.stringify({ ctf: ctfId, coherence, start: a, stop: b, step: c }),
        });
        host.showFile(file, dataset_id);
      } catch (e) {
        setError(e.message);
      } finally {
        setBusy(false);
      }
    };
    return h("div", { className: "engine-form model-form" },
      h("dl", { className: "props" },
        CtfChoice({ ctfs, ctfId, setCtfId, coherence, setCoherence, energy: s.energy_kv, disabled: busy, allowNone: false }),
        row("From (Å)", numberInput("Defocus from", from, setFrom, busy, { step: 1 }), "Defocus of the first image"),
        row("To (Å)", numberInput("Defocus to", to, setTo, busy, { step: 1 }), "Defocus of the last image, at most"),
        row("Step (Å)", numberInput("Defocus step", step, setStep, busy, { min: 0, step: 1 })),
        count !== null && row("Images", count)),
      ctfs && !usable.length && h("p", { className: "model-warning small" }, `No CTF at ${s.energy_kv} kV in the project`),
      count === null && chosen && h("p", { className: "model-warning small" }, "At least two defocus values"),
      tooMany && h("p", { className: "model-warning small" }, `Limit: ${MAX_DEFOCI} images`),
      !tooMany && tooMuchMemory && h("p", { className: "model-warning small" }, "Limit: 2 GB of images"),
      error && h("p", { className: "model-warning small" }, error),
      h("button", { className: "primary", disabled: busy || !valid, onClick: run }, busy ? "Running…" : "Run series"));
  }

  /** What a thickness series was made from. */
  function SeriesInfo({ d }) {
    const s = d.plugin.settings;
    const ctf = s.ctf_settings;
    const last = s.thicknesses[s.thicknesses.length - 1];
    return h("div", { className: "run-info" },
      h("dl", { className: "props" },
        row("Energy", `${s.energy_kv} kV`),
        row("Step", `${s.step} ${s.step === 1 ? "slice" : "slices"}`),
        row("Planes", s.planes),
        row("Maximum", `${num(last, 2)} Å`),
        ctf && row("Defocus", `${ctf.defocus} Å`),
        ctf && row("Aperture", ctf.semiangle ? `${ctf.semiangle} mrad` : "None"),
        ctf && ctf.focal_spread > 0 && row("Focal spread", s.coherence === "focal-series" ? "Focal series (7)" : "Envelope"),
        row("Potential", potentialLabel(s.potential)),
        row("Grid", s.gpts.join(" × ")),
        s.configs > 0 && row("Configurations", s.configs),
        row("Exit intensity", num(s.exit_intensity, 4)),
        row("abTEM", s.abtem),
        row("Folder", s.run)));
  }

  host.addProcess({
    id: "thickness-series", title: "Thickness series", section: "Simulation",
    tip: "A plane wave through this potential, kept every few slices, and the images a CTF forms of each",
    applies: (d) => d.plugin?.id === "abtem" && d.plugin.label === "Potential",
    render: (d) => h(ThicknessSection, { key: d.id, d }),
  });

  host.addProcess({
    id: "defocus-series", title: "Defocus series", section: "Simulation",
    tip: "This run's images through a range of defocus values, without repeating multislice",
    applies: (d) => d.plugin?.id === "abtem" && d.plugin.item === "run" && d.plugin.settings?.result === "hrtem",
    render: (d) => h(DefocusSection, { key: d.id, d }),
  });

  host.addItemView({
    item: "series",
    label: "Thickness series",
    render: (d) => h(SeriesInfo, { key: d.id, d }),
  });
}
