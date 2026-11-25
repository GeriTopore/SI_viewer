// The test plugin's frontend: one process and one bottom tab.
export function activate(host) {
  const { React } = host;
  const h = React.createElement;

  function Run({ d }) {
    const [job, setJob] = React.useState(null);
    const [error, setError] = React.useState(null);
    const start = async () => {
      setError(null);
      try {
        const done = await host.runJob(`/api/plugins/test-plugin/datasets/${d.id}/run`, { steps: 40, delay: 0.1 }, setJob);
        if (done.status === "failed") setError(done.error);
      } catch (e) {
        setError(e.message);
      }
    };
    const active = job && (job.status === "queued" || job.status === "running");
    return h("div", null,
      h("button", { onClick: active ? () => host.cancelJob(job.id) : start }, active ? "Cancel" : "Run"),
      job && h("span", { className: "muted small" }, ` ${job.status} ${Math.round(job.progress * 100)}%`),
      error && h("p", { className: "error small" }, error));
  }

  host.addProcess({
    id: "run", title: "Test plugin", tip: "Run the test worker", section: "Plugin",
    applies: (d) => d.kind === "image",
    render: (d) => h(Run, { d }),
  });
  host.addBottomTab({
    id: "info", label: "Plugin",
    applies: (d) => d.kind === "image",
    render: (d) => h("p", { className: "muted small" }, d.title),
  });
}
