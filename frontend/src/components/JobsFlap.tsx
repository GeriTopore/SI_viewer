import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { api, type JobSummary } from "../api/client";
import { useDatasets } from "../state/datasets";

/** How often the list is fetched (ms): while a job runs or the panel is
 * open, and otherwise (to notice jobs started anywhere). */
const POLL_BUSY = 700;
const POLL_IDLE = 4000;

const running = (status: JobSummary["status"] | undefined) => status === "queued" || status === "running";
const active = (j: JobSummary) => running(j.status);

/** Background jobs (fitting every pixel, alignments, background removal…):
 * a folder tab on the window's bottom edge at the left ("Jobs", with the
 * number running), opening a drawer above it with every running job's progress (and
 * Cancel) and the three latest to finish. When a job finishes, its results
 * are brought into the dataset list, whichever panel started it. */
export function JobsFlap() {
  const [open, setOpen] = useState(false);
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const seen = useRef(new Map<string, JobSummary["status"]>());
  const [now, setNow] = useState(() => Date.now() / 1000);

  useEffect(() => {
    let timer = 0;
    let stopped = false;
    const poll = async () => {
      let busy = false;
      try {
        const list = await api.listJobs(3);
        if (stopped) return;
        // A job seen running that has now finished: bring in its results.
        const finished = list.some((j) => !active(j) && running(seen.current.get(j.id)));
        seen.current = new Map(list.map((j) => [j.id, j.status]));
        if (finished) void useDatasets.getState().syncFiles();
        setJobs(list);
        setNow(Date.now() / 1000);
        busy = list.some(active);
      } catch {
        /* backend busy or restarted: try again */
      }
      if (!stopped) timer = window.setTimeout(poll, busy || open ? POLL_BUSY : POLL_IDLE);
    };
    void poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [open]);

  // The bottom edge: level with the datasets pane's.
  const [bottom, setBottom] = useState(28);
  useLayoutEffect(() => {
    const pane = document.querySelector(".sidebar");
    const place = () => {
      const r = pane?.getBoundingClientRect();
      if (r) setBottom(Math.round(window.innerHeight - r.bottom));
    };
    place();
    const watch = new ResizeObserver(place);
    if (pane) watch.observe(pane);
    window.addEventListener("resize", place);
    return () => {
      watch.disconnect();
      window.removeEventListener("resize", place);
    };
  }, []);

  const count = jobs.filter(active).length;
  return (
    <div className={`jobs-flap${open ? " open" : ""}`} style={{ bottom }}>
      <button
        className="jobs-flap-handle"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        title={open ? "Hide the background jobs" : "Show the background jobs"}
      >
        <span className="focus-flap-arrow" aria-hidden>{open ? "▾" : "▴"}</span>
        <span>Jobs</span>
        {count > 0 && <span className="jobs-count" aria-label={`${count} running`}>{count}</span>}
      </button>
      {open && (
        <div className="jobs-panel" role="region" aria-label="Background jobs">
          {jobs.length === 0 ? (
            <p className="muted small">No jobs yet.</p>
          ) : (
            <ul>
              {jobs.map((j) => (
                <JobRow key={j.id} job={j} now={now} onChange={(next) => setJobs((js) => js.map((x) => (x.id === next.id ? next : x)))} />
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

function JobRow({ job, now, onChange }: { job: JobSummary; now: number; onChange: (j: JobSummary) => void }) {
  const pct = Math.round(job.progress * 100);
  const ended = !active(job);
  const took = job.started && job.finished ? job.finished - job.started : null;
  const ago = job.finished ? now - job.finished : null;
  return (
    <li className={`jobs-row ${job.status}`}>
      <div className="jobs-row-title" title={job.title}>{job.title}</div>
      {ended ? (
        <div className="jobs-row-status small">
          {job.status === "done" ? "Done" : job.status === "failed" ? "Failed" : "Cancelled"}
          {took !== null && job.status === "done" && ` in ${duration(took)}`}
          {ago !== null && <span className="muted"> · {duration(ago)} ago</span>}
          {job.status === "failed" && job.error && <div className="field-error">{job.error}</div>}
        </div>
      ) : (
        <div className="q-job">
          <div className="q-progress" role="progressbar" aria-valuenow={pct}>
            <div style={{ width: `${pct}%` }} />
          </div>
          <div className="q-job-row small">
            <span>
              {job.status === "queued" ? "Waiting" : job.message || "Running"} {job.status === "running" && `${pct}%`}
            </span>
            <button className="link-button" onClick={() => api.cancelJob(job.id).then(onChange).catch(() => {})}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </li>
  );
}

/** e.g. "45 s", "3 min 20 s", "2 h 5 min". */
function duration(s: number): string {
  s = Math.max(0, Math.round(s));
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min${s % 60 ? ` ${s % 60} s` : ""}`;
  return `${Math.floor(s / 3600)} h${Math.floor((s % 3600) / 60) ? ` ${Math.floor((s % 3600) / 60)} min` : ""}`;
}
