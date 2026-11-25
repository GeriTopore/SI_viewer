"""Background jobs (e.g. fitting every pixel) with progress and cancellation.

Jobs run one at a time on a worker thread. A job function receives its
``Job`` and calls ``job.report(fraction, message)`` as it goes; that also
raises ``Cancelled`` once the job has been asked to stop.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence


class Cancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    kind: str
    title: str
    status: str = "queued"  # queued | running | done | failed | cancelled
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    result: dict | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    # A job run as a separate process: killed when cancelled.
    _process: subprocess.Popen | None = field(default=None, repr=False)

    def report(self, progress: float, message: str | None = None) -> None:
        if self._cancel.is_set():
            raise Cancelled()
        self.progress = max(0.0, min(1.0, progress))
        if message is not None:
            self.message = message

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")

    def summary(self, with_result: bool = True) -> dict:
        out = {
            "id": self.id, "kind": self.kind, "title": self.title, "status": self.status,
            "progress": self.progress, "message": self.message, "error": self.error,
            "created": self.created, "started": self.started, "finished": self.finished,
        }
        if with_result:
            out["result"] = self.result
        return out


def _kill(process: subprocess.Popen) -> None:
    """Stop ``process`` and whatever it started."""
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="job")
        # Separate processes queue among themselves, beside the jobs above.
        self._process_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="process-job")

    def submit_process(
        self,
        kind: str,
        title: str,
        command: Sequence[str],
        on_result: Callable[[dict], dict] | None = None,
        cwd: str | Path | None = None,
    ) -> Job:
        """Run ``command`` as a job in its own process.

        It writes JSON lines to stdout: ``{"progress": 0.4, "message": "…"}``
        as it goes and ``{"result": {…}}`` at the end; other lines are ignored.
        ``on_result`` turns that result into the job's. Cancelling kills it.
        """
        job = Job(id=uuid.uuid4().hex[:10], kind=kind, title=title)
        self.jobs[job.id] = job

        def run() -> None:
            if job._cancel.is_set():
                job.status, job.finished = "cancelled", time.time()
                return
            job.status, job.started = "running", time.time()
            try:
                with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as stderr:
                    process = subprocess.Popen(
                        list(command), cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr,
                        text=True, encoding="utf-8", errors="replace", start_new_session=os.name == "posix",
                    )
                    job._process = process
                    if job._cancel.is_set():  # asked while it was starting
                        _kill(process)
                    result: dict = {}
                    for line in process.stdout:
                        try:
                            msg = json.loads(line)
                        except ValueError:
                            continue
                        if not isinstance(msg, dict):
                            continue
                        if isinstance(msg.get("progress"), (int, float)):
                            job.progress = max(0.0, min(1.0, float(msg["progress"])))
                        if isinstance(msg.get("message"), str):
                            job.message = msg["message"]
                        if isinstance(msg.get("result"), dict):
                            result = msg["result"]
                    code = process.wait()
                    if job._cancel.is_set():
                        job.status = "cancelled"
                    elif code != 0:
                        stderr.seek(0)
                        lines = [x for x in stderr.read().splitlines() if x.strip()]
                        job.status = "failed"
                        job.error = lines[-1] if lines else f"Process exited with code {code}"
                    else:
                        job.result = on_result(result) if on_result else result
                        job.progress, job.status = 1.0, "done"
            except Exception as e:  # reported to the client
                job.status, job.error = "failed", str(e) or type(e).__name__
                traceback.print_exc()
            finally:
                job._process = None
                job.finished = time.time()

        self._process_executor.submit(run)
        return job

    def submit(self, kind: str, title: str, fn: Callable[[Job], dict]) -> Job:
        job = Job(id=uuid.uuid4().hex[:10], kind=kind, title=title)
        self.jobs[job.id] = job

        def run() -> None:
            if job._cancel.is_set():
                job.status, job.finished = "cancelled", time.time()
                return
            job.status, job.started = "running", time.time()
            try:
                job.result = fn(job)
                job.progress, job.status = 1.0, "done"
            except Cancelled:
                job.status = "cancelled"
            except Exception as e:  # reported to the client
                job.status, job.error = "failed", str(e) or type(e).__name__
                traceback.print_exc()
            finally:
                job.finished = time.time()

        self._executor.submit(run)
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.jobs[job_id]
        job._cancel.set()
        if job.status == "queued":
            job.status, job.finished = "cancelled", time.time()
        if (process := job._process) is not None:
            _kill(process)
        return job

    def listing(self, recent: int = 3) -> list[Job]:
        """Every queued or running job (oldest first), then the ``recent``
        latest to finish (done, failed or cancelled; newest first)."""
        jobs = list(self.jobs.values())
        active = sorted((j for j in jobs if j.active), key=lambda j: j.created)
        ended = sorted((j for j in jobs if not j.active), key=lambda j: j.finished or j.created, reverse=True)
        return active + ended[: max(0, recent)]
