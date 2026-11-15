"""Background jobs (e.g. fitting every pixel) with progress and cancellation.

Jobs run one at a time on a worker thread. A job function receives its
``Job`` and calls ``job.report(fraction, message)`` as it goes; that also
raises ``Cancelled`` once the job has been asked to stop.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable


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


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="job")

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
        return job

    def listing(self, recent: int = 3) -> list[Job]:
        """Every queued or running job (oldest first), then the ``recent``
        latest to finish (done, failed or cancelled; newest first)."""
        jobs = list(self.jobs.values())
        active = sorted((j for j in jobs if j.active), key=lambda j: j.created)
        ended = sorted((j for j in jobs if not j.active), key=lambda j: j.finished or j.created, reverse=True)
        return active + ended[: max(0, recent)]
