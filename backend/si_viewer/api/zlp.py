"""Zero-loss peak alignment of EELS low-loss spectrum images (M10d), as a job."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..jobs import Job
from ..processing.zlp import ZlpOptions, check_signal
from .thickness import _low_loss

router = APIRouter(tags=["zlp"])


class ZlpBody(BaseModel):
    options: ZlpOptions = ZlpOptions()


@router.post("/api/datasets/{dataset_id}/zlp-alignment")
def post_zlp_alignment(dataset_id: str, body: ZlpBody, request: Request) -> dict:
    """Align the zero-loss peak of the low loss that ``dataset_id`` is, or is
    linked to (as a job): every spectrum is shifted so that its peak is at 0
    (in the energy axis' units), and the linked core losses get the same
    shifts. Like a rebin, this replaces the loaded data of all of them (the
    file on disk is not changed). The shifts are saved as the low loss's "ZLP
    shift" dataset (origin ``zlp-shift``), replacing an earlier one. The job's
    result is ``{file, partner_files, dataset_id, aligned, shift}``: the low
    loss's file, the other files of the datasets aligned, the shift map's id,
    the ids aligned (low loss first) and how far the axes were moved (mean
    shift, eV)."""
    session = request.app.state.session
    ll = _low_loss(request, dataset_id)
    options = body.options
    try:
        check_signal(ll.signal)  # fail now, not in the job
    except ValueError as e:
        raise HTTPException(422, str(e))

    def run(job: Job) -> dict:
        job.report(0.01, "Loading the spectrum images" if ll.lazy else "Starting")
        datasets, originals, result = session.compute_zlp_alignment(
            ll, options, lambda f, m=None: job.report(0.02 + 0.93 * f, m)
        )
        job.report(0.96, "Remaking what was made from them")  # last chance to cancel
        shift_map = session.apply_zlp_alignment(datasets, originals, result, options)
        files = dict.fromkeys(d.file_id for d in datasets if d.file_id != ll.file_id)
        return {
            "file": session.file_of(ll).summary(session),
            "partner_files": [session.files[f].summary(session) for f in files],
            "dataset_id": shift_map.id,
            "aligned": [d.id for d in datasets],
            "shift": result.mean_shift,
        }

    return request.app.state.jobs.submit("zlp-alignment", f"Zero-loss alignment of {ll.title}", run).summary()
