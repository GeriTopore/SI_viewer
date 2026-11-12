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
    # False: the aligned spectra become new datasets under their sources
    # (linked as they are); the sources are left as they are.
    in_place: bool = True


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
    for d in (ll, *session.zlp_followers(ll)):
        if d.insitu is not None:
            raise HTTPException(
                422, f"{d.title} is a frame of an in-situ series and can't be aligned: the next frame would undo it"
            )
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
        shift_map = session.apply_zlp_alignment(datasets, originals, result, options, body.in_place)
        aligned = datasets if body.in_place else [session.get_dataset(shift_map.parent_id), *session.linked_of(session.get_dataset(shift_map.parent_id))]
        files = dict.fromkeys(d.file_id for d in aligned if d.file_id != ll.file_id)
        return {
            "file": session.file_of(ll).summary(session),
            "partner_files": [session.files[f].summary(session) for f in files],
            "dataset_id": shift_map.id,
            "aligned": [d.id for d in aligned],
            "shift": result.mean_shift,
        }

    return request.app.state.jobs.submit("zlp-alignment", f"ZLP alignment of {ll.title}", run).summary()


class ShiftBody(BaseModel):
    # Energy added to the axis (its units): positive moves the spectra up.
    shift: float
    # False: a shifted copy, a new dataset under it.
    in_place: bool = True


@router.post("/api/datasets/{dataset_id}/energy-shift")
def post_energy_shift(dataset_id: str, body: ShiftBody, request: Request) -> dict:
    """Shift the energy axis of an EELS spectrum image (or LineScan) by a
    given amount: ZLP alignment by hand, for a core loss without a low loss.
    Windows, maps and background removals made from it move with it. Not
    ``in_place``: a shifted copy is made under it. Returns the dataset (the
    copy if one was made) and its file."""
    from .eds import _with_file

    session = request.app.state.session
    try:
        ds = session.get_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {dataset_id}")
    try:
        made = session.shift_energy(ds, body.shift, body.in_place)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _with_file(request, made)
