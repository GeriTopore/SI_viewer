"""Background jobs: per-pixel fitted maps (start), progress and cancel."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..jobs import Job
from ..processing.element_maps import line_label
from ..processing.filters import filter_image
from ..processing.fit_maps import composition_maps, fit_pixels
from ..processing.fitting import FitOptions, FitSource, fit_model, source_spectrum
from ..processing.xray import element_table
from ..session import DatasetKind
from .datasets import _dataset

router = APIRouter(tags=["jobs"])


class QuantSpec(BaseModel):
    """Composition maps from the fitted maps (Cliff-Lorimer)."""

    composition: bool = False
    kfactors: dict[str, float] = {}  # by line, e.g. "Fe_Ka"
    lines: dict[str, str] = {}  # element → line (default: its analysis line)
    exclude: list[str] = []
    # Pixels with fewer fitted element counts than this get no composition.
    min_counts: float = Field(5.0, ge=0)


class FitMapsBody(BaseModel):
    options: FitOptions = FitOptions()
    quant: QuantSpec = QuantSpec()


@router.post("/api/datasets/{dataset_id}/fit-maps")
def start_fit_maps(dataset_id: str, body: FitMapsBody, request: Request) -> dict:
    """Fit every pixel of a spectrum image (as a job). When it's done its
    result holds the file, with a fitted net-count map per element and, if
    asked, at% composition maps."""
    session = request.app.state.session
    si = _dataset(request, dataset_id)
    if si.kind is not DatasetKind.SPECTRUM_IMAGE or si.summary()["eds"] is None:
        raise HTTPException(400, f"{si.title} is not an EDS spectrum image")
    if not si.summary()["eds"]["elements"]:
        raise HTTPException(422, "Add elements in the periodic table before fitting")

    def run(job: Job) -> dict:
        job.report(0.02, "Fitting the summed spectrum")
        spec, _ = source_spectrum(si, FitSource(type="all"))
        fitted = fit_model(
            spec, body.options, lambda f, step: job.report(0.02 + 0.28 * f, f"Summed spectrum: {step}")
        )
        job.report(0.3, "Fitting each pixel")
        intensities = fit_pixels(si.signal, fitted, body.options, lambda f: job.report(0.3 + 0.6 * f))
        job.report(0.92, "Making maps")
        # The spectrum image's spatial filter applies to its maps; blurring the
        # (linear) fitted intensities equals fitting the blurred pixels.
        spatial = si.filter_spec
        if spatial.has_spatial:
            intensities = {line: filter_image(data, spatial) for line, data in intensities.items()}

        # One map per element: the chosen line (default: its analysis line).
        per_element = {}
        for el, analysis in fitted.chosen.items():
            line = body.quant.lines.get(el, analysis)
            if line in intensities:
                per_element[el] = (line, intensities[line])
        maps = [("fitted", el, f"{line_label(line)} (fit)", data) for el, (line, data) in per_element.items()]
        if body.quant.composition:
            included = {el: v for el, v in per_element.items() if el not in body.quant.exclude}
            if included:
                weights = {e["symbol"]: e["atomic_weight"] for e in element_table()}
                at = composition_maps(
                    {el: data for el, (_, data) in included.items()},
                    {el: body.quant.kfactors.get(line, 1.0) for el, (line, _) in included.items()},
                    weights,
                    body.quant.min_counts,
                )
                maps += [("composition", el, f"{el} at%", data) for el, data in at.items()]
        job.report(0.97)  # last chance to cancel before the results replace old ones
        session.set_fit_maps(si, maps)
        entry = session.file_of(si)
        return {"file": entry.summary(session), "skipped": [e for e in fitted.elements if e not in fitted.chosen]}

    job = request.app.state.jobs.submit("fit-maps", f"Fit every pixel of {si.title}", run)
    return job.summary()


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict:
    try:
        return request.app.state.jobs.jobs[job_id].summary()
    except KeyError:
        raise HTTPException(404, f"Unknown job: {job_id}")


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict:
    try:
        return request.app.state.jobs.cancel(job_id).summary()
    except KeyError:
        raise HTTPException(404, f"Unknown job: {job_id}")
