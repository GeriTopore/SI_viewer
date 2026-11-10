"""Rendering 1D plots (spectra, line profiles) as PNG, SVG or PDF."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from ..processing.plot_export import PlotRequest, render_plot

router = APIRouter(tags=["plot"])


@router.post("/api/plot/render")
def render(body: PlotRequest) -> Response:
    """The plot as the image bytes (PNG, SVG or PDF, as the options say)."""
    try:
        data, media_type = render_plot(body)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return Response(data, media_type=media_type, headers={"Cache-Control": "no-store"})
