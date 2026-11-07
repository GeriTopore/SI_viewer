"""Bundled example datasets (all offline), for trying features and tutorials."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..examples import list_examples

router = APIRouter(prefix="/api/examples", tags=["examples"])


@router.get("")
def get_examples() -> list[dict]:
    return list_examples()


@router.post("/{example_id}")
def open_example(example_id: str, request: Request) -> dict:
    """Open an example dataset as a file entry (returns its file summary)."""
    session = request.app.state.session
    try:
        entry = session.open_example(example_id)
    except KeyError:
        raise HTTPException(404, f"Unknown example: {example_id}")
    return entry.summary(session)
