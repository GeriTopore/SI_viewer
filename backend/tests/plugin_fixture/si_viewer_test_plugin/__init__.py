"""A minimal plugin for tests: a process, a bottom tab and a worker job."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from si_viewer.plugins import PluginContext, PluginManifest

HERE = Path(__file__).parent
manifest = PluginManifest(id="test-plugin", name="Test plugin", version="0.0.1", frontend=HERE / "frontend")


class RunBody(BaseModel):
    steps: int = 5
    delay: float = 0.05
    fail: bool = False


def register(ctx: PluginContext) -> None:
    router = APIRouter()

    @router.post("/datasets/{dataset_id}/run")
    def run(dataset_id: str, body: RunBody, request: Request) -> dict:
        session = ctx.session
        try:
            parent = session.get_dataset(dataset_id)
        except KeyError:
            raise HTTPException(404, f"Unknown dataset id: {dataset_id}")
        out = Path(tempfile.mkdtemp(prefix="si-viewer-plugin-")) / "result.npy"

        def on_result(result: dict) -> dict:
            import hyperspy.api as hs

            signal = hs.signals.Signal2D(np.load(result["path"]))
            ds = ctx.add_dataset(parent, signal, f"{parent.title} (test plugin)", body.model_dump())
            return {"dataset_id": ds.id, "file": session.file_of(ds).summary(session)}

        command = [sys.executable, "-I", str(HERE / "worker.py"), str(out), str(body.steps), str(body.delay), str(int(body.fail))]
        return ctx.submit_process("run", f"Test plugin on {parent.title}", command, on_result).summary()

    ctx.add_router(router)
