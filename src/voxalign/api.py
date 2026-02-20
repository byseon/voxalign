"""HTTP API for voxalign."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response

from voxalign import __version__
from voxalign.config import load_config
from voxalign.core import run_alignment
from voxalign.models import AlignRequest, AlignResponse, HealthResponse


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(
        title="voxalign",
        version=__version__,
        description="Multilingual forced aligner service API.",
    )
    config = load_config()

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__, env=config.env)

    @app.post("/v1/align", response_model=AlignResponse, tags=["alignment"])
    def align(request: AlignRequest, response: Response) -> AlignResponse:
        try:
            alignment = run_alignment(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if alignment.metadata.license_warning is not None:
            response.headers["X-VoxAlign-License-Warning"] = alignment.metadata.license_warning
        return alignment

    return app


app = create_app()
