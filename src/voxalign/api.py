"""HTTP API for voxalign."""

from __future__ import annotations

from fastapi import FastAPI

from voxalign import __version__
from voxalign.config import load_config
from voxalign.models import HealthResponse


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

    return app


app = create_app()
