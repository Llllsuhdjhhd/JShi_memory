from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

# FastAPI 应用：生命周期内挂载 REMSPipeline，路由见 routes.py（ingest/事件/角色/演化/健康检查）。

from fastapi import FastAPI

from ..config import REMSConfig
from ..pipeline import REMSPipeline

_pipeline: REMSPipeline | None = None


def get_pipeline() -> REMSPipeline:
    assert _pipeline is not None, "Pipeline not initialised — app not started"
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _pipeline
    config = REMSConfig()
    _pipeline = REMSPipeline.from_config(config)
    yield
    _pipeline = None


def create_app() -> FastAPI:
    from .routes import router

    app = FastAPI(
        title="REMS API",
        version="0.1.0",
        description="Recursive Evolutionary Memory System",
        lifespan=lifespan,
    )
    app.include_router(router, prefix="/api")
    return app


app = create_app()
