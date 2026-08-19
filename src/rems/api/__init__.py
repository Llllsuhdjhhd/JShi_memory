"""REMS HTTP API package.

FastAPI 应用入口位于 :mod:`rems.api.app`；REST 路由见 :mod:`rems.api.routes`。
运行：``uvicorn rems.api.app:app --reload``。
"""

from .app import app, create_app, get_pipeline

__all__ = ["app", "create_app", "get_pipeline"]
