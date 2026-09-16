"""v1 API 라우터 집합."""

from fastapi import APIRouter

from app.api.v1 import analyses, datasets, packages, profiling

api_router = APIRouter()
api_router.include_router(datasets.router)
api_router.include_router(profiling.router)
api_router.include_router(analyses.router)
api_router.include_router(packages.router)
api_router.include_router(packages.shared_router)
