"""이용자 통계분석 서비스 백엔드 API."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from scalar_fastapi import get_scalar_api_reference

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.database import init_db
from app.core.exceptions import register_exception_handlers

STATIC_DIR = Path(__file__).parent / "static"

DESCRIPTION = """
교육데이터플랫폼의 이용자 통계분석 서비스를 위한 REST API입니다.

* **데이터 반입** — CSV/TSV/Excel/JSON/Parquet 업로드, 인코딩·구분자 자동 판별, 변수 역할 자동 추론
* **기초 통계·탐색** — 평균·중앙값·사분위수 등 기술통계, 결측치·이상치 자동 탐지, 상관관계 히트맵
* **고급 통계·예측** — 선형/로지스틱 회귀, 의사결정나무, 군집분석, 주성분분석
  (변수 순서 유지·다중 변수 선택·가중치 부여 지원, 결정계수·RMSE 등 성능 지표 산출)
* **시각화** — 기법에 최적화된 차트 자동 추천과 차트용 집계 데이터 제공 (렌더링은 클라이언트 담당)
* **패키지** — 분석 조건·전처리 로직·리포트를 ZIP으로 보존하고 재현·공유
"""


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    description=DESCRIPTION,
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_tags=[
        {"name": "데이터셋 반입", "description": "로컬 파일을 시스템으로 반입한다."},
        {"name": "기초 통계·탐색", "description": "기술통계, 결측·이상치, 상관관계 리포트."},
        {"name": "고급 통계·예측 분석", "description": "회귀·의사결정나무·군집 등 분석과 차트."},
        {"name": "분석 패키지 (보존·재현·공유)", "description": "분석 결과 패키징과 재현."},
    ],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/docs", include_in_schema=False)
def scalar_docs_html() -> Any:
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=f"{app.title} - API Reference",
        scalar_js_url="/static/scalar/scalar-api-reference.js",
        scalar_favicon_url="/static/common/favicon.ico",
        with_default_fonts=False,
        telemetry=False,
    )


@app.get("/swagger", include_in_schema=False)
def swagger_ui_html() -> Any:
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/swagger-ui.css",
        swagger_favicon_url="/static/common/favicon.ico",
    )


@app.get("/redoc", include_in_schema=False)
def redoc_html() -> Any:
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - ReDoc",
        redoc_js_url="/static/redoc/redoc.standalone.js",
        redoc_favicon_url="/static/common/favicon.ico",
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/health", tags=["시스템"], summary="헬스 체크")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": app.version,
    }
