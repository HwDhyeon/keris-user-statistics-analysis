"""고급 통계·예측 분석 API."""

from typing import Annotated, Any

from fastapi import APIRouter, Query, status
from sqlmodel import func, select

from app.api.deps import AnalysisDep, SessionDep
from app.models.enums import AnalysisMethod, AnalysisStatus
from app.models.tables import Analysis
from app.schemas.analysis import (
    AnalysisCreate,
    AnalysisDetail,
    AnalysisRead,
    ChartDataOptions,
    ChartRecommendation,
)
from app.schemas.common import Message, Page
from app.services import runner
from app.services.analysis.registry import METHOD_INFO

router = APIRouter(tags=["고급 통계·예측 분석"])


@router.get("/methods", summary="지원 분석 기법 목록")
def list_methods() -> list[dict[str, Any]]:
    return [{"method": method.value, **info} for method, info in METHOD_INFO.items()]


@router.post(
    "/analyses",
    response_model=AnalysisDetail,
    status_code=status.HTTP_201_CREATED,
    summary="분석 수행",
    description=(
        "회귀/의사결정나무/군집 등 분석을 수행한다. "
        "변수 순서 유지(category_orders), 다중 변수 선택(features), 가중치(weight_column)를 지원하며 "
        "결정계수·RMSE 등 모델 성능 지표를 함께 산출한다. "
        "차트는 서버에서 그리지 않고, 추천(chart_recommendations)과 "
        "그리는 데 필요한 데이터(chart_data)를 반환한다."
    ),
)
def create_analysis(payload: AnalysisCreate, session: SessionDep) -> AnalysisDetail:
    return AnalysisDetail.model_validate(
        runner.create_analysis(session, payload), from_attributes=True
    )


@router.get("/analyses", response_model=Page[AnalysisRead], summary="분석 목록")
def list_analyses(
    session: SessionDep,
    dataset_id: Annotated[str | None, Query()] = None,
    method: Annotated[AnalysisMethod | None, Query()] = None,
    analysis_status: Annotated[AnalysisStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[AnalysisRead]:
    statement = select(Analysis)
    count_statement = select(func.count()).select_from(Analysis)
    for condition in (
        (Analysis.dataset_id == dataset_id) if dataset_id else None,
        (Analysis.method == method) if method else None,
        (Analysis.status == analysis_status) if analysis_status else None,
    ):
        if condition is not None:
            statement = statement.where(condition)
            count_statement = count_statement.where(condition)

    total = session.exec(count_statement).one()
    rows = session.exec(
        statement.order_by(Analysis.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return Page(
        items=[AnalysisRead.model_validate(r, from_attributes=True) for r in rows],
        total=int(total),
        limit=limit,
        offset=offset,
    )


@router.get("/analyses/{analysis_id}", response_model=AnalysisDetail, summary="분석 결과 상세")
def get_analysis(analysis: AnalysisDep) -> AnalysisDetail:
    return AnalysisDetail.model_validate(analysis, from_attributes=True)


@router.delete("/analyses/{analysis_id}", response_model=Message, summary="분석 삭제")
def delete_analysis(analysis: AnalysisDep, session: SessionDep) -> Message:
    session.delete(analysis)
    session.commit()
    return Message(message=f"분석 '{analysis.name}'을(를) 삭제했습니다.")


@router.get(
    "/analyses/{analysis_id}/chart-recommendations",
    response_model=list[ChartRecommendation],
    summary="차트 추천",
    description=(
        "분석 기법에 최적화된 차트 종류를 우선순위와 함께 추천한다. "
        "각 항목의 data_key는 chart_data에서 꺼내 쓸 데이터를, encoding은 축·색에 대응하는 "
        "필드명을 가리킨다. 실제 렌더링은 클라이언트가 수행한다."
    ),
)
def get_chart_recommendations(analysis: AnalysisDep) -> list[ChartRecommendation]:
    return [ChartRecommendation.model_validate(r) for r in analysis.chart_recommendations]


@router.get(
    "/analyses/{analysis_id}/chart-data",
    summary="차트용 데이터",
    description="추천 차트를 그리는 데 필요한 집계 데이터. key를 지정하면 해당 항목만 반환한다.",
)
def get_chart_data(
    analysis: AnalysisDep,
    key: Annotated[list[str] | None, Query(description="반환할 데이터 키")] = None,
) -> dict[str, Any]:
    data = analysis.chart_data
    if key:
        return {k: v for k, v in data.items() if k in set(key)}
    return data


@router.post(
    "/analyses/{analysis_id}/chart-data",
    response_model=AnalysisDetail,
    summary="차트용 데이터 재생성",
    description="표본 점 개수·히스토그램 구간 수 등을 바꿔 차트용 데이터를 다시 만든다.",
)
def rebuild_chart_data(
    analysis: AnalysisDep, payload: ChartDataOptions, session: SessionDep
) -> AnalysisDetail:
    return AnalysisDetail.model_validate(
        runner.rebuild_chart_data(session, analysis, payload), from_attributes=True
    )
