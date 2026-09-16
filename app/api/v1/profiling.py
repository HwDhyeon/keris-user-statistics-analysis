"""자동 프로파일링·상관관계·탐색 리포트 API."""

from typing import Annotated, Any

from fastapi import APIRouter, Query

from app.api.deps import DatasetDep
from app.models.enums import CorrelationMethod, OutlierMethod
from app.schemas.analysis import ChartDataOptions
from app.schemas.dataset import (
    CorrelationResult,
    DatasetDetail,
    DatasetProfile,
    OutlierReport,
    ProfileReport,
)
from app.services import profiling, recommendations, storage

router = APIRouter(prefix="/datasets/{dataset_id}", tags=["기초 통계·탐색"])


@router.get(
    "/profile",
    response_model=DatasetProfile,
    summary="기초 통계량 및 결측·이상치 자동 탐지",
    description="평균·중앙값·사분위수 등 기술통계와 결측치·이상치를 자동으로 산출한다.",
)
def get_profile(
    dataset: DatasetDep,
    outlier_method: Annotated[OutlierMethod, Query()] = OutlierMethod.IQR,
    outlier_threshold: Annotated[float, Query(gt=0)] = 1.5,
) -> DatasetProfile:
    frame = storage.load_frame(dataset.id)
    return profiling.build_profile(
        dataset.id, frame, outlier_method=outlier_method, outlier_threshold=outlier_threshold
    )


@router.get("/outliers", response_model=list[OutlierReport], summary="이상치 탐지")
def get_outliers(
    dataset: DatasetDep,
    method: Annotated[OutlierMethod, Query()] = OutlierMethod.IQR,
    threshold: Annotated[float, Query(gt=0)] = 1.5,
    columns: Annotated[list[str] | None, Query()] = None,
) -> list[OutlierReport]:
    frame = storage.load_frame(dataset.id)
    return profiling.detect_outliers(frame, method=method, threshold=threshold, columns=columns)


@router.get(
    "/correlation",
    response_model=CorrelationResult,
    summary="변수 간 상관관계",
    description="상관계수 행렬과 쌍별 유의확률, 히트맵용 셀 데이터를 제공한다.",
)
def get_correlation(
    dataset: DatasetDep,
    method: Annotated[CorrelationMethod, Query()] = CorrelationMethod.PEARSON,
    columns: Annotated[list[str] | None, Query()] = None,
    min_abs: Annotated[float, Query(ge=0, le=1, description="이 값 미만의 |r| 쌍은 제외")] = 0.0,
) -> CorrelationResult:
    frame = storage.load_frame(dataset.id)
    return _correlation(dataset.id, frame, method, columns, min_abs)


@router.get(
    "/report",
    response_model=ProfileReport,
    summary="탐색 리포트",
    description=(
        "프로파일·상관관계와 함께, 어떤 차트를 그리면 좋은지(chart_recommendations)와 "
        "그리는 데 필요한 데이터(chart_data)를 한 번에 제공한다. 렌더링은 클라이언트가 수행한다."
    ),
)
def get_report(
    dataset: DatasetDep,
    correlation_method: Annotated[CorrelationMethod, Query()] = CorrelationMethod.PEARSON,
    histogram_bins: Annotated[int, Query(ge=5, le=100)] = 30,
    max_points: Annotated[int, Query(ge=100, le=100000)] = 5000,
) -> ProfileReport:
    frame = storage.load_frame(dataset.id)
    profile = profiling.build_profile(dataset.id, frame)
    options = ChartDataOptions(histogram_bins=histogram_bins, max_points=max_points)

    correlation = _correlation(dataset.id, frame, correlation_method, None, 0.0)
    has_correlation = len(correlation.columns) >= 2

    chart_data: dict[str, Any] = recommendations.dataset_chart_data(frame, options)
    if has_correlation:
        chart_data["correlation_matrix"] = {
            "columns": correlation.columns,
            "cells": correlation.heatmap_cells,
        }
        chart_data["scatter_pairs"] = recommendations.scatter_pairs(
            frame, correlation.pairs, options.max_points
        )

    return ProfileReport(
        dataset=DatasetDetail.model_validate(dataset, from_attributes=True),
        profile=profile,
        correlation=correlation if has_correlation else None,
        chart_recommendations=recommendations.dataset_recommendations(frame, has_correlation),
        chart_data=chart_data,
    )


def _correlation(
    dataset_id: str,
    frame: Any,
    method: CorrelationMethod,
    columns: list[str] | None,
    min_abs: float,
) -> CorrelationResult:
    cols, matrix, pairs = profiling.correlation_matrix(
        frame, method=method, columns=columns, min_abs=min_abs
    )
    cells = [
        {"x": cols[i], "y": cols[j], "r": matrix[i][j]}
        for i in range(len(cols))
        for j in range(len(cols))
        if matrix and matrix[i][j] is not None
    ]
    return CorrelationResult(
        dataset_id=dataset_id,
        method=method,
        columns=cols,
        matrix=matrix,
        pairs=pairs,
        heatmap_cells=cells,
    )
