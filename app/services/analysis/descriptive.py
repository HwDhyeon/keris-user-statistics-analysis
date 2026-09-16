"""기술통계 및 상관분석 엔진."""

import pandas as pd

from app.models.enums import AnalysisMethod, ChartKind, CorrelationMethod, OutlierMethod
from app.schemas.analysis import AnalysisParams, PreprocessingSpec
from app.services import profiling
from app.services.analysis.base import AnalysisOutcome, jsonable, rec
from app.services.preprocessing import apply_category_orders, apply_filters, handle_missing


def _prepare(frame: pd.DataFrame, params: AnalysisParams, spec: PreprocessingSpec) -> tuple[pd.DataFrame, list[dict]]:
    """군집/회귀와 달리 인코딩 없이 필터·결측 처리만 적용한다.

    Args:
        frame (pd.DataFrame): 원본 데이터프레임.
        params (AnalysisParams): 분석 대상 변수(features)를 담은 파라미터.
        spec (PreprocessingSpec): 필터·결측 처리·범주 순서 등 전처리 스펙.

    Returns:
        tuple[pd.DataFrame, list[dict]]: 전처리가 적용된 데이터프레임과 적용된 처리 단계 기록.
    """
    columns = params.features or list(frame.columns)
    work = frame[[c for c in columns if c in frame.columns]].copy()
    work, steps = apply_filters(work, spec.filters)
    work, s = handle_missing(work, spec)
    steps += s
    work, s = apply_category_orders(work, spec.category_orders)
    steps += s
    return work, steps


class DescriptiveEngine:
    """기초 통계량 + 결측/이상치 요약을 산출하는 기술통계 분석 엔진."""

    method = AnalysisMethod.DESCRIPTIVE

    def run(self, frame: pd.DataFrame, params: AnalysisParams, spec: PreprocessingSpec) -> AnalysisOutcome:
        work, steps = _prepare(frame, params, spec)
        profile = profiling.build_profile(
            "", work, outlier_method=OutlierMethod.IQR, outlier_threshold=spec.outlier_threshold
        )

        result = {
            "method": "기술통계 분석",
            "columns": [str(c) for c in work.columns],
            "n_observations": len(work),
            "numeric_summary": [s.model_dump() for s in profile.numeric],
            "categorical_summary": [s.model_dump() for s in profile.categorical],
            "missing": [m.model_dump() for m in profile.missing],
            "outliers": [o.model_dump() for o in profile.outliers],
            "warnings": profile.warnings,
            "preprocessing_steps": steps,
        }
        metrics = {
            "n_rows": len(work),
            "n_columns": int(work.shape[1]),
            "n_numeric": len(profile.numeric),
            "n_categorical": len(profile.categorical),
            "missing_ratio": profile.missing_ratio,
            "n_duplicated_rows": profile.n_duplicated_rows,
        }
        artifacts = {"frame": work, "profile": profile}
        recommendations = [
            rec(
                ChartKind.HISTOGRAM, "수치형 변수 분포", "분포 형태와 치우침 확인", 1,
                data_key="histograms",
                encoding={"x": "bin_center", "y": "count"},
                options={"grouped_by_column": True, "bin_bounds": ["bin_start", "bin_end"]},
            ),
            rec(
                ChartKind.BOXPLOT, "상자그림", "이상치와 사분위 범위 확인", 2,
                data_key="boxplots",
                encoding={"x": "column", "y": "median"},
                options={"whisker_fields": ["lower_fence", "q1", "median", "q3", "upper_fence"]},
            ),
            rec(
                ChartKind.BAR, "범주형 변수 빈도", "범주 구성비 확인", 3,
                data_key="category_counts",
                encoding={"x": "count", "y": "value"},
                options={"grouped_by_column": True, "sort": "-x"},
            ),
        ]
        if profile.missing:
            recommendations.append(
                rec(
                    ChartKind.MISSING_MATRIX, "변수별 결측률", "결측 발생 구조 확인", 4,
                    data_key="missing_ratios",
                    encoding={"x": "ratio", "y": "column"},
                    options={"sort": "-x", "format": "percent"},
                )
            )
        return AnalysisOutcome(jsonable(result), jsonable(metrics), artifacts, recommendations)


class CorrelationEngine:
    """변수 간 상관관계를 사전 파악하는 상관분석 엔진."""

    method = AnalysisMethod.CORRELATION

    def run(self, frame: pd.DataFrame, params: AnalysisParams, spec: PreprocessingSpec) -> AnalysisOutcome:
        work, steps = _prepare(frame, params, spec)
        method = CorrelationMethod(params.correlation_method)
        columns, matrix, pairs = profiling.correlation_matrix(work, method=method)

        strong = [p for p in pairs if abs(p.coefficient) >= 0.7]
        result = {
            "method": f"상관분석 ({method.value})",
            "correlation_method": method.value,
            "columns": columns,
            "matrix": matrix,
            "pairs": [p.model_dump() for p in pairs],
            "strong_pairs": [p.model_dump() for p in strong],
            "n_observations": len(work),
            "interpretation": (
                f"|r| ≥ 0.7인 강한 상관 쌍이 {len(strong)}개 있습니다."
                if strong
                else "강한 상관(|r| ≥ 0.7)을 보이는 변수 쌍은 없습니다."
            ),
            "preprocessing_steps": steps,
        }
        metrics = {
            "n_variables": len(columns),
            "n_pairs": len(pairs),
            "n_strong_pairs": len(strong),
            "max_abs_coefficient": abs(pairs[0].coefficient) if pairs else None,
        }
        artifacts = {"frame": work, "columns": columns, "matrix": matrix, "pairs": pairs}
        recommendations = [
            rec(
                ChartKind.HEATMAP, "상관계수 히트맵", "변수 간 관계를 한눈에 파악", 1,
                data_key="correlation_matrix",
                encoding={"x": "x", "y": "y", "color": "r", "label": "r"},
                options={"domain": [-1, 1], "diverging": True, "cells": "cells"},
            ),
            rec(
                ChartKind.SCATTER, "상위 상관 쌍 산점도", "관계의 형태(선형/비선형) 확인", 2,
                data_key="scatter_pairs",
                encoding={"x": "x", "y": "y"},
                options={"group_by": "pair", "points": "points", "fit_line": True},
            ),
        ]
        return AnalysisOutcome(jsonable(result), jsonable(metrics), artifacts, recommendations)
