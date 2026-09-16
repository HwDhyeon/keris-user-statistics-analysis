"""분석 엔진 공통 기반."""

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from app.models.enums import AnalysisMethod, ChartKind
from app.schemas.analysis import AnalysisParams, ChartRecommendation, PreprocessingSpec


@dataclass
class AnalysisOutcome:
    """엔진 실행 결과."""

    result: dict[str, Any]
    metrics: dict[str, Any]
    # 차트 생성을 위한 중간 데이터 (DataFrame 등). 저장되지 않는다.
    artifacts: dict[str, Any] = field(default_factory=dict)
    recommendations: list[ChartRecommendation] = field(default_factory=list)


class Engine(Protocol):
    method: AnalysisMethod

    def run(self, frame: pd.DataFrame, params: AnalysisParams, spec: PreprocessingSpec) -> AnalysisOutcome: ...


def rec(
    kind: ChartKind,
    title: str,
    reason: str,
    priority: int = 5,
    *,
    data_key: str | None = None,
    encoding: dict[str, str] | None = None,
    options: dict[str, Any] | None = None,
) -> ChartRecommendation:
    """차트 추천 한 건. data_key는 chart_data에서 클라이언트가 꺼내 쓸 데이터 키다."""
    return ChartRecommendation(
        kind=kind,
        title=title,
        reason=reason,
        priority=priority,
        data_key=data_key,
        encoding=encoding or {},
        options=options or {},
    )


def jsonable(value: Any) -> Any:
    """numpy/pandas 타입을 JSON 직렬화 가능한 값으로 변환한다."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(value, (np.ndarray,)):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, pd.Series):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [{str(k): jsonable(v) for k, v in row.items()} for row in value.to_dict("records")]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, str):
        return value
    return str(value)


def num(value: Any) -> float | None:
    """NaN/Inf를 None으로 바꾼 float."""
    try:
        result = float(value)
    except TypeError, ValueError:
        return None
    return None if (math.isnan(result) or math.isinf(result)) else result


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series | None,
    test_size: float,
    random_state: int,
    stratify: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series | None, pd.Series | None]:
    """test_size가 0이면 전체 데이터로 적합하고 홀드아웃은 학습셋과 동일하게 둔다."""
    if test_size <= 0:
        return X, X, y, y, weights, weights

    from sklearn.model_selection import train_test_split

    strat = y if (stratify and y.value_counts().min() >= 2) else None
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=strat)
    w_train = weights.iloc[train_idx] if weights is not None else None
    w_test = weights.iloc[test_idx] if weights is not None else None
    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
        w_train,
        w_test,
    )


def sample_points(frame: pd.DataFrame, limit: int, random_state: int = 42) -> pd.DataFrame:
    """차트용으로 행 수를 제한한다."""
    if len(frame) <= limit:
        return frame
    return frame.sample(n=limit, random_state=random_state)
