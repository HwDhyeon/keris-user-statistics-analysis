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
    """분석 엔진 실행 결과.

    Attributes:
        result (dict[str, Any]): 분석 결과로 저장될 데이터(계수, 지표 상세 등).
        metrics (dict[str, Any]): 결정계수·RMSE 등 모델 성능 지표 요약.
        artifacts (dict[str, Any]): 차트 생성을 위한 중간 데이터(DataFrame 등). DB에는 저장되지 않는다.
        recommendations (list[ChartRecommendation]): 엔진이 직접 제시하는 차트 추천 목록.
    """

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
    """차트 추천 한 건을 만든다. data_key는 chart_data에서 클라이언트가 꺼내 쓸 데이터 키다.

    Args:
        kind (ChartKind): 추천 차트 종류.
        title (str): 차트 제목.
        reason (str): 이 차트를 추천하는 이유.
        priority (int, optional): 추천 우선순위(작을수록 우선). Defaults to 5.
        data_key (str | None, optional): chart_data 내 이 차트가 사용할 데이터 키. Defaults to None.
        encoding (dict[str, str] | None, optional): 축·색 등에 대응하는 데이터 필드명. Defaults to None.
        options (dict[str, Any] | None, optional): 참조선·정렬 등 렌더링 힌트. Defaults to None.

    Returns:
        ChartRecommendation: 구성된 차트 추천 객체.
    """
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
    """numpy/pandas 타입을 JSON 직렬화 가능한 값으로 변환한다.

    Args:
        value (Any): 변환할 값(numpy 스칼라, ndarray, pandas Series/DataFrame, dict, list 등).

    Returns:
        Any: bool, int, float, str, list, dict 등 JSON으로 직렬화 가능한 값. NaN/Inf는 None으로 변환된다.
    """
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
    """NaN/Inf를 None으로 바꾼 float.

    Args:
        value (Any): float로 변환할 값.

    Returns:
        float | None: 유한한 float 값. 변환 불가·NaN·Inf인 경우 None.
    """
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
    """test_size가 0이면 전체 데이터로 적합하고 홀드아웃은 학습셋과 동일하게 둔다.

    Args:
        X (pd.DataFrame): 독립변수 설계행렬.
        y (pd.Series): 종속변수.
        weights (pd.Series | None): 가중치. 없으면 None.
        test_size (float): 검증용으로 분리할 비율(0이면 분리하지 않음).
        random_state (int): 분리 재현을 위한 난수 시드.
        stratify (bool, optional): True이고 각 클래스가 2건 이상이면 y 기준 층화추출한다. Defaults to False.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series | None, pd.Series | None]:
            학습용 X, 검증용 X, 학습용 y, 검증용 y, 학습용 가중치, 검증용 가중치.
    """
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
    """차트용으로 행 수를 제한한다.

    Args:
        frame (pd.DataFrame): 표본을 추출할 데이터프레임.
        limit (int): 남길 최대 행 수.
        random_state (int, optional): 표본추출 재현을 위한 난수 시드. Defaults to 42.

    Returns:
        pd.DataFrame: limit 이하로 표본추출된 데이터프레임. 원본이 limit 이하면 그대로 반환.
    """
    if len(frame) <= limit:
        return frame
    return frame.sample(n=limit, random_state=random_state)
