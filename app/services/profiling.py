"""기초 통계량, 결측치·이상치 자동 탐지, 상관관계 분석."""

import math
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from scipy import stats as sps

from app.models.enums import ColumnRole, CorrelationMethod, OutlierMethod
from app.schemas.dataset import (
    CategoricalSummary,
    CorrelationPair,
    DatasetProfile,
    MissingReport,
    NumericSummary,
    OutlierReport,
)
from app.services.ingest import _clean, infer_role


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if ptypes.is_numeric_dtype(frame[c]) and not ptypes.is_bool_dtype(frame[c])]


def describe_numeric(series: pd.Series, name: str) -> NumericSummary:
    """평균·중앙값·사분위수를 포함한 기초 통계량."""
    values = pd.to_numeric(series, errors="coerce")
    clean = values.dropna()
    if clean.empty:
        return NumericSummary(column=name, count=0, missing=int(values.isna().sum()))

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0

    return NumericSummary(
        column=name,
        count=int(clean.size),
        missing=int(values.isna().sum()),
        mean=_f(mean),
        std=_f(std),
        variance=_f(std**2),
        min=_f(clean.min()),
        q1=_f(q1),
        median=_f(clean.median()),
        q3=_f(q3),
        max=_f(clean.max()),
        iqr=_f(q3 - q1),
        skewness=_f(clean.skew()) if len(clean) > 2 else None,
        kurtosis=_f(clean.kurtosis()) if len(clean) > 3 else None,
        cv=_f(std / mean) if mean not in (0, None) else None,
        sum=_f(clean.sum()),
    )


def describe_categorical(series: pd.Series, name: str, top_n: int = 10) -> CategoricalSummary:
    clean = series.dropna()
    counts = clean.value_counts()
    total = int(clean.size)
    entropy = None
    if total and not counts.empty:
        probs = counts.to_numpy(dtype=float) / total
        entropy = float(-(probs * np.log2(probs)).sum())

    return CategoricalSummary(
        column=name,
        count=total,
        missing=int(series.isna().sum()),
        n_unique=int(counts.size),
        mode=_clean(counts.index[0]) if not counts.empty else None,
        top_values=[
            {"value": _clean(k), "count": int(v), "ratio": round(int(v) / total, 6) if total else 0.0}
            for k, v in counts.head(top_n).items()
        ],
        entropy=_f(entropy),
    )


def detect_missing(frame: pd.DataFrame) -> list[MissingReport]:
    total = len(frame)
    reports = [
        MissingReport(
            column=str(col),
            n_missing=int(frame[col].isna().sum()),
            ratio=round(float(frame[col].isna().mean()), 6) if total else 0.0,
        )
        for col in frame.columns
    ]
    return sorted(reports, key=lambda r: r.n_missing, reverse=True)


def detect_outliers(
    frame: pd.DataFrame,
    *,
    method: OutlierMethod = OutlierMethod.IQR,
    threshold: float = 1.5,
    columns: list[str] | None = None,
    sample_limit: int = 20,
) -> list[OutlierReport]:
    """수치형 변수의 이상치를 자동 탐지한다."""
    targets = columns or numeric_columns(frame)
    reports: list[OutlierReport] = []

    for col in targets:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        clean = values.dropna()
        if clean.size < 4:
            continue

        mask, lower, upper = outlier_mask(clean, method, threshold)
        idx = clean.index[mask]
        reports.append(
            OutlierReport(
                column=str(col),
                method=method,
                n_outliers=int(mask.sum()),
                ratio=round(float(mask.sum()) / float(clean.size), 6),
                lower_bound=_f(lower),
                upper_bound=_f(upper),
                sample_indices=[int(i) for i in idx[:sample_limit]],
                sample_values=[_f(v) or 0.0 for v in clean.loc[idx[:sample_limit]].tolist()],
            )
        )
    return sorted(reports, key=lambda r: r.n_outliers, reverse=True)


def outlier_mask(
    clean: pd.Series, method: OutlierMethod, threshold: float
) -> tuple[np.ndarray, float | None, float | None]:
    """지정한 방식으로 이상치 여부 마스크와 판정 경계값을 계산한다."""
    if method is OutlierMethod.IQR:
        q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
        return ((clean < lower) | (clean > upper)).to_numpy(), lower, upper

    if method is OutlierMethod.ZSCORE:
        limit = threshold if threshold > 1.5 else 3.0
        std = float(clean.std(ddof=1))
        if std == 0:
            return np.zeros(clean.size, dtype=bool), None, None
        z = np.abs((clean - float(clean.mean())) / std)
        mean = float(clean.mean())
        return (z > limit).to_numpy(), mean - limit * std, mean + limit * std

    if method is OutlierMethod.MODIFIED_ZSCORE:
        limit = threshold if threshold > 1.5 else 3.5
        median = float(clean.median())
        mad = float(np.median(np.abs(clean - median)))
        if mad == 0:
            return np.zeros(clean.size, dtype=bool), None, None
        z = 0.6745 * np.abs(clean - median) / mad
        span = limit * mad / 0.6745
        return (z > limit).to_numpy(), median - span, median + span

    # ISOLATION_FOREST
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(contamination="auto", random_state=42)
    pred = model.fit_predict(clean.to_numpy().reshape(-1, 1))
    return (pred == -1), None, None


def build_profile(
    dataset_id: str,
    frame: pd.DataFrame,
    *,
    outlier_method: OutlierMethod = OutlierMethod.IQR,
    outlier_threshold: float = 1.5,
) -> DatasetProfile:
    """반입 데이터의 종합 프로파일."""
    total_cells = int(frame.size)
    total_missing = int(frame.isna().sum().sum())

    numeric_summaries: list[NumericSummary] = []
    categorical_summaries: list[CategoricalSummary] = []
    warnings: list[str] = []

    for col in frame.columns:
        role = infer_role(frame[col])
        if role is ColumnRole.NUMERIC:
            numeric_summaries.append(describe_numeric(frame[col], str(col)))
        elif role in {ColumnRole.CATEGORICAL, ColumnRole.BOOLEAN}:
            categorical_summaries.append(describe_categorical(frame[col], str(col)))
        elif role is ColumnRole.CONSTANT:
            warnings.append(f"'{col}' 변수는 값이 하나뿐이라 분석에 기여하지 않습니다.")
        elif role is ColumnRole.IDENTIFIER:
            warnings.append(f"'{col}' 변수는 식별자로 보입니다. 분석 변수에서 제외를 권장합니다.")

    missing = detect_missing(frame)
    for report in missing:
        if report.ratio >= 0.5:
            warnings.append(f"'{report.column}' 변수의 결측률이 {report.ratio:.1%}로 높습니다.")

    outliers = detect_outliers(frame, method=outlier_method, threshold=outlier_threshold)
    for report in outliers:
        if report.ratio >= 0.1:
            warnings.append(f"'{report.column}' 변수에 이상치가 {report.ratio:.1%} 존재합니다.")

    n_duplicated = int(frame.duplicated().sum())
    if n_duplicated:
        warnings.append(f"중복 행이 {n_duplicated}건 있습니다.")

    return DatasetProfile(
        dataset_id=dataset_id,
        n_rows=len(frame),
        n_cols=int(frame.shape[1]),
        n_duplicated_rows=n_duplicated,
        total_missing_cells=total_missing,
        missing_ratio=round(total_missing / total_cells, 6) if total_cells else 0.0,
        numeric=numeric_summaries,
        categorical=categorical_summaries,
        missing=[m for m in missing if m.n_missing > 0],
        outliers=outliers,
        warnings=warnings,
    )


def correlation_matrix(
    frame: pd.DataFrame,
    *,
    method: CorrelationMethod = CorrelationMethod.PEARSON,
    columns: list[str] | None = None,
    min_abs: float = 0.0,
) -> tuple[list[str], list[list[float | None]], list[CorrelationPair]]:
    """변수 간 상관계수 행렬과 유의확률을 계산한다."""
    cols = [c for c in (columns or numeric_columns(frame)) if c in frame.columns]
    if len(cols) < 2:
        return cols, [], []

    subset = frame[cols].apply(pd.to_numeric, errors="coerce")
    matrix_df = subset.corr(method=method.value)
    matrix = [[_f(v) for v in row] for row in matrix_df.to_numpy()]

    pairs: list[CorrelationPair] = []
    for i, x in enumerate(cols):
        for y in cols[i + 1 :]:
            paired = subset[[x, y]].dropna()
            if len(paired) < 3:
                continue
            coef, p_value = _corr_with_p(paired[x], paired[y], method)
            if coef is None or abs(coef) < min_abs:
                continue
            pairs.append(
                CorrelationPair(
                    x=x, y=y, coefficient=coef, p_value=p_value, strength=_strength(coef)
                )
            )

    pairs.sort(key=lambda p: abs(p.coefficient), reverse=True)
    return cols, matrix, pairs


def _corr_with_p(
    x: pd.Series, y: pd.Series, method: CorrelationMethod
) -> tuple[float | None, float | None]:
    if x.nunique() < 2 or y.nunique() < 2:
        return None, None
    try:
        if method is CorrelationMethod.SPEARMAN:
            res = sps.spearmanr(x, y)
        elif method is CorrelationMethod.KENDALL:
            res = sps.kendalltau(x, y)
        else:
            res = sps.pearsonr(x, y)
    except (ValueError, FloatingPointError):
        return None, None
    return _f(res[0]), _f(res[1])


def _strength(coef: float) -> str:
    a = abs(coef)
    if a >= 0.7:
        return "강함"
    if a >= 0.4:
        return "보통"
    if a >= 0.2:
        return "약함"
    return "거의 없음"


def _f(value: Any) -> float | None:
    """NaN/Inf를 None으로 바꾼 float."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result
