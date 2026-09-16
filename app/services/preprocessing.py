"""전처리 파이프라인. 적용 내역을 기록해 재현 가능하게 만든다."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from app.core.exceptions import AnalysisError
from app.models.enums import EncodeStrategy, ImputeStrategy, OutlierMethod, ScaleStrategy
from app.schemas.analysis import FilterCondition, PreprocessingSpec
from app.services.profiling import outlier_mask


@dataclass
class DesignMatrix:
    """모델 적합에 바로 쓸 수 있는 형태로 정리된 데이터."""

    X: pd.DataFrame
    y: pd.Series | None
    weights: pd.Series | None
    frame: pd.DataFrame
    feature_names: list[str]
    # 원-핫 파생열 → 원본 변수 이름
    feature_origin: dict[str, str]
    target_classes: list[Any] | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)


def apply_filters(frame: pd.DataFrame, filters: list[FilterCondition]) -> tuple[pd.DataFrame, list[dict]]:
    steps: list[dict] = []
    for cond in filters:
        if cond.column not in frame.columns:
            msg = f"필터 대상 변수를 찾을 수 없습니다: {cond.column}"
            raise AnalysisError(msg)
        before = len(frame)
        series = frame[cond.column]
        match cond.op:
            case "eq":
                mask = series == cond.value
            case "ne":
                mask = series != cond.value
            case "gt":
                mask = pd.to_numeric(series, errors="coerce") > float(cond.value)
            case "gte":
                mask = pd.to_numeric(series, errors="coerce") >= float(cond.value)
            case "lt":
                mask = pd.to_numeric(series, errors="coerce") < float(cond.value)
            case "lte":
                mask = pd.to_numeric(series, errors="coerce") <= float(cond.value)
            case "in":
                mask = series.isin(list(cond.value or []))
            case "not_in":
                mask = ~series.isin(list(cond.value or []))
            case "between":
                low, high = cond.value
                numeric = pd.to_numeric(series, errors="coerce")
                mask = numeric.between(float(low), float(high))
            case "notna":
                mask = series.notna()
            case "isna":
                mask = series.isna()
            case _:  # pragma: no cover - Literal로 제한됨
                msg = f"지원하지 않는 필터 연산자: {cond.op}"
                raise AnalysisError(msg)
        frame = frame[mask.fillna(False)]
        steps.append(
            {
                "step": "filter",
                "column": cond.column,
                "op": cond.op,
                "value": cond.value,
                "rows_before": before,
                "rows_after": len(frame),
            }
        )
    return frame, steps


def handle_missing(frame: pd.DataFrame, spec: PreprocessingSpec) -> tuple[pd.DataFrame, list[dict]]:
    """열별 대치를 먼저 적용하고, 남은 결측치를 전역 전략으로 처리한다."""
    steps: list[dict] = []

    for imputation in spec.column_imputations:
        col = imputation.column
        if col not in frame.columns:
            continue
        n_before = int(frame[col].isna().sum())
        if n_before == 0:
            continue
        frame = frame.copy()
        frame[col] = _impute_series(frame[col], imputation.strategy, imputation.fill_value)
        steps.append(
            {
                "step": "impute",
                "column": col,
                "strategy": imputation.strategy.value,
                "fill_value": imputation.fill_value,
                "filled": n_before - int(frame[col].isna().sum()),
            }
        )

    handled = {i.column for i in spec.column_imputations}
    remaining = [c for c in frame.columns if c not in handled]

    if spec.missing_strategy is ImputeStrategy.DROP_ROWS:
        before = len(frame)
        frame = frame.dropna(subset=remaining) if remaining else frame.dropna()
        if before != len(frame):
            steps.append({"step": "drop_missing_rows", "rows_before": before, "rows_after": len(frame)})
    elif spec.missing_strategy is not ImputeStrategy.NONE:
        frame = frame.copy()
        for col in remaining:
            if frame[col].isna().any():
                frame[col] = _impute_series(frame[col], spec.missing_strategy, spec.missing_fill_value)
        steps.append({"step": "impute_all", "strategy": spec.missing_strategy.value})

    return frame, steps


def _impute_series(series: pd.Series, strategy: ImputeStrategy, fill_value: Any) -> pd.Series:
    match strategy:
        case ImputeStrategy.NONE:
            return series
        case ImputeStrategy.DROP_ROWS:
            return series
        case ImputeStrategy.MEAN:
            numeric = pd.to_numeric(series, errors="coerce")
            return numeric.fillna(numeric.mean())
        case ImputeStrategy.MEDIAN:
            numeric = pd.to_numeric(series, errors="coerce")
            return numeric.fillna(numeric.median())
        case ImputeStrategy.MODE:
            mode = series.mode(dropna=True)
            return series.fillna(mode.iloc[0]) if not mode.empty else series
        case ImputeStrategy.CONSTANT:
            return series.fillna(fill_value)
        case ImputeStrategy.FORWARD_FILL:
            return series.ffill().bfill()
    return series


def remove_outlier_rows(
    frame: pd.DataFrame, columns: list[str], method: OutlierMethod, threshold: float
) -> tuple[pd.DataFrame, list[dict]]:
    """선택한 방식(IQR/Z-score/Modified Z-score/Isolation Forest)으로 이상치 행을 제거한다."""
    steps: list[dict] = []
    for col in columns:
        if col not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[col], errors="coerce")
        clean = numeric.dropna()
        if clean.size < 4:
            continue
        mask, lower, upper = outlier_mask(clean, method, threshold)
        outlier_idx = clean.index[mask]
        before = len(frame)
        frame = frame.drop(index=outlier_idx)
        steps.append(
            {
                "step": "remove_outliers",
                "column": col,
                "method": method.value,
                "threshold": threshold,
                "lower": lower,
                "upper": upper,
                "rows_before": before,
                "rows_after": len(frame),
            }
        )
    return frame, steps


def apply_category_orders(frame: pd.DataFrame, orders: dict[str, list[Any]]) -> tuple[pd.DataFrame, list[dict]]:
    """지정된 범주 순서를 ordered Categorical로 고정한다."""
    steps: list[dict] = []
    if not orders:
        return frame, steps
    frame = frame.copy()
    for col, order in orders.items():
        if col not in frame.columns:
            continue
        present = set(frame[col].dropna().astype(object).unique())
        unknown = present - set(order)
        if unknown:
            msg = f"'{col}' 변수의 범주 순서 목록에 없는 값이 있습니다: {sorted(map(str, unknown))}"
            raise AnalysisError(msg)
        frame[col] = pd.Categorical(frame[col], categories=order, ordered=True)
        steps.append({"step": "category_order", "column": col, "order": list(order)})
    return frame, steps


def encode_features(
    frame: pd.DataFrame, features: list[str], spec: PreprocessingSpec
) -> tuple[pd.DataFrame, dict[str, str], list[dict]]:
    """범주형 변수를 인코딩하고 파생열 → 원본 변수 매핑을 돌려준다."""
    steps: list[dict] = []
    pieces: list[pd.DataFrame] = []
    origin: dict[str, str] = {}

    for col in features:
        series = frame[col]
        if ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series):
            pieces.append(series.astype(float).to_frame(col))
            origin[col] = col
            continue

        if spec.encoding is EncodeStrategy.ORDINAL:
            ordered = series
            if not isinstance(series.dtype, pd.CategoricalDtype):
                categories = spec.category_orders.get(col) or list(pd.unique(series.dropna()))
                ordered = pd.Categorical(series, categories=categories, ordered=True)
            codes = pd.Series(pd.Categorical(ordered).codes, index=frame.index, dtype=float)
            codes[codes < 0] = np.nan
            pieces.append(codes.to_frame(col))
            origin[col] = col
            steps.append(
                {
                    "step": "encode",
                    "column": col,
                    "strategy": "ordinal",
                    "categories": [str(c) for c in pd.Categorical(ordered).categories],
                }
            )
        else:
            dummies = pd.get_dummies(series, prefix=col, prefix_sep="=", drop_first=spec.drop_first, dtype=float)
            if dummies.empty or dummies.shape[1] == 0:
                continue
            pieces.append(dummies)
            for name in dummies.columns:
                origin[str(name)] = col
            steps.append(
                {
                    "step": "encode",
                    "column": col,
                    "strategy": "onehot",
                    "drop_first": spec.drop_first,
                    "created": [str(c) for c in dummies.columns],
                }
            )

    if not pieces:
        raise AnalysisError("인코딩 후 사용할 수 있는 독립변수가 없습니다.")

    matrix = pd.concat(pieces, axis=1)
    return matrix, origin, steps


def scale_matrix(matrix: pd.DataFrame, strategy: ScaleStrategy) -> tuple[pd.DataFrame, list[dict]]:
    if strategy is ScaleStrategy.NONE or matrix.empty:
        return matrix, []
    scaler = {
        ScaleStrategy.STANDARD: StandardScaler,
        ScaleStrategy.MINMAX: MinMaxScaler,
        ScaleStrategy.ROBUST: RobustScaler,
    }[strategy]()
    scaled = pd.DataFrame(scaler.fit_transform(matrix), columns=matrix.columns, index=matrix.index)
    return scaled, [{"step": "scale", "strategy": strategy.value, "columns": list(map(str, matrix.columns))}]


def build_design_matrix(
    frame: pd.DataFrame,
    *,
    features: list[str],
    target: str | None,
    weight_column: str | None,
    spec: PreprocessingSpec,
    target_as_category: bool = False,
    extra_columns: list[str] | None = None,
) -> DesignMatrix:
    """필터 → 결측 처리 → 이상치 제거 → 범주 순서 → 인코딩 → 스케일링.

    extra_columns는 위계 집단(group_column)·시점(time_column)처럼 인코딩 대상은
    아니지만 전처리(필터·결측 처리)는 함께 거쳐야 하는 변수에 사용한다.
    결과의 frame(design.frame)에는 남지만 X(인코딩된 설계행렬)에는 포함되지 않는다.
    """
    steps: list[dict] = []
    extras = [c for c in (target, weight_column, *(extra_columns or [])) if c]
    used = list(dict.fromkeys([*features, *extras]))
    missing_cols = [c for c in used if c not in frame.columns]
    if missing_cols:
        msg = f"데이터셋에 없는 변수입니다: {', '.join(missing_cols)}"
        raise AnalysisError(msg)

    work = frame[used].copy()

    work, s = apply_filters(work, spec.filters)
    steps += s

    if spec.drop_duplicates:
        before = len(work)
        work = work.drop_duplicates()
        steps.append({"step": "drop_duplicates", "rows_before": before, "rows_after": len(work)})

    work, s = handle_missing(work, spec)
    steps += s

    if spec.remove_outliers:
        cols = spec.outlier_columns or [c for c in used if ptypes.is_numeric_dtype(work[c])]
        work, s = remove_outlier_rows(work, cols, spec.outlier_method, spec.outlier_threshold)
        steps += s

    work, s = apply_category_orders(work, spec.category_orders)
    steps += s

    if work.empty:
        raise AnalysisError("전처리 후 남은 데이터가 없습니다. 필터·결측 처리 조건을 확인하세요.")

    weights = None
    if weight_column:
        weights = pd.to_numeric(work[weight_column], errors="coerce")
        if weights.isna().any():
            msg = f"가중치 변수 '{weight_column}'에 수치가 아닌 값이 있습니다."
            raise AnalysisError(msg)
        if (weights < 0).any():
            msg = f"가중치 변수 '{weight_column}'에 음수가 있습니다."
            raise AnalysisError(msg)
        steps.append({"step": "weights", "column": weight_column, "sum": float(weights.sum())})

    X, origin, s = encode_features(work, features, spec)
    steps += s

    X = X.dropna()
    if X.empty:
        raise AnalysisError("인코딩 후 유효한 행이 없습니다.")

    X, s = scale_matrix(X, spec.scaling)
    steps += s

    y = None
    classes: list[Any] | None = None
    if target:
        raw = work.loc[X.index, target]
        if target_as_category:
            order = spec.category_orders.get(target)
            categorical = pd.Categorical(raw, categories=order, ordered=bool(order)) if order else pd.Categorical(raw)
            classes = [str(c) for c in categorical.categories]
            y = pd.Series(categorical.codes, index=X.index, name=target)
            if (y < 0).any():
                valid = y >= 0
                X, y = X[valid], y[valid]
        else:
            y = pd.to_numeric(raw, errors="coerce")
            valid = y.notna()
            X, y = X[valid], y[valid]
            if y.empty:
                msg = f"종속변수 '{target}'를 수치로 변환할 수 없습니다."
                raise AnalysisError(msg)

    if weights is not None:
        weights = weights.loc[X.index]

    if len(X) < 3:
        msg = f"분석 가능한 행이 {len(X)}건뿐입니다. 최소 3건 이상 필요합니다."
        raise AnalysisError(msg)

    return DesignMatrix(
        X=X,
        y=y,
        weights=weights,
        frame=work.loc[X.index],
        feature_names=[str(c) for c in X.columns],
        feature_origin=origin,
        target_classes=classes,
        steps=steps,
    )
