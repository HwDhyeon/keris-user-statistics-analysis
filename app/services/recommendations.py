"""차트 추천과 차트용 데이터 산출.

렌더링은 웹 클라이언트가 수행한다. 서버는 두 가지만 제공한다.

1. **추천** — 분석 기법에 최적화된 차트 종류와 그 근거, 어떤 필드를 어느 축에 둘지.
2. **데이터** — 그 차트를 그리는 데 필요한 집계 데이터(구간별 빈도, 좌표, 행렬 등).

원시 데이터를 그대로 내보내지 않고 차트 단위로 집계·표본추출해 응답 크기를 제한한다.
"""

from typing import Any

import numpy as np
import pandas as pd

from app.models.enums import AnalysisMethod, ChartKind, ColumnRole
from app.schemas.analysis import ChartDataOptions, ChartRecommendation
from app.services.analysis.base import AnalysisOutcome, jsonable, num, sample_points

# 기법별 기본 추천. 엔진이 제시한 추천이 우선하고, 이 표는 누락분을 보완한다.
DEFAULT_RECOMMENDATIONS: dict[AnalysisMethod, list[ChartKind]] = {
    AnalysisMethod.DESCRIPTIVE: [ChartKind.HISTOGRAM, ChartKind.BOXPLOT, ChartKind.BAR],
    AnalysisMethod.CORRELATION: [ChartKind.HEATMAP, ChartKind.SCATTER],
    AnalysisMethod.LINEAR_REGRESSION: [ChartKind.SCATTER_WITH_FIT, ChartKind.RESIDUAL, ChartKind.BAR],
    AnalysisMethod.LOGISTIC_REGRESSION: [ChartKind.CONFUSION_MATRIX, ChartKind.ROC_CURVE, ChartKind.BAR],
    AnalysisMethod.DECISION_TREE_REGRESSOR: [ChartKind.TREE_DIAGRAM, ChartKind.FEATURE_IMPORTANCE],
    AnalysisMethod.DECISION_TREE_CLASSIFIER: [
        ChartKind.TREE_DIAGRAM,
        ChartKind.FEATURE_IMPORTANCE,
        ChartKind.CONFUSION_MATRIX,
    ],
    AnalysisMethod.KMEANS: [ChartKind.CLUSTER_SCATTER, ChartKind.ELBOW],
    AnalysisMethod.HIERARCHICAL: [ChartKind.CLUSTER_SCATTER, ChartKind.DENDROGRAM],
    AnalysisMethod.DBSCAN: [ChartKind.CLUSTER_SCATTER],
    AnalysisMethod.PCA: [ChartKind.SCREE, ChartKind.HEATMAP],
    AnalysisMethod.ANOVA: [ChartKind.MEANS_PLOT, ChartKind.BOXPLOT, ChartKind.BAR],
    AnalysisMethod.MULTILEVEL: [ChartKind.BAR, ChartKind.RANDOM_EFFECTS, ChartKind.VARIANCE_COMPONENTS],
    AnalysisMethod.GROWTH_CURVE: [ChartKind.TRAJECTORY, ChartKind.BAR, ChartKind.VARIANCE_COMPONENTS],
    AnalysisMethod.PSM: [ChartKind.PS_DISTRIBUTION, ChartKind.LOVE_PLOT],
}


def recommend(outcome: AnalysisOutcome, method: AnalysisMethod) -> list[ChartRecommendation]:
    """엔진 추천과 기법별 기본값을 병합해 우선순위 순으로 반환한다.

    Args:
        outcome (AnalysisOutcome): 분석 엔진이 산출한 결과(추천 목록 포함).
        method (AnalysisMethod): 수행된 분석 기법.

    Returns:
        list[ChartRecommendation]: 엔진 추천과 기법별 기본 추천을 중복 없이 병합해 우선순위 순으로 정렬한 목록.
    """

    seen = {r.kind for r in outcome.recommendations}
    merged = list(outcome.recommendations)
    for i, kind in enumerate(DEFAULT_RECOMMENDATIONS.get(method, [])):
        if kind not in seen:
            merged.append(
                ChartRecommendation(
                    kind=kind,
                    title=kind.value,
                    reason="해당 분석 기법의 표준 시각화",
                    priority=10 + i,
                )
            )

    return sorted(merged, key=lambda r: r.priority)


def build_chart_data(outcome: AnalysisOutcome, method: AnalysisMethod, options: ChartDataOptions) -> dict[str, Any]:
    """추천 차트가 참조할 데이터를 키별로 만든다.

    Args:
        outcome (AnalysisOutcome): 분석 엔진이 산출한 결과(원시 산출물 포함).
        method (AnalysisMethod): 수행된 분석 기법.
        options (ChartDataOptions): 표본 점 개수·히스토그램 구간 수 등 차트 데이터 산출 옵션.

    Returns:
        dict[str, Any]: 차트 종류별로 그릴 재료가 되는 집계 데이터 딕셔너리. include=False면 빈 딕셔너리.
    """

    if not options.include:
        return {}

    a = outcome.artifacts
    data: dict[str, Any] = {}

    if (predictions := a.get("predictions")) is not None and not predictions.empty:
        sampled = sample_points(predictions, options.max_points)
        data["predictions"] = jsonable(sampled)
        data["residual_histogram"] = _histogram(sampled["residual"], options.histogram_bins)

    if (roc := a.get("roc")) is not None and not roc.empty:
        data["roc_curve"] = jsonable(_thin(roc, 500))

    if (projection := a.get("projection")) is not None and not projection.empty:
        sampled = sample_points(projection, options.max_points)
        data["projection"] = jsonable(sampled)
        counts = sampled["cluster"].value_counts().sort_index()
        data["cluster_sizes"] = [{"cluster": str(k), "count": int(v)} for k, v in counts.items()]

    if (centroid_projection := a.get("centroid_projection")) is not None and not centroid_projection.empty:
        data["centroid_projection"] = jsonable(centroid_projection)

    if (cluster_rows := a.get("cluster_rows")) is not None and not cluster_rows.empty:
        data["cluster_rows"] = jsonable(sample_points(cluster_rows, options.max_points))

    if cluster_profiles := a.get("cluster_profiles"):
        data["cluster_profiles"] = jsonable(cluster_profiles)

    if elbow := a.get("elbow"):
        data["elbow"] = jsonable(elbow)

    if scree := a.get("scree"):
        data["scree"] = jsonable(scree)

    if confusion := a.get("confusion_matrix"):
        data["confusion_matrix"] = _confusion_cells(confusion)

    if coefficients := a.get("coefficients"):
        data["coefficients"] = _coefficient_rows(coefficients)

    if importance := a.get("importance"):
        data["feature_importance"] = jsonable(importance)

    if tree := a.get("tree"):
        data["tree"] = _tree_graph(tree)

    if dendrogram := a.get("dendrogram"):
        data["dendrogram"] = jsonable(dendrogram)

    for key in (
        "group_means",
        "group_boxplots",
        "anova_table",
        "random_effects",
        "variance_components",
        "trajectories",
        "propensity_distribution",
        "balance",
        "att",
    ):
        if (value := a.get(key)) is not None:
            data[key] = jsonable(value)

    if method is AnalysisMethod.CORRELATION:
        if (columns := a.get("columns")) and (matrix := a.get("matrix")):
            data["correlation_matrix"] = _matrix_cells(columns, matrix)
        frame = a.get("frame")
        if frame is not None and (pairs := a.get("pairs")):
            data["scatter_pairs"] = scatter_pairs(frame, pairs, options.max_points)

    if method is AnalysisMethod.DESCRIPTIVE and (frame := a.get("frame")) is not None:
        data |= dataset_chart_data(frame, options)

    return data


def dataset_chart_data(frame: pd.DataFrame, options: ChartDataOptions) -> dict[str, Any]:
    """데이터셋 탐색용 분포 데이터 (기술통계·리포트 공용).

    Args:
        frame (pd.DataFrame): 대상 데이터프레임.
        options (ChartDataOptions): 히스토그램 구간 수·범주 상위 개수 등 차트 데이터 산출 옵션.

    Returns:
        dict[str, Any]: 히스토그램, 상자그림, 범주별 빈도, 결측 비율 등을 담은 집계 데이터.
    """
    from app.services.ingest import infer_role
    from app.services.profiling import numeric_columns

    data: dict[str, Any] = {}

    histograms: dict[str, list[dict[str, Any]]] = {}
    boxplots: list[dict[str, Any]] = []
    for col in numeric_columns(frame)[:20]:
        values = pd.to_numeric(frame[col], errors="coerce").dropna()
        if values.empty:
            continue
        histograms[str(col)] = _histogram(values, options.histogram_bins)
        boxplots.append(_boxplot_stats(values, str(col)))
    if histograms:
        data["histograms"] = histograms
    if boxplots:
        data["boxplots"] = boxplots

    categories: dict[str, list[dict[str, Any]]] = {}
    for col in frame.columns:
        if infer_role(frame[col]) not in {ColumnRole.CATEGORICAL, ColumnRole.BOOLEAN}:
            continue
        counts = frame[col].value_counts().head(options.max_categories)
        total = int(frame[col].notna().sum()) or 1
        categories[str(col)] = [
            {"value": str(k), "count": int(v), "ratio": round(int(v) / total, 6)} for k, v in counts.items()
        ]
        if len(categories) >= 20:
            break
    if categories:
        data["category_counts"] = categories

    missing = [
        {"column": str(col), "ratio": round(float(ratio), 6), "count": int(frame[col].isna().sum())}
        for col, ratio in frame.isna().mean().items()
        if ratio > 0
    ]
    if missing:
        data["missing_ratios"] = sorted(missing, key=lambda m: m["ratio"], reverse=True)

    return data


def dataset_recommendations(frame: pd.DataFrame, has_correlation: bool) -> list[ChartRecommendation]:
    """데이터셋 탐색 리포트용 추천.

    Args:
        frame (pd.DataFrame): 대상 데이터프레임.
        has_correlation (bool): 상관관계 분석 결과(2개 이상의 수치형 변수)가 있는지 여부.

    Returns:
        list[ChartRecommendation]: 상관 히트맵·산점도·히스토그램·상자그림·결측 매트릭스 등 추천 차트 목록.
    """
    recommendations: list[ChartRecommendation] = []
    if has_correlation:
        recommendations.append(
            ChartRecommendation(
                kind=ChartKind.HEATMAP,
                title="상관계수 히트맵",
                reason="변수 간 관계를 한눈에 파악해 분석 모델 수립의 근거로 삼는다",
                priority=1,
                data_key="correlation_matrix",
                encoding={"x": "x", "y": "y", "color": "r"},
                options={"domain": [-1, 1], "diverging": True},
            )
        )
        recommendations.append(
            ChartRecommendation(
                kind=ChartKind.SCATTER,
                title="상위 상관 쌍 산점도",
                reason="관계의 형태(선형/비선형)와 이상치 영향 확인",
                priority=2,
                data_key="scatter_pairs",
                encoding={"x": "x", "y": "y"},
                options={"group_by": "pair", "fit_line": True},
            )
        )

    from app.services.profiling import numeric_columns

    if numeric_columns(frame):
        recommendations.append(
            ChartRecommendation(
                kind=ChartKind.HISTOGRAM,
                title="수치형 변수 분포",
                reason="분포 형태와 치우침 확인",
                priority=3,
                data_key="histograms",
                encoding={"x": "bin_start", "y": "count"},
                options={"grouped_by_column": True},
            )
        )
        recommendations.append(
            ChartRecommendation(
                kind=ChartKind.BOXPLOT,
                title="상자그림",
                reason="사분위 범위와 이상치 경계 확인",
                priority=4,
                data_key="boxplots",
                encoding={"x": "column", "y": "median"},
                options={"whisker_fields": ["lower_fence", "q1", "median", "q3", "upper_fence"]},
            )
        )
    if frame.isna().any().any():
        recommendations.append(
            ChartRecommendation(
                kind=ChartKind.MISSING_MATRIX,
                title="변수별 결측률",
                reason="결측 발생 구조 확인",
                priority=5,
                data_key="missing_ratios",
                encoding={"x": "ratio", "y": "column"},
                options={"sort": "-x", "format": "percent"},
            )
        )
    return recommendations


# --- 데이터 변환 헬퍼 -----------------------------------------------------


def _histogram(values: pd.Series, bins: int) -> list[dict[str, Any]]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return []
    counts, edges = np.histogram(clean.to_numpy(), bins=min(bins, max(clean.nunique(), 1)))
    return [
        {
            "bin_start": num(edges[i]),
            "bin_end": num(edges[i + 1]),
            "bin_center": num((edges[i] + edges[i + 1]) / 2),
            "count": int(c),
        }
        for i, c in enumerate(counts)
    ]


def _boxplot_stats(values: pd.Series, column: str) -> dict[str, Any]:
    q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = values[(values >= lower) & (values <= upper)]
    outliers = values[(values < lower) | (values > upper)]

    return {
        "column": column,
        "min": num(values.min()),
        "q1": num(q1),
        "median": num(values.median()),
        "q3": num(q3),
        "max": num(values.max()),
        # 수염 끝은 울타리 안쪽의 실제 최솟값·최댓값
        "lower_fence": num(inside.min()) if not inside.empty else num(values.min()),
        "upper_fence": num(inside.max()) if not inside.empty else num(values.max()),
        "outliers": [num(v) for v in outliers.head(50)],
        "n_outliers": int(outliers.size),
    }


def _thin(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    """ROC 곡선처럼 순서가 의미 있는 데이터는 균등 간격으로 솎아낸다.

    Args:
        frame (pd.DataFrame): 솎아낼 데이터프레임.
        limit (int): 남길 최대 행 수.

    Returns:
        pd.DataFrame: 균등 간격으로 추출된 행만 남은 데이터프레임. 원본이 limit 이하면 그대로 반환.
    """

    if len(frame) <= limit:
        return frame

    indices = np.linspace(0, len(frame) - 1, limit).astype(int)
    return frame.iloc[indices]


def _matrix_cells(columns: list[str], matrix: list[list[float | None]]) -> dict[str, Any]:
    cells = [
        {"x": columns[i], "y": columns[j], "r": matrix[i][j]}
        for i in range(len(columns))
        for j in range(len(columns))
        if matrix and matrix[i][j] is not None
    ]

    return {"columns": columns, "cells": cells}


def _confusion_cells(confusion: dict[str, Any]) -> dict[str, Any]:
    labels, matrix = confusion["labels"], confusion["matrix"]

    return {
        "labels": labels,
        "cells": [
            {"actual": labels[i], "predicted": labels[j], "count": int(matrix[i][j])}
            for i in range(len(labels))
            for j in range(len(labels))
        ],
    }


def _coefficient_rows(coefficients: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """절편을 제외하고 막대그래프에 바로 쓸 수 있는 행으로 정리한다.

    Args:
        coefficients (list[dict[str, Any]]): 회귀 계수(또는 오즈비) 원본 목록.

    Returns:
        list[dict[str, Any]]: 절편(const)을 제외하고 값·유의성·신뢰구간을 정리한 행 목록.
    """

    rows: list[dict[str, Any]] = []
    for row in coefficients:
        if not isinstance(row, dict) or "term" not in row or row["term"] == "const":
            continue

        value = row.get("odds_ratio", row.get("coefficient"))
        rows.append(
            {
                "term": row["term"],
                "source_variable": row.get("source_variable"),
                "value": value,
                "measure": "odds_ratio" if row.get("odds_ratio") is not None else "coefficient",
                "p_value": row.get("p_value"),
                "significant": bool(row.get("significant")),
                "ci_lower": row.get("or_ci_lower", row.get("ci_lower")),
                "ci_upper": row.get("or_ci_upper", row.get("ci_upper")),
            }
        )

    return rows


def _tree_graph(tree: dict[str, Any]) -> dict[str, Any]:
    """중첩 트리를 노드/링크 목록과 계층 구조 양쪽으로 제공한다.

    Args:
        tree (dict[str, Any]): 의사결정나무의 중첩된 노드 구조(루트 노드부터 children으로 재귀).

    Returns:
        dict[str, Any]: 평탄화된 노드 목록(nodes), 부모-자식 연결 목록(links), 원본 계층 구조(hierarchy).
    """

    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []

    def walk(node: dict[str, Any], parent: int | None, branch: str | None) -> None:
        nodes.append(
            {
                "id": node["node_id"],
                "parent": parent,
                "depth": node["depth"],
                "branch": branch,
                "label": node.get("condition") or f"→ {node.get('predicted_class', node.get('predicted_value'))}",
                "n_samples": node["n_samples"],
                "impurity": node["impurity"],
                "is_leaf": node["is_leaf"],
                "predicted": node.get("predicted_class", node.get("predicted_value")),
                "confidence": node.get("confidence"),
            }
        )

        if parent is not None:
            links.append({"source": parent, "target": node["node_id"], "branch": branch})

        for i, child in enumerate(node.get("children", [])):
            walk(child, node["node_id"], "yes" if i == 0 else "no")

    walk(tree, None, None)
    return {"nodes": nodes, "links": links, "hierarchy": tree}


def scatter_pairs(frame: pd.DataFrame, pairs: list[Any], max_points: int) -> list[dict[str, Any]]:
    """상관이 높은 상위 쌍의 좌표를 산점도용으로 추출한다.

    Args:
        frame (pd.DataFrame): 원본 데이터프레임.
        pairs (list[Any]): 상관계수 순으로 정렬된 변수 쌍 목록(CorrelationPair 등, x/y/coefficient/p_value 속성 필요).
        max_points (int): 전체 산점도 점 개수 상한. 쌍당 배분되는 표본 수 계산에 사용.

    Returns:
        list[dict[str, Any]]: 상위 4개 쌍에 대해 좌표·상관계수·p-value를 담은 산점도 데이터 목록.
    """

    out: list[dict[str, Any]] = []
    per_pair = max(max_points // 4, 100)
    for pair in pairs[:4]:
        subset = frame[[pair.x, pair.y]].dropna()

        if subset.empty:
            continue

        sampled = sample_points(subset, per_pair)
        out.append(
            {
                "pair": f"{pair.x} vs {pair.y}",
                "x_label": pair.x,
                "y_label": pair.y,
                "coefficient": pair.coefficient,
                "p_value": pair.p_value,
                "points": [{"x": num(x), "y": num(y)} for x, y in zip(sampled[pair.x], sampled[pair.y], strict=True)],
            }
        )

    return out
