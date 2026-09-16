"""군집분석(K-평균, 계층적, DBSCAN) 및 주성분분석."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from app.core.exceptions import AnalysisError
from app.models.enums import AnalysisMethod, ChartKind
from app.schemas.analysis import AnalysisParams, PreprocessingSpec
from app.services.analysis.base import AnalysisOutcome, jsonable, num, rec
from app.services.preprocessing import build_design_matrix


def _quality(X: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """군집 품질 지표. 잡음(-1)은 제외하고 계산한다."""
    mask = labels >= 0
    unique = np.unique(labels[mask])
    if unique.size < 2 or mask.sum() < 3:
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}
    sub_X, sub_labels = X[mask], labels[mask]
    return {
        "silhouette": num(silhouette_score(sub_X, sub_labels)),
        "calinski_harabasz": num(calinski_harabasz_score(sub_X, sub_labels)),
        "davies_bouldin": num(davies_bouldin_score(sub_X, sub_labels)),
    }


def _profiles(frame: pd.DataFrame, labels: np.ndarray, features: list[str]) -> list[dict[str, Any]]:
    """군집별 크기와 변수 평균(군집 해석 근거)."""
    tagged = frame.copy()
    tagged["_cluster"] = labels
    rows: list[dict[str, Any]] = []
    for cluster, group in tagged.groupby("_cluster"):
        numeric = group[features].apply(pd.to_numeric, errors="coerce")
        rows.append(
            {
                "cluster": int(cluster),
                "label": "잡음" if int(cluster) < 0 else f"군집 {int(cluster)}",
                "size": len(group),
                "ratio": round(len(group) / len(tagged), 6),
                "means": {c: num(numeric[c].mean()) for c in numeric.columns},
                "medians": {c: num(numeric[c].median()) for c in numeric.columns},
            }
        )
    return sorted(rows, key=lambda r: r["cluster"])


def _projection(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """2차원 산점도용 좌표. 변수가 3개 이상이면 PCA로 축약한다.

    열 이름은 x/y로 고정하고, 축이 실제로 무엇인지는 x_label/y_label에 담는다.
    """
    if X.shape[1] >= 2:
        reduced = X.shape[1] > 2
        values = PCA(n_components=2, random_state=42).fit_transform(X) if reduced else X.to_numpy()
        axes = ("주성분1", "주성분2") if reduced else (str(X.columns[0]), str(X.columns[1]))
    else:
        values = np.column_stack([X.to_numpy().ravel(), np.zeros(len(X))])
        axes = (str(X.columns[0]), "")
    return pd.DataFrame(
        {
            "x": values[:, 0],
            "y": values[:, 1],
            "cluster": labels.astype(str),
            "x_label": axes[0],
            "y_label": axes[1],
        }
    )


def _cluster_scatter_rec(
    X: pd.DataFrame, reason: str = "군집 간 분리 정도를 2차원에서 확인"
) -> Any:
    """군집 산점도 추천. 변수가 3개 이상이면 좌표가 주성분으로 축약됨을 알린다."""
    reduced = X.shape[1] > 2
    return rec(
        ChartKind.CLUSTER_SCATTER,
        "군집 산점도",
        reason + (" (주성분 2개로 축약된 좌표)" if reduced else ""),
        1,
        data_key="projection",
        encoding={"x": "x", "y": "y", "color": "cluster"},
        options={"pca_reduced": reduced},
    )


class KMeansEngine:
    method = AnalysisMethod.KMEANS

    def run(self, frame: pd.DataFrame, params: AnalysisParams, spec: PreprocessingSpec) -> AnalysisOutcome:
        design = build_design_matrix(
            frame, features=params.features, target=None,
            weight_column=params.weight_column, spec=spec,
        )
        X = design.X
        weights = design.weights.to_numpy() if design.weights is not None else None

        elbow: list[dict[str, Any]] = []
        k = params.n_clusters
        if params.auto_k_range:
            lo, hi = int(params.auto_k_range[0]), int(params.auto_k_range[1])
            hi = min(hi, len(X) - 1)
            if lo < 2 or hi < lo:
                msg = f"auto_k_range가 올바르지 않습니다: [{lo}, {hi}]"
                raise AnalysisError(msg)
            for candidate in range(lo, hi + 1):
                model = KMeans(n_clusters=candidate, n_init=10, random_state=params.random_state)
                labels = model.fit_predict(X, sample_weight=weights)
                sil = _quality(X.to_numpy(), labels)["silhouette"]
                elbow.append({"k": candidate, "inertia": num(model.inertia_), "silhouette": sil})
            best = max(elbow, key=lambda e: e["silhouette"] if e["silhouette"] is not None else -1)
            k = int(best["k"])

        if k >= len(X):
            msg = f"군집 수({k})가 데이터 행 수({len(X)})보다 많거나 같을 수 없습니다."
            raise AnalysisError(msg)

        model = KMeans(n_clusters=k, n_init=10, random_state=params.random_state)
        labels = model.fit_predict(X, sample_weight=weights)

        metrics = _quality(X.to_numpy(), labels) | {
            "inertia": num(model.inertia_),
            "n_iter": int(model.n_iter_),
            "n_clusters": k,
        }
        result = {
            "method": "K-평균 군집분석",
            "features": params.features,
            "n_clusters": k,
            "auto_selected": bool(params.auto_k_range),
            "elbow": elbow,
            "n_observations": len(X),
            "weighted": weights is not None,
            "centroids": [
                {"cluster": i, **{str(c): num(v) for c, v in zip(X.columns, row, strict=True)}}
                for i, row in enumerate(model.cluster_centers_)
            ],
            "cluster_profiles": _profiles(design.frame, labels, params.features),
            "labels_preview": labels[:200].tolist(),
            "preprocessing_steps": design.steps,
        }

        artifacts = {"projection": _projection(X, labels), "elbow": elbow, "design": design, "labels": labels}
        recommendations = [
            _cluster_scatter_rec(X),
            rec(
                ChartKind.BAR, "군집별 크기", "군집 균형 확인", 3,
                data_key="cluster_sizes",
                encoding={"x": "cluster", "y": "count"},
            ),
        ]
        if elbow:
            recommendations.insert(
                1,
                rec(
                    ChartKind.ELBOW, "엘보 곡선", "적정 군집 수 판단 근거", 2,
                    data_key="elbow",
                    encoding={"x": "k", "y": "inertia"},
                    options={"secondary_y": "silhouette", "selected_k": k},
                ),
            )
        return AnalysisOutcome(jsonable(result), jsonable(metrics), artifacts, recommendations)


class HierarchicalEngine:
    method = AnalysisMethod.HIERARCHICAL

    def run(self, frame: pd.DataFrame, params: AnalysisParams, spec: PreprocessingSpec) -> AnalysisOutcome:
        design = build_design_matrix(
            frame, features=params.features, target=None,
            weight_column=params.weight_column, spec=spec,
        )
        X = design.X
        if params.n_clusters >= len(X):
            msg = f"군집 수({params.n_clusters})가 데이터 행 수({len(X)})보다 작아야 합니다."
            raise AnalysisError(msg)

        model = AgglomerativeClustering(n_clusters=params.n_clusters, linkage=params.linkage)
        labels = model.fit_predict(X)

        # 덴드로그램 병합 이력 (행 수가 많으면 생략)
        dendrogram: dict[str, Any] | None = None
        if len(X) <= 1000:
            from scipy.cluster.hierarchy import linkage as scipy_linkage

            method = "ward" if params.linkage == "ward" else params.linkage
            z = scipy_linkage(X.to_numpy(), method=method)
            dendrogram = {
                "merges": [
                    {"left": int(a), "right": int(b), "distance": num(d), "size": int(n)}
                    for a, b, d, n in z
                ],
                "n_leaves": len(X),
            }

        metrics = _quality(X.to_numpy(), labels) | {"n_clusters": params.n_clusters}
        result = {
            "method": "계층적 군집분석",
            "features": params.features,
            "linkage": params.linkage,
            "n_clusters": params.n_clusters,
            "n_observations": len(X),
            "cluster_profiles": _profiles(design.frame, labels, params.features),
            "dendrogram": dendrogram,
            "labels_preview": labels[:200].tolist(),
            "preprocessing_steps": design.steps,
        }
        artifacts = {"projection": _projection(X, labels), "dendrogram": dendrogram, "design": design}
        recommendations = [
            _cluster_scatter_rec(X),
            rec(
                ChartKind.DENDROGRAM, "덴드로그램", "군집 병합 과정과 절단 높이 확인", 2,
                data_key="dendrogram",
                encoding={"label": "distance"},
                options={"merge_list": "merges", "n_leaves": "n_leaves"},
            ),
        ]
        return AnalysisOutcome(jsonable(result), jsonable(metrics), artifacts, recommendations)


class DBSCANEngine:
    method = AnalysisMethod.DBSCAN

    def run(self, frame: pd.DataFrame, params: AnalysisParams, spec: PreprocessingSpec) -> AnalysisOutcome:
        design = build_design_matrix(
            frame, features=params.features, target=None,
            weight_column=params.weight_column, spec=spec,
        )
        X = design.X
        labels = DBSCAN(eps=params.eps, min_samples=params.min_samples).fit_predict(X)

        n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
        n_noise = int((labels == -1).sum())
        metrics = _quality(X.to_numpy(), labels) | {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": round(n_noise / len(X), 6),
        }
        result = {
            "method": "DBSCAN 밀도기반 군집분석",
            "features": params.features,
            "eps": params.eps,
            "min_samples": params.min_samples,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "n_observations": len(X),
            "cluster_profiles": _profiles(design.frame, labels, params.features),
            "labels_preview": labels[:200].tolist(),
            "preprocessing_steps": design.steps,
        }
        if n_clusters == 0:
            result["warning"] = "군집이 형성되지 않았습니다. eps를 키우거나 min_samples를 줄여보세요."

        artifacts = {"projection": _projection(X, labels), "design": design}
        recommendations = [
            _cluster_scatter_rec(X, reason="밀도 기반 군집과 잡음점(-1) 분포 확인")
        ]
        return AnalysisOutcome(jsonable(result), jsonable(metrics), artifacts, recommendations)


class PCAEngine:
    method = AnalysisMethod.PCA

    def run(self, frame: pd.DataFrame, params: AnalysisParams, spec: PreprocessingSpec) -> AnalysisOutcome:
        design = build_design_matrix(
            frame, features=params.features, target=None, weight_column=None, spec=spec
        )
        X = design.X
        n_components = min(params.n_components, X.shape[1], len(X))
        model = PCA(n_components=n_components, random_state=params.random_state)
        scores = model.fit_transform(X)

        ratios = model.explained_variance_ratio_
        components = [
            {
                "component": f"주성분{i + 1}",
                "explained_variance": num(model.explained_variance_[i]),
                "explained_variance_ratio": num(ratios[i]),
                "cumulative_ratio": num(float(np.cumsum(ratios)[i])),
                "loadings": {str(c): num(v) for c, v in zip(X.columns, model.components_[i], strict=True)},
            }
            for i in range(n_components)
        ]
        metrics = {
            "n_components": n_components,
            "total_explained_variance": num(float(ratios.sum())),
            "first_component_ratio": num(ratios[0]) if len(ratios) else None,
        }
        result = {
            "method": "주성분분석",
            "features": params.features,
            "n_observations": len(X),
            "components": components,
            "scores_preview": [
                {f"주성분{j + 1}": num(v) for j, v in enumerate(row)} for row in scores[:200]
            ],
            "preprocessing_steps": design.steps,
        }
        artifacts = {
            "scree": [
                {
                    "component": c["component"],
                    "ratio": c["explained_variance_ratio"],
                    "cumulative": c["cumulative_ratio"],
                }
                for c in components
            ],
            "loadings": [
                {"component": c["component"], "variable": variable, "loading": value}
                for c in components
                for variable, value in c["loadings"].items()
            ],
            "design": design,
        }
        recommendations = [
            rec(
                ChartKind.SCREE, "스크리 도표", "적정 주성분 수 판단", 1,
                data_key="scree",
                encoding={"x": "component", "y": "ratio"},
                options={"secondary_y": "cumulative", "format": "percent"},
            ),
            rec(
                ChartKind.HEATMAP, "성분 적재량 히트맵", "변수-성분 기여도 확인", 2,
                data_key="loadings",
                encoding={"x": "component", "y": "variable", "color": "loading"},
                options={"domain": [-1, 1], "diverging": True},
            ),
        ]
        return AnalysisOutcome(jsonable(result), jsonable(metrics), artifacts, recommendations)
