"""군집분석(K-평균, 계층적, DBSCAN) 및 주성분분석."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from app.core.exceptions import AnalysisError
from app.models.enums import AnalysisMethod, ChartKind
from app.schemas.analysis import AnalysisParams, PreprocessingSpec
from app.services.analysis.base import AnalysisOutcome, jsonable, num, rec
from app.services.preprocessing import build_design_matrix


def _quality(X: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """군집 품질 지표를 계산한다. 잡음(-1)은 제외하고 계산한다.

    Args:
        X (np.ndarray): 군집화에 사용된 설계행렬.
        labels (np.ndarray): 각 표본에 배정된 군집 라벨(잡음은 -1).

    Returns:
        dict[str, Any]: 실루엣 계수, Calinski-Harabasz 지수, Davies-Bouldin 지수.
            유효 군집이 2개 미만이거나 표본이 3개 미만이면 모두 None.
    """
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
    """군집별 크기와 변수 평균(군집 해석 근거)을 계산한다.

    z_means는 전체 평균 대비 표준화 편차로, 군집별 프로파일 비교 막대차트에 바로 쓸 수 있다.

    Args:
        frame (pd.DataFrame): 원본(비인코딩) 데이터프레임.
        labels (np.ndarray): 각 표본에 배정된 군집 라벨.
        features (list[str]): 프로파일을 계산할 변수 목록.

    Returns:
        list[dict[str, Any]]: 군집 번호 순으로 정렬된 군집별 크기·비율·평균·중앙값·표준화 편차 목록.
    """
    tagged = frame.copy()
    tagged["_cluster"] = labels
    numeric_all = tagged[features].apply(pd.to_numeric, errors="coerce")
    overall_mean = numeric_all.mean()
    overall_std = numeric_all.std(ddof=0).replace(0, np.nan)

    rows: list[dict[str, Any]] = []
    for cluster, group in tagged.groupby("_cluster"):
        numeric = group[features].apply(pd.to_numeric, errors="coerce")
        means = numeric.mean()
        z = (means - overall_mean) / overall_std
        rows.append(
            {
                "cluster": int(cluster),
                "label": "잡음" if int(cluster) < 0 else f"군집 {int(cluster)}",
                "size": len(group),
                "ratio": round(len(group) / len(tagged), 6),
                "means": {c: num(means[c]) for c in numeric.columns},
                "medians": {c: num(numeric[c].median()) for c in numeric.columns},
                "z_means": {c: num(z[c]) for c in numeric.columns},
            }
        )
    return sorted(rows, key=lambda r: r["cluster"])


def _anova_f_tests(frame: pd.DataFrame, labels: np.ndarray, features: list[str]) -> list[dict[str, Any]]:
    """변수별 일원분산분석(F검정): 군집이 각 변수를 통계적으로 유의하게 구분하는지 확인한다.

    Args:
        frame (pd.DataFrame): 원본(비인코딩) 데이터프레임.
        labels (np.ndarray): 각 표본에 배정된 군집 라벨.
        features (list[str]): 검정할 변수 목록.

    Returns:
        list[dict[str, Any]]: 변수별 F통계량, p-value, 유의수준(0.001/0.01/0.05)별 유의 여부.
            군집이 2개 미만이면 해당 변수는 결과에서 제외된다.
    """
    unique_labels = np.unique(labels)
    results: list[dict[str, Any]] = []
    for col in features:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        groups = [values[labels == c].dropna().to_numpy() for c in unique_labels]
        groups = [g for g in groups if g.size > 1]
        if len(groups) < 2:
            continue
        f_stat, p_value = sps.f_oneway(*groups)
        f_stat, p_value = num(f_stat), num(p_value)
        results.append(
            {
                "feature": col,
                "f_statistic": f_stat,
                "p_value": p_value,
                "significant_001": p_value is not None and p_value < 0.001,
                "significant_01": p_value is not None and p_value < 0.01,
                "significant_05": p_value is not None and p_value < 0.05,
            }
        )
    return results


def _cluster_rows(
    frame: pd.DataFrame,
    X: pd.DataFrame,
    labels: np.ndarray,
    centroids: np.ndarray,
    profiles: list[dict[str, Any]],
    features: list[str],
    label_columns: list[str],
) -> pd.DataFrame:
    """표본별 원시데이터 + 대표변수 실측/예측(군집평균)/잔차 + 중심으로부터 거리를 담은 표를 만든다.

    예측치는 각 표본이 속한 군집의 대표변수(첫 번째 피처) 평균값이다. 거리는 표준화된
    설계행렬(X) 공간에서 배정된 군집 중심까지의 유클리드 거리로, 군집 적합도의 지표다.

    Args:
        frame (pd.DataFrame): 원본(비인코딩) 데이터프레임.
        X (pd.DataFrame): 인코딩·스케일링된 설계행렬.
        labels (np.ndarray): 각 표본에 배정된 군집 라벨.
        centroids (np.ndarray): 군집 중심 좌표(X와 동일한 공간).
        profiles (list[dict[str, Any]]): _profiles가 계산한 군집별 프로파일(대표변수 평균 조회용).
        features (list[str]): 표에 포함할 변수 목록.
        label_columns (list[str]): 원시데이터·잔차표에 함께 표시할 식별자 변수 목록.

    Returns:
        pd.DataFrame: 행 ID, 군집, 실측/예측/잔차, 중심까지 거리, 식별자·특성 변수를 담은 표.
    """
    primary = features[0]
    cluster_mean = {p["cluster"]: p["means"].get(primary) for p in profiles}

    actual = pd.to_numeric(frame[primary], errors="coerce").to_numpy(dtype=float)
    predicted = np.array([cluster_mean.get(int(c)) for c in labels], dtype=float)
    residual = actual - predicted
    with np.errstate(divide="ignore", invalid="ignore"):
        residual_ratio = np.where(actual != 0, residual / actual * 100, np.nan)

    diffs = X.to_numpy() - centroids[labels]
    distance = np.sqrt((diffs**2).sum(axis=1))

    data: dict[str, Any] = {
        "row_id": frame.index.astype(str),
        "cluster": labels.astype(str),
        "actual": actual,
        "predicted": predicted,
        "residual": residual,
        "residual_ratio": residual_ratio,
        "distance_to_centroid": distance,
    }
    for col in label_columns:
        if col in frame.columns and col not in data:
            data[col] = frame[col].astype(str).to_numpy()
    for col in features:
        if col not in data:
            data[col] = pd.to_numeric(frame[col], errors="coerce").to_numpy()
    return pd.DataFrame(data, index=frame.index)


def _projection(
    X: pd.DataFrame, labels: np.ndarray, centroids: np.ndarray | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """2차원 산점도용 좌표를 계산한다. 변수가 3개 이상이면 PCA로 축약한다.

    열 이름은 x/y로 고정하고, 축이 실제로 무엇인지는 x_label/y_label에 담는다.
    centroids를 주면 포인트와 동일한 좌표계(동일 PCA 변환)로 투영해 함께 반환한다.

    Args:
        X (pd.DataFrame): 군집화에 사용된 설계행렬.
        labels (np.ndarray): 각 표본에 배정된 군집 라벨.
        centroids (np.ndarray | None, optional): 함께 투영할 군집 중심 좌표. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame | None]: 표본별 2차원 좌표(x, y, cluster, x_label, y_label)와,
            centroids를 지정한 경우 동일 좌표계로 투영된 중심점 좌표(없으면 None).
    """
    centroid_values: np.ndarray | None = None
    if X.shape[1] >= 2:
        reduced = X.shape[1] > 2
        if reduced:
            pca = PCA(n_components=2, random_state=42)
            values = pca.fit_transform(X.to_numpy())
            if centroids is not None:
                centroid_values = pca.transform(centroids)
        else:
            values = X.to_numpy()
            if centroids is not None:
                centroid_values = centroids
        axes = ("주성분1", "주성분2") if reduced else (str(X.columns[0]), str(X.columns[1]))
    else:
        values = np.column_stack([X.to_numpy().ravel(), np.zeros(len(X))])
        if centroids is not None:
            centroid_values = np.column_stack([centroids[:, 0], np.zeros(len(centroids))])
        axes = (str(X.columns[0]), "")

    points = pd.DataFrame(
        {
            "x": values[:, 0],
            "y": values[:, 1],
            "cluster": labels.astype(str),
            "x_label": axes[0],
            "y_label": axes[1],
        }
    )
    centroid_df = None
    if centroid_values is not None:
        centroid_df = pd.DataFrame(
            {
                "cluster": [str(i) for i in range(len(centroid_values))],
                "x": centroid_values[:, 0],
                "y": centroid_values[:, 1],
            }
        )
    return points, centroid_df


def _cluster_scatter_rec(
    X: pd.DataFrame, reason: str = "군집 간 분리 정도를 2차원에서 확인", has_centroids: bool = False
) -> Any:
    """군집 산점도 추천을 만든다. 변수가 3개 이상이면 좌표가 주성분으로 축약됨을 알린다.

    Args:
        X (pd.DataFrame): 군집화에 사용된 설계행렬(차원 축소 여부 판단용).
        reason (str, optional): 추천 이유. Defaults to "군집 간 분리 정도를 2차원에서 확인".
        has_centroids (bool, optional): 중심점 데이터(centroid_projection)를 함께 제공하는지 여부. Defaults to False.

    Returns:
        Any: 군집 산점도 차트 추천 객체(ChartRecommendation).
    """
    reduced = X.shape[1] > 2
    return rec(
        ChartKind.CLUSTER_SCATTER,
        "군집 산점도",
        reason + (" (주성분 2개로 축약된 좌표)" if reduced else ""),
        1,
        data_key="projection",
        encoding={"x": "x", "y": "y", "color": "cluster"},
        options={
            "pca_reduced": reduced,
            **({"centroid_data_key": "centroid_projection"} if has_centroids else {}),
        },
    )


class KMeansEngine:
    method = AnalysisMethod.KMEANS

    def run(self, frame: pd.DataFrame, params: AnalysisParams, spec: PreprocessingSpec) -> AnalysisOutcome:
        design = build_design_matrix(
            frame, features=params.features, target=None,
            weight_column=params.weight_column, spec=spec,
            extra_columns=params.label_columns,
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

        # 군집 간/전체 분산비(SSB/SST): 군집이 전체 변동을 얼마나 설명하는지.
        grand_mean = X.mean(axis=0).to_numpy()
        total_ss = float(np.square(X.to_numpy() - grand_mean).sum())
        between_ss = total_ss - float(model.inertia_)
        variance_ratio = (between_ss / total_ss) if total_ss > 0 else None

        metrics = _quality(X.to_numpy(), labels) | {
            "inertia": num(model.inertia_),
            "n_iter": int(model.n_iter_),
            "n_clusters": k,
            "between_ss": num(between_ss),
            "total_ss": num(total_ss),
            "variance_ratio": num(variance_ratio),
        }

        profiles = _profiles(design.frame, labels, params.features)
        anova_tests = _anova_f_tests(design.frame, labels, params.features)
        sig_features = [t["feature"] for t in anova_tests if t["significant_001"]]
        summary = (
            f"전체 {len(X)}개 표본이 {k}개 군집으로 분류되었으며, "
            + (
                f"{', '.join(sig_features)} 변수의 군집 간 분산 F검정 결과 모두 p < .001 수준에서 유의함."
                if sig_features and len(sig_features) == len(anova_tests)
                else f"{', '.join(sig_features)} 변수는 군집 간 유의한 차이를 보임(p < .001)."
                if sig_features
                else "군집 간 변수별 분산 차이가 통계적으로 유의하지 않은 변수가 있음."
            )
        )

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
            "cluster_profiles": profiles,
            "anova": anova_tests,
            "summary": summary,
            "labels_preview": labels[:200].tolist(),
            "preprocessing_steps": design.steps,
        }

        projection, centroid_projection = _projection(X, labels, model.cluster_centers_)
        cluster_rows = _cluster_rows(
            design.frame, X, labels, model.cluster_centers_, profiles, params.features, params.label_columns
        )
        artifacts = {
            "projection": projection,
            "centroid_projection": centroid_projection,
            "cluster_rows": cluster_rows,
            "cluster_profiles": profiles,
            "elbow": elbow,
            "design": design,
            "labels": labels,
        }
        recommendations = [
            _cluster_scatter_rec(X, has_centroids=True),
            rec(
                ChartKind.BAR, "군집별 크기", "군집 균형 확인", 3,
                data_key="cluster_sizes",
                encoding={"x": "cluster", "y": "count"},
            ),
            rec(
                ChartKind.BAR, "군집별 주요 지표 프로파일", "전체 평균 대비 군집별 편차(표준화) 비교", 4,
                data_key="cluster_profiles",
                encoding={"x": "z_means", "y": "feature", "color": "cluster"},
                options={"diverging": True, "reference_line": 0},
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
        projection, _ = _projection(X, labels)
        artifacts = {"projection": projection, "dendrogram": dendrogram, "design": design}
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

        projection, _ = _projection(X, labels)
        artifacts = {"projection": projection, "design": design}
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
