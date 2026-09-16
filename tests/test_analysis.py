"""분석 기법별 동작 및 성능 지표."""

import pytest
from fastapi.testclient import TestClient


def run(client: TestClient, payload: dict) -> dict:
    response = client.post("/api/v1/analyses", json=payload)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "succeeded", body.get("error")
    return body


def test_methods_catalog(client: TestClient) -> None:
    methods = {m["method"] for m in client.get("/api/v1/methods").json()}
    assert {"linear_regression", "logistic_regression", "decision_tree_classifier", "kmeans"} <= methods


def test_linear_regression_reports_r2_and_rmse(client: TestClient, dataset_id: str) -> None:
    body = run(client, {
        "dataset_id": dataset_id,
        "method": "linear_regression",
        "params": {"target": "성취도점수", "features": ["학습시간", "출석률"], "test_size": 0.25},
        "preprocessing": {"missing_strategy": "median"},
    })
    metrics = body["metrics"]
    assert metrics["r_squared"] > 0.8
    assert metrics["rmse"] > 0
    assert metrics["adjusted_r_squared"] is not None
    assert "durbin_watson" in metrics

    coefficients = {c["term"]: c for c in body["result"]["coefficients"]}
    assert coefficients["학습시간"]["p_value"] < 0.01
    assert coefficients["학습시간"]["significant"] is True
    assert "vif" in body["result"]["diagnostics"]
    assert body["result"]["equation"].startswith("성취도점수 =")


def test_linear_regression_with_weights_and_category_order(client: TestClient, dataset_id: str) -> None:
    body = run(client, {
        "dataset_id": dataset_id,
        "method": "linear_regression",
        "params": {
            "target": "성취도점수",
            "features": ["학습시간", "학년"],
            "weight_column": "가중치",
        },
        "preprocessing": {
            "missing_strategy": "drop_rows",
            "encoding": "ordinal",
            # 변수 값 순서 유지
            "category_orders": {"학년": ["1학년", "2학년", "3학년"]},
        },
    })
    assert body["result"]["weighted"] is True
    steps = body["result"]["preprocessing_steps"]
    order_step = next(s for s in steps if s["step"] == "category_order")
    assert order_step["order"] == ["1학년", "2학년", "3학년"]
    encode_step = next(s for s in steps if s["step"] == "encode")
    assert encode_step["categories"] == ["1학년", "2학년", "3학년"]


def test_category_order_rejects_unknown_value(client: TestClient, dataset_id: str) -> None:
    response = client.post("/api/v1/analyses", json={
        "dataset_id": dataset_id,
        "method": "linear_regression",
        "params": {"target": "성취도점수", "features": ["학습시간", "학년"]},
        "preprocessing": {"category_orders": {"학년": ["1학년", "2학년"]}},
    })
    assert response.status_code == 201
    assert response.json()["status"] == "failed"
    assert "범주 순서" in response.json()["error"]


def test_logistic_regression_reports_auc(client: TestClient, dataset_id: str) -> None:
    body = run(client, {
        "dataset_id": dataset_id,
        "method": "logistic_regression",
        "params": {
            "target": "합격여부",
            "features": ["학습시간", "출석률"],
            "positive_label": "합격",
            "test_size": 0.3,
        },
        "preprocessing": {"missing_strategy": "median"},
    })
    metrics = body["metrics"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["roc_auc"] > 0.8
    assert metrics["mcfadden"] is not None
    assert body["result"]["classes"][1] == "합격"
    assert all("odds_ratio" in c for c in body["result"]["coefficients"])


def test_decision_tree_returns_structure_and_importance(client: TestClient, dataset_id: str) -> None:
    body = run(client, {
        "dataset_id": dataset_id,
        "method": "decision_tree_classifier",
        "params": {
            "target": "합격여부",
            "features": ["학습시간", "출석률", "학년"],
            "max_depth": 4,
            "test_size": 0.25,
        },
        "preprocessing": {"missing_strategy": "median"},
    })
    result = body["result"]
    assert result["tree_depth"] <= 4
    assert result["tree"]["node_id"] == 0
    assert "children" in result["tree"]
    assert result["feature_importance"][0]["importance"] > 0
    assert len(result["rules"]) > 0
    assert body["metrics"]["accuracy"] > 0.7


def test_kmeans_auto_selects_k(client: TestClient, dataset_id: str) -> None:
    body = run(client, {
        "dataset_id": dataset_id,
        "method": "kmeans",
        "params": {"features": ["학습시간", "출석률", "성취도점수"], "auto_k_range": [2, 5]},
        "preprocessing": {"missing_strategy": "median", "scaling": "standard"},
    })
    assert body["result"]["auto_selected"] is True
    assert len(body["result"]["elbow"]) == 4
    assert body["metrics"]["silhouette"] is not None
    assert sum(p["size"] for p in body["result"]["cluster_profiles"]) == body["result"]["n_observations"]


@pytest.mark.parametrize("method,params", [
    ("descriptive", {}),
    ("correlation", {"correlation_method": "spearman"}),
    ("hierarchical", {"features": ["학습시간", "성취도점수"], "n_clusters": 3}),
    ("dbscan", {"features": ["학습시간", "성취도점수"], "eps": 0.6, "min_samples": 5}),
    ("pca", {"features": ["학습시간", "출석률", "성취도점수"], "n_components": 2}),
])
def test_remaining_methods(client: TestClient, dataset_id: str, method: str, params: dict) -> None:
    body = run(client, {
        "dataset_id": dataset_id,
        "method": method,
        "params": params,
        "preprocessing": {"missing_strategy": "median", "scaling": "standard"},
    })
    assert body["metrics"]


def test_methods_catalog_includes_education_methods(client: TestClient) -> None:
    methods = {m["method"] for m in client.get("/api/v1/methods").json()}
    assert {"anova", "multilevel", "growth_curve", "psm"} <= methods


def test_anova_detects_region_group_differences(client: TestClient, dataset_id: str) -> None:
    body = run(client, {
        "dataset_id": dataset_id,
        "method": "anova",
        "params": {"target": "성취도점수", "factors": ["지역"]},
        "preprocessing": {"missing_strategy": "median"},
    })
    result = body["result"]
    assert result["factors"] == ["지역"]
    assert len(result["anova_table"]) >= 1
    assert body["metrics"]["model_r_squared"] is not None
    assert "post_hoc" in result and "지역" in result["post_hoc"]


def test_ancova_adds_covariate_and_two_way_factors(client: TestClient, dataset_id: str) -> None:
    body = run(client, {
        "dataset_id": dataset_id,
        "method": "anova",
        "params": {
            "target": "성취도점수",
            "factors": ["지역", "학년"],
            "covariates": ["출석률"],
        },
        "preprocessing": {"missing_strategy": "median"},
    })
    result = body["result"]
    assert result["covariates"] == ["출석률"]
    assert result["interaction_included"] is True
    terms = {t["term"] for t in result["anova_table"]}
    assert any("×" in t for t in terms)


def test_anova_rejects_single_level_factor(client: TestClient, dataset_id: str) -> None:
    response = client.post("/api/v1/analyses", json={
        "dataset_id": dataset_id,
        "method": "anova",
        "params": {"target": "성취도점수", "factors": ["지역"]},
        "preprocessing": {"filters": [{"column": "지역", "op": "eq", "value": "수도권"}]},
    })
    assert response.status_code == 201
    assert response.json()["status"] == "failed"


def test_multilevel_reports_icc_and_fixed_effects(client: TestClient, panel_dataset_id: str) -> None:
    body = run(client, {
        "dataset_id": panel_dataset_id,
        "method": "multilevel",
        "params": {"target": "점수", "features": ["시점"], "group_column": "학교ID"},
    })
    result = body["result"]
    metrics = body["metrics"]
    assert result["n_groups"] == 12
    assert 0.0 <= metrics["unconditional_icc"] <= 1.0
    fixed = {f["term"]: f for f in result["fixed_effects"]}
    assert "시점" in fixed
    assert fixed["시점"]["significant"] is True


def test_multilevel_requires_group_column(client: TestClient, dataset_id: str) -> None:
    response = client.post("/api/v1/analyses", json={
        "dataset_id": dataset_id,
        "method": "multilevel",
        "params": {"target": "성취도점수", "features": ["학습시간"]},
    })
    assert response.status_code == 422


def test_growth_curve_estimates_average_growth_rate(client: TestClient, panel_dataset_id: str) -> None:
    body = run(client, {
        "dataset_id": panel_dataset_id,
        "method": "growth_curve",
        "params": {
            "target": "점수",
            "time_column": "시점",
            "group_column": "학생ID",
            "random_slope": True,
        },
    })
    result = body["result"]
    metrics = body["metrics"]
    assert result["n_individuals"] > 100
    assert metrics["average_growth_rate"] > 0
    assert metrics["average_growth_rate_p_value"] < 0.01
    assert result["variance_components"]["slope_variance"] is not None
    assert len(result["individual_trajectories_preview"]) > 0


def test_growth_curve_requires_time_and_group_columns(client: TestClient, dataset_id: str) -> None:
    response = client.post("/api/v1/analyses", json={
        "dataset_id": dataset_id,
        "method": "growth_curve",
        "params": {"target": "성취도점수"},
    })
    assert response.status_code == 422


def test_psm_matches_and_estimates_att(client: TestClient, dataset_id: str) -> None:
    body = run(client, {
        "dataset_id": dataset_id,
        "method": "psm",
        "params": {
            "target": "성취도점수",
            "features": ["학습시간", "출석률"],
            "treatment_column": "프로그램참여",
            "positive_label": "참여",
        },
        "preprocessing": {"missing_strategy": "median"},
    })
    result = body["result"]
    metrics = body["metrics"]
    assert result["treated_label"] == "참여"
    assert metrics["n_matched_treated"] > 0
    # 매칭 후 균형(SMD)이 매칭 전보다 개선되어야 한다.
    assert metrics["mean_abs_smd_after"] <= metrics["mean_abs_smd_before"]
    assert result["att"] is not None


def test_psm_requires_treatment_column(client: TestClient, dataset_id: str) -> None:
    response = client.post("/api/v1/analyses", json={
        "dataset_id": dataset_id,
        "method": "psm",
        "params": {"features": ["학습시간"]},
    })
    assert response.status_code == 422


def test_missing_target_is_rejected(client: TestClient, dataset_id: str) -> None:
    response = client.post("/api/v1/analyses", json={
        "dataset_id": dataset_id, "method": "linear_regression", "params": {"features": ["학습시간"]},
    })
    assert response.status_code == 422


def test_unknown_dataset_returns_404(client: TestClient) -> None:
    response = client.post("/api/v1/analyses", json={
        "dataset_id": "nope", "method": "descriptive", "params": {},
    })
    assert response.status_code == 404
