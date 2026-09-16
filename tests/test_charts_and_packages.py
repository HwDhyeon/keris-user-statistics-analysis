"""차트 추천·차트용 데이터, 그리고 패키지 보존·재현·공유."""

import io
import json
import zipfile

from fastapi.testclient import TestClient


def _regression(client: TestClient, dataset_id: str, **overrides) -> dict:
    payload = {
        "dataset_id": dataset_id,
        "method": "linear_regression",
        "params": {"target": "성취도점수", "features": ["학습시간", "출석률"], "test_size": 0.2},
        "preprocessing": {"missing_strategy": "median"},
    } | overrides
    response = client.post("/api/v1/analyses", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def test_regression_recommends_charts_without_rendering(client: TestClient, dataset_id: str) -> None:
    body = _regression(client, dataset_id)
    recommendations = body["chart_recommendations"]

    # 회귀분석에는 회귀선이 포함된 산점도가 최우선 추천된다.
    assert recommendations[0]["kind"] == "scatter_with_fit"
    assert recommendations[0]["options"]["fit_line"] is True
    kinds = {r["kind"] for r in recommendations}
    assert {"scatter_with_fit", "residual", "bar", "histogram"} <= kinds

    # 추천은 그릴 데이터의 위치와 축 매핑을 가리킨다. 렌더링 결과물(스펙/이미지)은 없다.
    residual = next(r for r in recommendations if r["kind"] == "residual")
    assert residual["data_key"] == "predictions"
    assert residual["encoding"] == {"x": "predicted", "y": "residual"}
    assert residual["options"]["reference_line"] == 0
    assert all("spec" not in r and "image_url" not in r for r in recommendations)


def test_chart_data_matches_recommendation_keys(client: TestClient, dataset_id: str) -> None:
    body = _regression(client, dataset_id)
    data = body["chart_data"]

    # 모든 추천의 data_key가 실제 데이터로 채워져 있어야 한다.
    for recommendation in body["chart_recommendations"]:
        key = recommendation["data_key"]
        if key is not None:
            assert key in data, f"{recommendation['kind']}의 데이터({key})가 없습니다"

    point = data["predictions"][0]
    assert {"actual", "predicted", "residual"} <= point.keys()
    assert data["residual_histogram"][0].keys() >= {"bin_start", "bin_end", "count"}

    coefficient = next(c for c in data["coefficients"] if c["term"] == "학습시간")
    assert coefficient["measure"] == "coefficient"
    assert coefficient["significant"] is True
    # 절편은 막대그래프 대상이 아니다.
    assert all(c["term"] != "const" for c in data["coefficients"])


def test_chart_data_can_be_skipped_and_rebuilt(client: TestClient, dataset_id: str) -> None:
    body = _regression(client, dataset_id, chart_data={"include": False})
    assert body["chart_data"] == {}
    # 데이터를 빼도 추천은 그대로 제공된다.
    assert len(body["chart_recommendations"]) > 0

    rebuilt = client.post(
        f"/api/v1/analyses/{body['id']}/chart-data",
        json={"include": True, "max_points": 120, "histogram_bins": 8},
    )
    assert rebuilt.status_code == 200, rebuilt.text
    data = rebuilt.json()["chart_data"]
    assert len(data["predictions"]) <= 120
    assert len(data["residual_histogram"]) <= 8


def test_chart_data_endpoint_filters_by_key(client: TestClient, dataset_id: str) -> None:
    body = _regression(client, dataset_id)
    response = client.get(
        f"/api/v1/analyses/{body['id']}/chart-data", params={"key": ["coefficients"]}
    )
    assert response.status_code == 200
    assert set(response.json()) == {"coefficients"}

    recommendations = client.get(f"/api/v1/analyses/{body['id']}/chart-recommendations")
    assert recommendations.status_code == 200
    assert recommendations.json()[0]["kind"] == "scatter_with_fit"


def test_logistic_chart_data(client: TestClient, dataset_id: str) -> None:
    body = client.post("/api/v1/analyses", json={
        "dataset_id": dataset_id,
        "method": "logistic_regression",
        "params": {
            "target": "합격여부",
            "features": ["학습시간", "출석률"],
            "positive_label": "합격",
            "test_size": 0.3,
        },
        "preprocessing": {"missing_strategy": "median"},
    }).json()
    data = body["chart_data"]

    assert data["confusion_matrix"]["labels"] == ["불합격", "합격"]
    assert len(data["confusion_matrix"]["cells"]) == 4
    assert data["roc_curve"][0].keys() == {"fpr", "tpr"}
    assert data["coefficients"][0]["measure"] == "odds_ratio"


def test_tree_chart_data_is_graph_shaped(client: TestClient, dataset_id: str) -> None:
    body = client.post("/api/v1/analyses", json={
        "dataset_id": dataset_id,
        "method": "decision_tree_classifier",
        "params": {"target": "합격여부", "features": ["학습시간", "출석률"], "max_depth": 3},
        "preprocessing": {"missing_strategy": "median"},
    }).json()
    tree = body["chart_data"]["tree"]

    assert tree["nodes"][0]["parent"] is None
    assert len(tree["links"]) == len(tree["nodes"]) - 1
    assert {link["branch"] for link in tree["links"]} == {"yes", "no"}
    assert tree["hierarchy"]["node_id"] == 0

    importance = body["chart_data"]["feature_importance"]
    assert importance[0]["importance"] >= importance[-1]["importance"]


def test_cluster_chart_data(client: TestClient, dataset_id: str) -> None:
    body = client.post("/api/v1/analyses", json={
        "dataset_id": dataset_id,
        "method": "kmeans",
        "params": {"features": ["학습시간", "출석률", "성취도점수"], "auto_k_range": [2, 4]},
        "preprocessing": {"missing_strategy": "median", "scaling": "standard"},
    }).json()
    data = body["chart_data"]

    # 변수가 3개이므로 좌표는 주성분으로 축약되고, 추천이 그 사실을 알려준다.
    scatter = next(r for r in body["chart_recommendations"] if r["kind"] == "cluster_scatter")
    assert scatter["options"]["pca_reduced"] is True
    assert scatter["encoding"] == {"x": "x", "y": "y", "color": "cluster"}
    assert data["projection"][0]["x_label"] == "주성분1"

    assert sum(c["count"] for c in data["cluster_sizes"]) == len(data["projection"])
    assert data["elbow"][0].keys() >= {"k", "inertia", "silhouette"}


def test_descriptive_chart_data(client: TestClient, dataset_id: str) -> None:
    body = client.post("/api/v1/analyses", json={
        "dataset_id": dataset_id, "method": "descriptive",
        "preprocessing": {"missing_strategy": "none"},
    }).json()
    data = body["chart_data"]

    assert "학습시간" in data["histograms"]
    box = next(b for b in data["boxplots"] if b["column"] == "학습시간")
    assert box["lower_fence"] <= box["q1"] <= box["median"] <= box["q3"] <= box["upper_fence"]
    assert box["n_outliers"] >= 1
    assert {c["value"] for c in data["category_counts"]["학년"]} == {"1학년", "2학년", "3학년"}
    assert any(m["column"] == "출석률" for m in data["missing_ratios"])


def test_dataset_report_provides_recommendations_and_data(client: TestClient, dataset_id: str) -> None:
    body = client.get(f"/api/v1/datasets/{dataset_id}/report").json()
    assert body["profile"]["n_rows"] == 300

    kinds = {r["kind"] for r in body["chart_recommendations"]}
    assert {"heatmap", "scatter", "histogram", "boxplot"} <= kinds

    data = body["chart_data"]
    assert data["correlation_matrix"]["cells"][0].keys() == {"x", "y", "r"}
    pair = data["scatter_pairs"][0]
    assert pair["points"][0].keys() == {"x", "y"}
    assert body["correlation"]["heatmap_cells"]


def test_package_bundles_and_reproduces(client: TestClient, dataset_id: str) -> None:
    analysis = _regression(client, dataset_id)

    created = client.post("/api/v1/packages", json={
        "name": "학업성취도 회귀분석 패키지",
        "description": "학습시간·출석률이 성취도에 미치는 영향",
        "analysis_ids": [analysis["id"]],
        "include_data_snapshot": True,
    })
    assert created.status_code == 201, created.text
    package = created.json()
    assert package["archive_bytes"] > 0
    assert package["manifest"]["environment"]["packages"]["scikit-learn"]
    assert package["manifest"]["analyses"][0]["chart_recommendations"][0]["kind"] == "scatter_with_fit"

    download = client.get(f"/api/v1/packages/{package['id']}/download")
    assert download.status_code == 200
    with zipfile.ZipFile(io.BytesIO(download.content)) as zf:
        names = zf.namelist()
        assert "manifest.json" in names
        assert "README.md" in names
        assert any(n.endswith("/report.md") for n in names)
        assert any(n.endswith("/chart-recommendations.json") for n in names)
        assert any(n.endswith("/chart-data.json") for n in names)
        assert any(n.startswith("data/") for n in names)
        # 렌더링 산출물은 더 이상 패키지에 들어가지 않는다.
        assert not any(n.endswith((".png", ".vl.json")) for n in names)

        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["analyses"][0]["preprocessing"]["missing_strategy"] == "median"

    reproduced = client.post(f"/api/v1/packages/{package['id']}/reproduce", json={})
    assert reproduced.status_code == 200, reproduced.text
    body = reproduced.json()
    assert body["matched"] is True, body["comparisons"]
    assert body["comparisons"][0]["differences"] == []


def test_package_sharing(client: TestClient, dataset_id: str) -> None:
    analysis = _regression(client, dataset_id)
    package = client.post("/api/v1/packages", json={
        "name": "공유 테스트", "analysis_ids": [analysis["id"]],
    }).json()

    share = client.post(f"/api/v1/packages/{package['id']}/share").json()
    token = share["share_token"]
    assert token

    shared = client.get(f"/api/v1/shared/{token}")
    assert shared.status_code == 200
    assert shared.json()["id"] == package["id"]
    assert client.get(f"/api/v1/shared/{token}/download").status_code == 200

    client.delete(f"/api/v1/packages/{package['id']}/share")
    assert client.get(f"/api/v1/shared/{token}").status_code == 404


def test_failed_analysis_cannot_be_packaged(client: TestClient, dataset_id: str) -> None:
    failed = client.post("/api/v1/analyses", json={
        "dataset_id": dataset_id,
        "method": "linear_regression",
        "params": {"target": "성취도점수", "features": ["없는변수"]},
    }).json()
    assert failed["status"] == "failed"

    response = client.post("/api/v1/packages", json={"name": "x", "analysis_ids": [failed["id"]]})
    assert response.status_code == 400
