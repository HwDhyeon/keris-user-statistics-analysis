"""데이터 반입 및 프로파일링."""

from fastapi.testclient import TestClient


def test_upload_infers_roles(client: TestClient, dataset_id: str) -> None:
    body = client.get(f"/api/v1/datasets/{dataset_id}").json()
    assert body["n_rows"] == 300
    assert body["n_cols"] == 9

    roles = {c["name"]: c["role"] for c in body["columns"]}
    assert roles["학습시간"] == "numeric"
    assert roles["학년"] == "categorical"
    assert roles["학생ID"] == "identifier"

    # 범주 순서는 원본 등장 순서로 보존된다.
    categories = next(c["categories"] for c in body["columns"] if c["name"] == "학년")
    assert sorted(categories) == ["1학년", "2학년", "3학년"]


def test_rejects_unsupported_extension(client: TestClient) -> None:
    response = client.post(
        "/api/v1/datasets/upload", files={"file": ("bad.exe", b"binary", "application/octet-stream")}
    )
    assert response.status_code == 400
    assert response.json()["code"] == "ingest_error"


def test_preview_and_columns(client: TestClient, dataset_id: str) -> None:
    preview = client.get(f"/api/v1/datasets/{dataset_id}/preview", params={"limit": 5}).json()
    assert preview["returned_rows"] == 5
    assert preview["total_rows"] == 300

    columns = client.get(f"/api/v1/datasets/{dataset_id}/columns").json()
    assert len(columns) == 9


def test_profile_detects_missing_and_outliers(client: TestClient, dataset_id: str) -> None:
    profile = client.get(f"/api/v1/datasets/{dataset_id}/profile").json()

    attendance = next(m for m in profile["missing"] if m["column"] == "출석률")
    assert attendance["n_missing"] == 5

    study = next(s for s in profile["numeric"] if s["column"] == "학습시간")
    assert study["q1"] < study["median"] < study["q3"]
    assert study["mean"] is not None

    outliers = {o["column"]: o["n_outliers"] for o in profile["outliers"]}
    assert outliers["학습시간"] >= 1


def test_correlation_finds_linear_relationship(client: TestClient, dataset_id: str) -> None:
    body = client.get(f"/api/v1/datasets/{dataset_id}/correlation").json()
    pair = next(
        p for p in body["pairs"]
        if {p["x"], p["y"]} == {"학습시간", "성취도점수"}
    )
    assert pair["coefficient"] > 0.5
    assert pair["p_value"] < 0.01
    assert body["heatmap_cells"][0].keys() == {"x", "y", "r"}


def test_report_bundles_profile_and_correlation(client: TestClient, dataset_id: str) -> None:
    body = client.get(f"/api/v1/datasets/{dataset_id}/report").json()
    assert body["profile"]["n_rows"] == 300
    assert body["correlation"] is not None
    assert len(body["chart_recommendations"]) > 0
