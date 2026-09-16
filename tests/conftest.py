import io
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 설정 캐시가 만들어지기 전에 임시 저장소를 지정한다.
_TMP = Path(tempfile.mkdtemp(prefix="usa-test-"))
os.environ["USA_STORAGE_DIR"] = str(_TMP)
os.environ["USA_DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"

from fastapi.testclient import TestClient  # noqa: E402

from app.core.database import init_db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _db() -> None:
    init_db()


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def sample_csv() -> bytes:
    """선형 관계·범주형·결측·이상치를 모두 포함한 합성 데이터."""
    rng = np.random.default_rng(42)
    n = 300
    study = rng.normal(5, 1.5, n).clip(0.5, None)
    attendance = rng.uniform(60, 100, n)
    grade = rng.choice(["1학년", "2학년", "3학년"], n)
    region = rng.choice(["수도권", "광역시", "기타"], n, p=[0.5, 0.3, 0.2])
    score = 30 + 6 * study + 0.3 * attendance + rng.normal(0, 4, n)

    # 프로그램 참여는 학습시간(공변량)과 상관되도록 구성해 PSM의 선택편의(selection bias)를 재현한다.
    participation_score = 0.6 * (study - study.mean()) + rng.normal(0, 1.2, n)
    participation = participation_score > np.quantile(participation_score, 0.6)

    frame = pd.DataFrame(
        {
            "학생ID": [f"S{i:04d}" for i in range(n)],
            "학습시간": study.round(2),
            "출석률": attendance.round(1),
            "학년": grade,
            "지역": region,
            "성취도점수": score.round(1),
            "합격여부": np.where(score > score.mean(), "합격", "불합격"),
            "가중치": rng.uniform(0.5, 2.0, n).round(3),
            "프로그램참여": np.where(participation, "참여", "미참여"),
        }
    )
    frame.loc[:4, "출석률"] = np.nan  # 결측치
    frame.loc[5, "학습시간"] = 15.0  # 이상치

    buffer = io.BytesIO()
    frame.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def dataset_id(client: TestClient, sample_csv: bytes) -> str:
    response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("성취도.csv", sample_csv, "text/csv")},
        data={"name": "학업성취도 표본"},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


@pytest.fixture(scope="session")
def panel_csv() -> bytes:
    """학생(패널 개체) × 시점 구조의 위계·성장모형용 합성 데이터.

    학교마다 다른 절편(기저 수준)을 부여하고, 학생마다 다른 성장 기울기를 부여해
    다층모형(ICC)과 성장모형(확률기울기)이 실제로 검출할 신호를 만든다.
    """
    rng = np.random.default_rng(7)
    n_schools = 12
    students_per_school = 15
    waves = [0, 1, 2]

    school_effect = rng.normal(0, 6, n_schools)  # 학교별 절편 편차
    rows: list[dict] = []
    student_id = 0
    for school in range(n_schools):
        school_type = "공립" if school % 2 == 0 else "사립"
        for _ in range(students_per_school):
            base = 50 + school_effect[school] + rng.normal(0, 5)
            slope = 4 + rng.normal(0, 1.5)  # 학생별 성장 기울기 편차
            for wave in waves:
                score = base + slope * wave + rng.normal(0, 3)
                rows.append(
                    {
                        "학생ID": f"P{student_id:04d}",
                        "학교ID": f"school_{school:02d}",
                        "학교유형": school_type,
                        "시점": wave,
                        "점수": round(float(score), 2),
                    }
                )
            student_id += 1

    frame = pd.DataFrame(rows)
    buffer = io.BytesIO()
    frame.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def panel_dataset_id(client: TestClient, panel_csv: bytes) -> str:
    response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("패널.csv", panel_csv, "text/csv")},
        data={"name": "학생 패널 표본"},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]
