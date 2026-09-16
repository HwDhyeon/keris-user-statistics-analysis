"""데이터셋 물리 저장소 (Parquet 기반)."""

import hashlib
from pathlib import Path

import pandas as pd

from app.core.config import settings


def compute_checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dataset_path(dataset_id: str) -> Path:
    return settings.dataset_dir / f"{dataset_id}.parquet"


def save_frame(dataset_id: str, frame: pd.DataFrame) -> Path:
    path = dataset_path(dataset_id)
    # 열 이름이 문자열이 아니면 Parquet 저장이 실패한다.
    frame = frame.rename(columns=str)
    frame.to_parquet(path, index=False)
    return path


def load_frame(dataset_id: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = dataset_path(dataset_id)
    if not path.exists():
        msg = f"데이터셋 파일을 찾을 수 없습니다: {path.name}"
        raise FileNotFoundError(msg)
    return pd.read_parquet(path, columns=columns)


def delete_frame(dataset_id: str) -> None:
    dataset_path(dataset_id).unlink(missing_ok=True)
