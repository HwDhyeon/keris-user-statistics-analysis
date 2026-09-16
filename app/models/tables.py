"""SQLModel 테이블 정의."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import Column, Text
from sqlalchemy.types import JSON
from sqlmodel import Field, Relationship, SQLModel

from app.models.enums import AnalysisMethod, AnalysisStatus, ColumnRole, DatasetStatus


def _uuid() -> str:
    return uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class Dataset(SQLModel, table=True):
    """반입된 사용자 데이터셋."""

    __tablename__ = "dataset"

    id: str = Field(default_factory=_uuid, primary_key=True)
    name: str = Field(index=True)
    description: str | None = None
    original_filename: str
    source_format: str
    storage_path: str
    checksum: str = Field(index=True)
    size_bytes: int = 0
    n_rows: int = 0
    n_cols: int = 0
    status: DatasetStatus = Field(default=DatasetStatus.READY)
    error: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    ingest_options: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_now, index=True)

    columns: list[DatasetColumn] = Relationship(
        back_populates="dataset",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "order_by": "DatasetColumn.position"},
    )
    analyses: list[Analysis] = Relationship(
        back_populates="dataset", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


class DatasetColumn(SQLModel, table=True):
    """데이터셋 변수(열) 메타데이터."""

    __tablename__ = "dataset_column"

    id: str = Field(default_factory=_uuid, primary_key=True)
    dataset_id: str = Field(foreign_key="dataset.id", index=True)
    name: str
    position: int
    dtype: str
    role: ColumnRole
    n_missing: int = 0
    missing_ratio: float = 0.0
    n_unique: int = 0
    # 범주 순서를 원본 등장/사전 순으로 보존한다. 분석 시 ORDINAL 인코딩 기준이 된다.
    categories: list[Any] | None = Field(default=None, sa_column=Column(JSON))
    stats: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    dataset: Dataset | None = Relationship(back_populates="columns")


class Analysis(SQLModel, table=True):
    """수행된 분석 1건. 조건과 전처리 로직을 함께 보존한다."""

    __tablename__ = "analysis"

    id: str = Field(default_factory=_uuid, primary_key=True)
    dataset_id: str = Field(foreign_key="dataset.id", index=True)
    name: str
    method: AnalysisMethod = Field(index=True)
    status: AnalysisStatus = Field(default=AnalysisStatus.PENDING, index=True)
    params: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    preprocessing: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    result: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    metrics: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    # 차트 렌더링은 클라이언트가 수행한다. 서버는 "무엇을 그릴지"와 "그릴 데이터"만 보관한다.
    chart_recommendations: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    chart_data: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    error: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    duration_ms: int | None = None
    created_at: datetime = Field(default_factory=_now, index=True)
    completed_at: datetime | None = None

    dataset: Dataset | None = Relationship(back_populates="analyses")


class AnalysisPackage(SQLModel, table=True):
    """분석 조건·전처리·결과 리포트를 묶은 재현 가능 패키지."""

    __tablename__ = "analysis_package"

    id: str = Field(default_factory=_uuid, primary_key=True)
    name: str = Field(index=True)
    description: str | None = None
    dataset_id: str | None = Field(default=None, index=True)
    analysis_ids: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    manifest: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    archive_path: str | None = None
    archive_bytes: int = 0
    share_token: str | None = Field(default=None, index=True, unique=True)
    created_at: datetime = Field(default_factory=_now, index=True)
