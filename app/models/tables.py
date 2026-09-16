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
    """반입된 사용자 데이터셋 1건을 나타내는 테이블.

    Attributes:
        id (str): 데이터셋 고유 식별자(UUID hex).
        name (str): 데이터셋 이름.
        description (str | None): 데이터셋 설명.
        original_filename (str): 업로드 당시 원본 파일명.
        source_format (str): 원본 파일 포맷(csv, xlsx 등).
        storage_path (str): 파싱된 데이터가 저장된 경로.
        checksum (str): 원본 파일 내용의 체크섬.
        size_bytes (int): 원본 파일 크기(바이트).
        n_rows (int): 데이터 행 수.
        n_cols (int): 데이터 열 수.
        status (DatasetStatus): 반입 처리 상태.
        error (str | None): 반입 실패 시 오류 메시지.
        ingest_options (dict[str, Any]): 반입 시 적용된 옵션(시트, 구분자, 인코딩 등).
        created_at (datetime): 생성 일시.
        columns (list[DatasetColumn]): 이 데이터셋에 속한 변수(열) 메타데이터 목록.
        analyses (list[Analysis]): 이 데이터셋을 대상으로 수행된 분석 목록.
    """

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
    """데이터셋을 구성하는 변수(열) 하나의 메타데이터.

    Attributes:
        id (str): 열 메타데이터 고유 식별자(UUID hex).
        dataset_id (str): 소속 데이터셋의 식별자.
        name (str): 열 이름.
        position (int): 데이터셋 내 열 순서(0부터 시작).
        dtype (str): 원본 데이터 타입.
        role (ColumnRole): 프로파일링으로 추론한 변수 역할.
        n_missing (int): 결측값 개수.
        missing_ratio (float): 결측 비율(0~1).
        n_unique (int): 고유값 개수.
        categories (list[Any] | None): 범주형 변수의 범주 목록(등장/사전 순서 보존, ORDINAL 인코딩 기준).
        stats (dict[str, Any]): 열에 대한 기술통계 등 부가 통계 정보.
        dataset (Dataset | None): 이 열이 속한 데이터셋.
    """

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
    """수행된 분석 1건을 나타내는 테이블. 분석 조건과 전처리 로직을 함께 보존한다.

    Attributes:
        id (str): 분석 고유 식별자(UUID hex).
        dataset_id (str): 분석 대상 데이터셋의 식별자.
        name (str): 분석 이름.
        method (AnalysisMethod): 사용된 분석 기법.
        status (AnalysisStatus): 분석 진행 상태.
        params (dict[str, Any]): 분석 실행에 사용된 파라미터.
        preprocessing (dict[str, Any]): 분석 전 적용된 전처리 로직/옵션.
        result (dict[str, Any]): 분석 결과 데이터.
        metrics (dict[str, Any]): 결정계수·RMSE 등 모델 성능 지표.
        chart_recommendations (list[dict[str, Any]]): 추천 차트 목록(무엇을 그릴지).
        chart_data (dict[str, Any]): 차트를 그리는 데 필요한 집계 데이터. 렌더링은 클라이언트가 수행한다.
        error (str | None): 분석 실패 시 오류 메시지.
        duration_ms (int | None): 분석 소요 시간(밀리초).
        created_at (datetime): 생성 일시.
        completed_at (datetime | None): 완료 일시.
        dataset (Dataset | None): 분석 대상 데이터셋.
    """

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
    """분석 조건·전처리·결과 리포트를 묶은 재현 가능 패키지.

    Attributes:
        id (str): 패키지 고유 식별자(UUID hex).
        name (str): 패키지 이름.
        description (str | None): 패키지 설명.
        dataset_id (str | None): 연관된 데이터셋의 식별자.
        analysis_ids (list[str]): 패키지에 포함된 분석 식별자 목록.
        manifest (dict[str, Any]): 패키지 구성 내역(조건·전처리·결과 요약 등).
        archive_path (str | None): 생성된 ZIP 아카이브의 파일 경로.
        archive_bytes (int): 아카이브 파일 크기(바이트).
        share_token (str | None): 공유 링크 발급 시 사용되는 토큰.
        created_at (datetime): 생성 일시.
    """

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
