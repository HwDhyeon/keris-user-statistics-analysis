"""데이터셋 반입/조회 스키마."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.enums import ColumnRole, CorrelationMethod, DatasetStatus, OutlierMethod
from app.schemas.analysis import ChartRecommendation


class ColumnRead(BaseModel):
    id: str
    name: str
    position: int
    dtype: str
    role: ColumnRole
    n_missing: int
    missing_ratio: float
    n_unique: int
    categories: list[Any] | None = None
    stats: dict[str, Any] = Field(default_factory=dict)


class DatasetRead(BaseModel):
    id: str
    name: str
    description: str | None = None
    original_filename: str
    source_format: str
    checksum: str
    size_bytes: int
    n_rows: int
    n_cols: int
    status: DatasetStatus
    error: str | None = None
    ingest_options: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class DatasetDetail(DatasetRead):
    columns: list[ColumnRead] = Field(default_factory=list)


class DatasetUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


class DatasetPreview(BaseModel):
    dataset_id: str
    columns: list[str]
    rows: list[dict[str, Any]]
    total_rows: int
    returned_rows: int


class NumericSummary(BaseModel):
    """기초 통계량."""

    column: str
    count: int
    missing: int
    mean: float | None = None
    std: float | None = None
    variance: float | None = None
    min: float | None = None
    q1: float | None = None
    median: float | None = None
    q3: float | None = None
    max: float | None = None
    iqr: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    cv: float | None = Field(default=None, description="변동계수 std/mean")
    sum: float | None = None


class CategoricalSummary(BaseModel):
    column: str
    count: int
    missing: int
    n_unique: int
    mode: Any | None = None
    top_values: list[dict[str, Any]] = Field(default_factory=list)
    entropy: float | None = None


class MissingReport(BaseModel):
    column: str
    n_missing: int
    ratio: float


class OutlierReport(BaseModel):
    column: str
    method: OutlierMethod
    n_outliers: int
    ratio: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    sample_indices: list[int] = Field(default_factory=list)
    sample_values: list[float] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    """반입 직후 제공되는 자동 프로파일링 결과."""

    dataset_id: str
    n_rows: int
    n_cols: int
    n_duplicated_rows: int
    total_missing_cells: int
    missing_ratio: float
    numeric: list[NumericSummary] = Field(default_factory=list)
    categorical: list[CategoricalSummary] = Field(default_factory=list)
    missing: list[MissingReport] = Field(default_factory=list)
    outliers: list[OutlierReport] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CorrelationPair(BaseModel):
    x: str
    y: str
    coefficient: float
    p_value: float | None = None
    strength: str


class CorrelationResult(BaseModel):
    dataset_id: str
    method: CorrelationMethod
    columns: list[str]
    matrix: list[list[float | None]]
    pairs: list[CorrelationPair]
    # 히트맵을 그리기 좋은 (x, y, r) 셀 형태. 렌더링은 클라이언트가 수행한다.
    heatmap_cells: list[dict[str, Any]] = Field(default_factory=list)


class ProfileReport(BaseModel):
    """프로파일 + 상관관계 + 차트 추천/데이터를 묶은 탐색 리포트."""

    dataset: DatasetDetail
    profile: DatasetProfile
    correlation: CorrelationResult | None = None
    chart_recommendations: list[ChartRecommendation] = Field(default_factory=list)
    chart_data: dict[str, Any] = Field(default_factory=dict)
