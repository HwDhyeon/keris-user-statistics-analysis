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
    """수치형 변수의 기초 통계량.

    Attributes:
        column (str): 변수(열) 이름.
        count (int): 결측을 제외한 유효 관측치 수.
        missing (int): 결측치 개수.
        mean (float | None): 평균.
        std (float | None): 표준편차.
        variance (float | None): 분산.
        min (float | None): 최솟값.
        q1 (float | None): 1사분위수.
        median (float | None): 중앙값.
        q3 (float | None): 3사분위수.
        max (float | None): 최댓값.
        iqr (float | None): 사분위범위(Q3 - Q1).
        skewness (float | None): 왜도.
        kurtosis (float | None): 첨도.
        cv (float | None): 변동계수(std/mean).
        sum (float | None): 합계.
    """

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
    """반입 직후 제공되는 자동 프로파일링 결과.

    Attributes:
        dataset_id (str): 대상 데이터셋 식별자.
        n_rows (int): 전체 행 수.
        n_cols (int): 전체 열 수.
        n_duplicated_rows (int): 중복 행 수.
        total_missing_cells (int): 전체 결측 셀 수.
        missing_ratio (float): 전체 결측 비율.
        numeric (list[NumericSummary]): 수치형 변수별 기초 통계량.
        categorical (list[CategoricalSummary]): 범주형 변수별 요약.
        missing (list[MissingReport]): 변수별 결측 현황.
        outliers (list[OutlierReport]): 변수별 이상치 탐지 결과.
        warnings (list[str]): 프로파일링 과정에서 발견된 경고 메시지.
    """

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
    """프로파일 + 상관관계 + 차트 추천/데이터를 묶은 탐색 리포트.

    Attributes:
        dataset (DatasetDetail): 대상 데이터셋 상세 정보.
        profile (DatasetProfile): 기초 통계·결측·이상치 프로파일링 결과.
        correlation (CorrelationResult | None): 상관관계 분석 결과. 변수가 2개 미만이면 None.
        chart_recommendations (list[ChartRecommendation]): 추천 차트 목록.
        chart_data (dict[str, Any]): 추천 차트를 그리는 데 필요한 집계 데이터.
    """

    dataset: DatasetDetail
    profile: DatasetProfile
    correlation: CorrelationResult | None = None
    chart_recommendations: list[ChartRecommendation] = Field(default_factory=list)
    chart_data: dict[str, Any] = Field(default_factory=dict)
