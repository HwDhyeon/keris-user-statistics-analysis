"""분석 요청/응답 스키마."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.models.enums import (
    AnalysisMethod,
    AnalysisStatus,
    ChartKind,
    EncodeStrategy,
    ImputeStrategy,
    OutlierMethod,
    ScaleStrategy,
)


class FilterCondition(BaseModel):
    column: str
    op: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "between", "notna", "isna"]
    value: Any = None


class ColumnImputation(BaseModel):
    column: str
    strategy: ImputeStrategy
    fill_value: Any = None


class PreprocessingSpec(BaseModel):
    """분석 전 적용할 전처리 로직. 패키지에 그대로 보존되어 재현에 사용된다.

    Attributes:
        filters (list[FilterCondition]): 행 필터링 조건 목록.
        drop_duplicates (bool): 중복 행 제거 여부.
        missing_strategy (ImputeStrategy): 결측치 처리 전략.
        missing_fill_value (Any): missing_strategy가 CONSTANT일 때 채울 값.
        column_imputations (list[ColumnImputation]): 열별 개별 결측치 처리 지정 목록.
        remove_outliers (bool): 이상치 제거 여부.
        outlier_columns (list[str]): 이상치 탐지 대상 열 목록.
        outlier_method (OutlierMethod): 이상치 탐지 방법.
        outlier_threshold (float): 이상치 판정 임계값(IQR은 사분위범위 배수, Z-score·Modified Z-score는 절단 임계값).
        scaling (ScaleStrategy): 스케일링 전략.
        encoding (EncodeStrategy): 범주형 인코딩 방식.
        category_orders (dict[str, list[Any]]): 변수별로 유지할 범주 순서(예: {"학년": ["1학년", "2학년", "3학년"]}).
        drop_first (bool): 원-핫 인코딩 시 첫 범주를 기준(reference)으로 제거할지 여부.
    """

    filters: list[FilterCondition] = Field(default_factory=list)
    drop_duplicates: bool = False
    missing_strategy: ImputeStrategy = ImputeStrategy.DROP_ROWS
    missing_fill_value: Any = None
    column_imputations: list[ColumnImputation] = Field(default_factory=list)
    remove_outliers: bool = False
    outlier_columns: list[str] = Field(default_factory=list)
    outlier_method: OutlierMethod = OutlierMethod.IQR
    # IQR: 사분위범위 배수 / Z-score·Modified Z-score: 절단 임계값
    outlier_threshold: float = Field(default=1.5, gt=0)
    scaling: ScaleStrategy = ScaleStrategy.NONE
    encoding: EncodeStrategy = EncodeStrategy.ONEHOT
    # 변수별 범주 순서 유지: {"학년": ["1학년", "2학년", "3학년"]}
    category_orders: dict[str, list[Any]] = Field(default_factory=dict)
    # 원-핫 인코딩 시 첫 범주를 기준(reference)으로 제거할지 여부
    drop_first: bool = True


class AnalysisParams(BaseModel):
    """분석 실행 파라미터. 기법별로 사용하는 항목이 다르다.

    Attributes:
        target (str | None): 종속변수.
        features (list[str]): 독립변수(다중 선택).
        weight_column (str | None): 가중치 변수.
        test_size (float): 검증용으로 분리할 비율. 0이면 홀드아웃 없이 전체 적합.
        random_state (int): 재현성을 위한 난수 시드.
        fit_intercept (bool): 회귀 절편 포함 여부.
        include_diagnostics (bool): 회귀 진단(잔차 등) 포함 여부.
        positive_label (Any): 로지스틱 회귀에서 양성 클래스로 취급할 값.
        max_iter (int): 로지스틱 회귀 등 반복 최적화의 최대 반복 횟수.
        max_depth (int | None): 의사결정나무 최대 깊이.
        min_samples_split (int): 의사결정나무 노드 분할에 필요한 최소 표본 수.
        min_samples_leaf (int): 의사결정나무 리프 노드의 최소 표본 수.
        criterion (str | None): 의사결정나무 분할 기준.
        ccp_alpha (float): 의사결정나무 비용복잡도 가지치기 파라미터.
        n_clusters (int): 군집 개수(KMeans·계층적 군집).
        linkage (Literal["ward", "complete", "average", "single"]): 계층적 군집의 연결 방식.
        eps (float): DBSCAN 반경(eps).
        min_samples (int): DBSCAN 핵심 포인트 판정에 필요한 최소 이웃 수.
        auto_k_range (list[int] | None): [최소, 최대] 지정 시 엘보/실루엣 기반 군집 수 자동 탐색.
        label_columns (list[str]): 원시데이터·잔차표에 표시할 식별자 변수(예: 학교코드, 지역구분). 군집분석에는 사용되지 않음.
        n_components (int): 차원축소(PCA) 시 산출할 주성분 개수.
        correlation_method (Literal["pearson", "spearman", "kendall"]): 상관분석 방법.
        factors (list[str]): ANOVA 요인(범주형 독립변수).
        covariates (list[str]): ANCOVA 공변량(연속형).
        include_interaction (bool): 요인이 2개일 때 교호작용항 포함 여부.
        group_column (str | None): 위계 집단(학급·학교 등) 또는 패널 개체 식별 변수.
        time_column (str | None): 성장모형의 시점(시간) 변수.
        random_slope (bool): 시간(또는 주 예측변수)에 대한 확률기울기 포함 여부.
        treatment_column (str | None): 처치/프로그램 참여 여부 변수(이분형).
        caliper (float | None): 매칭 허용 거리(표준화 성향점수 기준). 미지정 시 자동 산출.
        n_neighbors (int): 처치군 1건당 매칭할 대조군 수.
        matching_replacement (bool): 대조군 중복 매칭 허용 여부.
    """

    # 공통
    target: str | None = Field(default=None, description="종속변수")
    features: list[str] = Field(default_factory=list, description="독립변수(다중 선택)")
    weight_column: str | None = Field(default=None, description="가중치 변수")
    test_size: float = Field(default=0.0, ge=0.0, lt=0.9, description="0이면 홀드아웃 없이 전체 적합")
    random_state: int = 42

    # 회귀
    fit_intercept: bool = True
    include_diagnostics: bool = True

    # 로지스틱
    positive_label: Any = None
    max_iter: int = Field(default=200, ge=10, le=10000)

    # 의사결정나무
    max_depth: int | None = Field(default=None, ge=1, le=50)
    min_samples_split: int = Field(default=2, ge=2)
    min_samples_leaf: int = Field(default=1, ge=1)
    criterion: str | None = None
    ccp_alpha: float = Field(default=0.0, ge=0.0)

    # 군집
    n_clusters: int = Field(default=3, ge=2, le=50)
    linkage: Literal["ward", "complete", "average", "single"] = "ward"
    eps: float = Field(default=0.5, gt=0)
    min_samples: int = Field(default=5, ge=1)
    auto_k_range: list[int] | None = Field(default=None, description="[최소, 최대] 지정 시 엘보/실루엣 탐색")
    label_columns: list[str] = Field(
        default_factory=list,
        description="원시데이터·잔차표에 표시할 식별자 변수(예: 학교코드, 지역구분). 군집분석에는 사용되지 않음",
    )

    # 차원축소
    n_components: int = Field(default=2, ge=1, le=50)

    # 상관분석
    correlation_method: Literal["pearson", "spearman", "kendall"] = "pearson"

    # 분산분석(ANOVA) / 공분산분석(ANCOVA)
    factors: list[str] = Field(default_factory=list, description="ANOVA 요인(범주형 독립변수)")
    covariates: list[str] = Field(default_factory=list, description="ANCOVA 공변량(연속형)")
    include_interaction: bool = Field(default=True, description="요인이 2개면 교호작용항 포함 여부")

    # 다층모형(HLM) / 성장모형
    group_column: str | None = Field(default=None, description="위계 집단(학급·학교 등) 또는 패널 개체 식별 변수")
    time_column: str | None = Field(default=None, description="성장모형의 시점(시간) 변수")
    random_slope: bool = Field(default=False, description="시간(또는 주 예측변수)에 대한 확률기울기 포함 여부")

    # 성향점수매칭(PSM)
    treatment_column: str | None = Field(default=None, description="처치/프로그램 참여 여부 변수(이분형)")
    caliper: float | None = Field(
        default=None, gt=0, description="매칭 허용 거리(표준화 성향점수 기준). 미지정 시 자동 산출"
    )
    n_neighbors: int = Field(default=1, ge=1, le=10, description="처치군 1건당 매칭할 대조군 수")
    matching_replacement: bool = Field(default=False, description="대조군 중복 매칭 허용 여부")

    model_config = {"extra": "allow"}


class ChartDataOptions(BaseModel):
    """차트용 데이터 산출 옵션.

    렌더링은 웹에서 수행하므로 백엔드는 그릴 재료(집계된 데이터)만 만들어 준다.

    Attributes:
        include (bool): false면 추천만 제공하고 데이터는 생략.
        max_points (int): 산점도 등 원시 점 개수 상한.
        histogram_bins (int): 히스토그램 구간 수.
        max_categories (int): 범주형 빈도 상위 N개.
    """

    include: bool = Field(default=True, description="false면 추천만 제공하고 데이터는 생략")
    max_points: int = Field(default=5000, ge=100, le=100000, description="산점도 등 원시 점 개수 상한")
    histogram_bins: int = Field(default=30, ge=5, le=100)
    max_categories: int = Field(default=20, ge=2, le=100, description="범주형 빈도 상위 N개")


class AnalysisCreate(BaseModel):
    dataset_id: str
    method: AnalysisMethod
    name: str | None = None
    params: AnalysisParams = Field(default_factory=AnalysisParams)
    preprocessing: PreprocessingSpec = Field(default_factory=PreprocessingSpec)
    chart_data: ChartDataOptions = Field(default_factory=ChartDataOptions)

    @model_validator(mode="after")
    def _check_required(self) -> AnalysisCreate:
        supervised = {
            AnalysisMethod.LINEAR_REGRESSION,
            AnalysisMethod.LOGISTIC_REGRESSION,
            AnalysisMethod.DECISION_TREE_REGRESSOR,
            AnalysisMethod.DECISION_TREE_CLASSIFIER,
        }
        if self.method in supervised:
            if not self.params.target:
                raise ValueError(f"{self.method} 분석에는 target(종속변수)이 필요합니다.")
            if not self.params.features:
                raise ValueError(f"{self.method} 분석에는 features(독립변수)가 1개 이상 필요합니다.")
        unsupervised = {
            AnalysisMethod.KMEANS,
            AnalysisMethod.HIERARCHICAL,
            AnalysisMethod.DBSCAN,
            AnalysisMethod.PCA,
        }
        if self.method in unsupervised and not self.params.features:
            raise ValueError(f"{self.method} 분석에는 features가 1개 이상 필요합니다.")

        if self.method is AnalysisMethod.ANOVA:
            if not self.params.target:
                raise ValueError("anova 분석에는 target(종속변수)이 필요합니다.")
            if not self.params.factors:
                raise ValueError("anova 분석에는 factors(범주형 요인)가 1개 이상 필요합니다.")

        if self.method is AnalysisMethod.MULTILEVEL:
            if not self.params.target:
                raise ValueError("multilevel 분석에는 target(종속변수)이 필요합니다.")
            if not self.params.group_column:
                raise ValueError("multilevel 분석에는 group_column(위계 집단 변수)이 필요합니다.")

        if self.method is AnalysisMethod.GROWTH_CURVE:
            if not self.params.target:
                raise ValueError("growth_curve 분석에는 target(종속변수)이 필요합니다.")
            if not self.params.time_column:
                raise ValueError("growth_curve 분석에는 time_column(시점 변수)이 필요합니다.")
            if not self.params.group_column:
                raise ValueError("growth_curve 분석에는 group_column(패널 개체 식별 변수)이 필요합니다.")

        if self.method is AnalysisMethod.PSM:
            if not self.params.treatment_column:
                raise ValueError("psm 분석에는 treatment_column(처치 여부 변수)이 필요합니다.")
            if not self.params.features:
                raise ValueError("psm 분석에는 features(성향점수 산출용 공변량)가 1개 이상 필요합니다.")

        return self


class ChartRecommendation(BaseModel):
    """어떤 차트를 그리면 좋은지에 대한 제안. 렌더링은 클라이언트가 수행한다.

    Attributes:
        kind (ChartKind): 추천 차트 종류.
        title (str): 차트 제목.
        reason (str): 이 차트를 추천하는 이유.
        priority (int): 추천 우선순위(작을수록 우선).
        data_key (str | None): chart_data 내 이 차트가 사용할 데이터 키.
        encoding (dict[str, str]): 축·색 등에 대응하는 데이터 필드명(x, y, color, label 등).
        options (dict[str, Any]): 참조선·정렬 등 렌더링 힌트.
    """

    kind: ChartKind
    title: str
    reason: str
    priority: int
    data_key: str | None = Field(
        default=None, description="chart_data 내 이 차트가 사용할 데이터 키"
    )
    encoding: dict[str, str] = Field(
        default_factory=dict, description="축·색 등에 대응하는 데이터 필드명 (x, y, color, label …)"
    )
    options: dict[str, Any] = Field(
        default_factory=dict, description="참조선·정렬 등 렌더링 힌트"
    )


class AnalysisRead(BaseModel):
    id: str
    dataset_id: str
    name: str
    method: AnalysisMethod
    status: AnalysisStatus
    params: dict[str, Any]
    preprocessing: dict[str, Any]
    metrics: dict[str, Any]
    chart_recommendations: list[ChartRecommendation] = Field(default_factory=list)
    error: str | None = None
    duration_ms: int | None = None
    created_at: datetime
    completed_at: datetime | None = None


class AnalysisDetail(AnalysisRead):
    result: dict[str, Any] = Field(default_factory=dict)
    chart_data: dict[str, Any] = Field(
        default_factory=dict, description="추천 차트를 그리는 데 필요한 집계 데이터"
    )
