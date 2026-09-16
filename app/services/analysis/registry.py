"""분석 기법 레지스트리."""

from app.core.exceptions import AnalysisError
from app.models.enums import AnalysisMethod
from app.services.analysis.anova import AnovaEngine
from app.services.analysis.base import Engine
from app.services.analysis.cluster import DBSCANEngine, HierarchicalEngine, KMeansEngine, PCAEngine
from app.services.analysis.descriptive import CorrelationEngine, DescriptiveEngine
from app.services.analysis.multilevel import GrowthCurveEngine, MultilevelEngine
from app.services.analysis.psm import PSMEngine
from app.services.analysis.regression import LinearRegressionEngine, LogisticRegressionEngine
from app.services.analysis.tree import DecisionTreeClassifierEngine, DecisionTreeRegressorEngine

ENGINES: dict[AnalysisMethod, Engine] = {
    AnalysisMethod.DESCRIPTIVE: DescriptiveEngine(),
    AnalysisMethod.CORRELATION: CorrelationEngine(),
    AnalysisMethod.LINEAR_REGRESSION: LinearRegressionEngine(),
    AnalysisMethod.LOGISTIC_REGRESSION: LogisticRegressionEngine(),
    AnalysisMethod.DECISION_TREE_REGRESSOR: DecisionTreeRegressorEngine(),
    AnalysisMethod.DECISION_TREE_CLASSIFIER: DecisionTreeClassifierEngine(),
    AnalysisMethod.KMEANS: KMeansEngine(),
    AnalysisMethod.HIERARCHICAL: HierarchicalEngine(),
    AnalysisMethod.DBSCAN: DBSCANEngine(),
    AnalysisMethod.PCA: PCAEngine(),
    AnalysisMethod.ANOVA: AnovaEngine(),
    AnalysisMethod.MULTILEVEL: MultilevelEngine(),
    AnalysisMethod.GROWTH_CURVE: GrowthCurveEngine(),
    AnalysisMethod.PSM: PSMEngine(),
}

METHOD_INFO: dict[AnalysisMethod, dict] = {
    AnalysisMethod.DESCRIPTIVE: {
        "label": "기술통계", "category": "기초",
        "description": "평균·중앙값·사분위수 등 기초 통계량과 결측·이상치 요약",
        "requires_target": False, "key_metrics": ["missing_ratio", "n_duplicated_rows"],
    },
    AnalysisMethod.CORRELATION: {
        "label": "상관분석", "category": "기초",
        "description": "피어슨/스피어만/켄달 상관계수와 유의확률",
        "requires_target": False, "key_metrics": ["max_abs_coefficient", "n_strong_pairs"],
    },
    AnalysisMethod.LINEAR_REGRESSION: {
        "label": "선형 회귀분석", "category": "예측",
        "description": "연속형 종속변수 예측. 가중치(WLS), 다중공선성·정규성 진단 지원",
        "requires_target": True, "key_metrics": ["r_squared", "adjusted_r_squared", "rmse", "mae"],
    },
    AnalysisMethod.LOGISTIC_REGRESSION: {
        "label": "로지스틱 회귀분석", "category": "예측",
        "description": "범주형 종속변수 예측. 오즈비와 유의확률 산출",
        "requires_target": True, "key_metrics": ["accuracy", "roc_auc", "f1", "mcfadden"],
    },
    AnalysisMethod.DECISION_TREE_REGRESSOR: {
        "label": "의사결정나무 (회귀)", "category": "예측",
        "description": "규칙 기반 연속값 예측. 트리 구조도와 변수 중요도 제공",
        "requires_target": True, "key_metrics": ["r_squared", "rmse"],
    },
    AnalysisMethod.DECISION_TREE_CLASSIFIER: {
        "label": "의사결정나무 (분류)", "category": "예측",
        "description": "규칙 기반 분류. 트리 구조도와 변수 중요도 제공",
        "requires_target": True, "key_metrics": ["accuracy", "f1", "roc_auc"],
    },
    AnalysisMethod.KMEANS: {
        "label": "K-평균 군집분석", "category": "군집",
        "description": "거리 기반 군집. 엘보·실루엣으로 적정 군집 수 자동 탐색",
        "requires_target": False, "key_metrics": ["silhouette", "inertia", "davies_bouldin"],
    },
    AnalysisMethod.HIERARCHICAL: {
        "label": "계층적 군집분석", "category": "군집",
        "description": "병합형 군집과 덴드로그램",
        "requires_target": False, "key_metrics": ["silhouette", "calinski_harabasz"],
    },
    AnalysisMethod.DBSCAN: {
        "label": "DBSCAN 군집분석", "category": "군집",
        "description": "밀도 기반 군집. 잡음점(이상치) 자동 분리",
        "requires_target": False, "key_metrics": ["n_clusters", "noise_ratio", "silhouette"],
    },
    AnalysisMethod.PCA: {
        "label": "주성분분석", "category": "차원축소",
        "description": "변수 축약과 설명분산 비율 산출",
        "requires_target": False, "key_metrics": ["total_explained_variance"],
    },
    AnalysisMethod.ANOVA: {
        "label": "분산분석 (ANOVA/ANCOVA)", "category": "집단비교",
        "description": "지역별·학교유형별 등 범주형 요인 간 평균 비교. 공변량 추가 시 ANCOVA",
        "requires_target": True, "key_metrics": ["model_r_squared", "n_significant_terms", "levene_p_value"],
    },
    AnalysisMethod.MULTILEVEL: {
        "label": "다층모형 (HLM)", "category": "위계모형",
        "description": "학생 within 학급 within 학교 위계구조에서 집단 간·집단 내 분산을 분리",
        "requires_target": True, "key_metrics": ["icc", "unconditional_icc", "aic", "bic"],
    },
    AnalysisMethod.GROWTH_CURVE: {
        "label": "성장모형 (Growth Curve, 확률효과)", "category": "위계모형",
        "description": "패널 데이터에서 개체별 시간에 따른 변화 궤적과 평균 성장률 추정",
        "requires_target": True, "key_metrics": ["average_growth_rate", "average_growth_rate_p_value", "aic"],
    },
    AnalysisMethod.PSM: {
        "label": "성향점수매칭 (PSM)", "category": "인과추론",
        "description": "정책/프로그램 참여 여부에 대한 성향점수 산출 후 매칭으로 처치효과(ATT) 추정",
        "requires_target": False, "key_metrics": ["match_rate", "mean_abs_smd_after", "att"],
    },
}


def get_engine(method: AnalysisMethod) -> Engine:
    engine = ENGINES.get(method)
    if engine is None:  # pragma: no cover - Enum으로 제한됨
        msg = f"지원하지 않는 분석 기법입니다: {method}"
        raise AnalysisError(msg)
    return engine
