"""도메인 열거형."""

from enum import StrEnum


class DatasetStatus(StrEnum):
    READY = "ready"
    FAILED = "failed"


class ColumnRole(StrEnum):
    """프로파일링으로 추론한 변수 역할."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    CONSTANT = "constant"
    IDENTIFIER = "identifier"


class AnalysisMethod(StrEnum):
    DESCRIPTIVE = "descriptive"
    CORRELATION = "correlation"
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE_REGRESSOR = "decision_tree_regressor"
    DECISION_TREE_CLASSIFIER = "decision_tree_classifier"
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"
    PCA = "pca"
    ANOVA = "anova"
    MULTILEVEL = "multilevel"
    GROWTH_CURVE = "growth_curve"
    PSM = "psm"


class AnalysisStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class OutlierMethod(StrEnum):
    IQR = "iqr"
    ZSCORE = "zscore"
    MODIFIED_ZSCORE = "modified_zscore"
    ISOLATION_FOREST = "isolation_forest"


class CorrelationMethod(StrEnum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class ChartKind(StrEnum):
    HISTOGRAM = "histogram"
    BOXPLOT = "boxplot"
    BAR = "bar"
    SCATTER = "scatter"
    SCATTER_WITH_FIT = "scatter_with_fit"
    HEATMAP = "heatmap"
    LINE = "line"
    RESIDUAL = "residual"
    ROC_CURVE = "roc_curve"
    CONFUSION_MATRIX = "confusion_matrix"
    TREE_DIAGRAM = "tree_diagram"
    FEATURE_IMPORTANCE = "feature_importance"
    CLUSTER_SCATTER = "cluster_scatter"
    ELBOW = "elbow"
    DENDROGRAM = "dendrogram"
    SCREE = "scree"
    MISSING_MATRIX = "missing_matrix"
    MEANS_PLOT = "means_plot"
    RANDOM_EFFECTS = "random_effects"
    VARIANCE_COMPONENTS = "variance_components"
    TRAJECTORY = "trajectory"
    PS_DISTRIBUTION = "ps_distribution"
    LOVE_PLOT = "love_plot"


class ImputeStrategy(StrEnum):
    NONE = "none"
    DROP_ROWS = "drop_rows"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    FORWARD_FILL = "ffill"


class ScaleStrategy(StrEnum):
    NONE = "none"
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"


class EncodeStrategy(StrEnum):
    """범주형 인코딩. ORDINAL은 지정한 범주 순서를 그대로 유지한다."""

    ONEHOT = "onehot"
    ORDINAL = "ordinal"
