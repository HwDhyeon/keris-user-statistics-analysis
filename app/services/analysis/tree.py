"""의사결정나무 분석 (분류/회귀)."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

from app.core.exceptions import AnalysisError
from app.models.enums import AnalysisMethod, ChartKind
from app.schemas.analysis import AnalysisParams, PreprocessingSpec
from app.services.analysis.base import AnalysisOutcome, jsonable, num, rec, split_train_test
from app.services.preprocessing import build_design_matrix


def tree_to_dict(tree: Any, feature_names: list[str], class_labels: list[str] | None) -> dict[str, Any]:
    """sklearn 트리를 중첩 딕셔너리(트리 구조도용)로 변환한다.

    Args:
        tree (Any): 학습된 sklearn 의사결정나무 모델(DecisionTreeClassifier/Regressor).
        feature_names (list[str]): 분기 조건에 사용할 변수 이름 목록.
        class_labels (list[str] | None): 분류 모델의 클래스 라벨 목록. 회귀 모델이면 None.

    Returns:
        dict[str, Any]: 루트 노드부터 재귀적으로 children을 포함하는 트리 구조 딕셔너리.
    """
    t = tree.tree_

    def build(node: int, depth: int) -> dict[str, Any]:
        n_samples = int(t.n_node_samples[node])
        impurity = float(t.impurity[node])
        payload: dict[str, Any] = {
            "node_id": int(node),
            "depth": depth,
            "n_samples": n_samples,
            "impurity": round(impurity, 6),
        }
        values = t.value[node][0]
        if class_labels:
            counts = np.asarray(values, dtype=float)
            total = counts.sum() or 1.0
            payload["class_distribution"] = {
                label: round(float(c), 6) for label, c in zip(class_labels, counts, strict=False)
            }
            payload["predicted_class"] = class_labels[int(np.argmax(counts))]
            payload["confidence"] = round(float(counts.max() / total), 6)
        else:
            payload["predicted_value"] = round(float(values[0]), 6)

        if t.children_left[node] == t.children_right[node]:  # 리프
            payload["is_leaf"] = True
            return payload

        feature = feature_names[int(t.feature[node])]
        threshold = float(t.threshold[node])
        payload |= {
            "is_leaf": False,
            "split_feature": feature,
            "threshold": round(threshold, 6),
            "condition": f"{feature} <= {threshold:.4f}",
            "children": [
                build(int(t.children_left[node]), depth + 1),
                build(int(t.children_right[node]), depth + 1),
            ],
        }
        return payload

    return build(0, 0)


def _rules(tree: Any, feature_names: list[str], class_labels: list[str] | None, limit: int = 30) -> list[str]:
    text = export_text(tree, feature_names=feature_names, max_depth=10)
    lines = [line for line in text.splitlines() if line.strip()]
    if class_labels:
        for i, label in enumerate(class_labels):
            lines = [line.replace(f"class: {i}", f"class: {label}") for line in lines]
    return lines[:limit]


class _BaseTreeEngine:
    classifier: bool

    def run(self, frame: pd.DataFrame, params: AnalysisParams, spec: PreprocessingSpec) -> AnalysisOutcome:
        design = build_design_matrix(
            frame,
            features=params.features,
            target=params.target,
            weight_column=params.weight_column,
            spec=spec,
            target_as_category=self.classifier,
        )
        X, y, w = design.X, design.y, design.weights
        if y is None:
            raise AnalysisError("종속변수가 필요합니다.")

        classes = design.target_classes if self.classifier else None
        if self.classifier and (not classes or len(classes) < 2):
            msg = f"종속변수 '{params.target}'의 범주가 2개 이상이어야 합니다."
            raise AnalysisError(msg)

        X_tr, X_te, y_tr, y_te, w_tr, w_te = split_train_test(
            X, y, w, params.test_size, params.random_state, stratify=self.classifier
        )

        criterion = params.criterion or ("gini" if self.classifier else "squared_error")
        model_cls = DecisionTreeClassifier if self.classifier else DecisionTreeRegressor
        model = model_cls(
            criterion=criterion,
            max_depth=params.max_depth,
            min_samples_split=params.min_samples_split,
            min_samples_leaf=params.min_samples_leaf,
            ccp_alpha=params.ccp_alpha,
            random_state=params.random_state,
        )
        model.fit(X_tr, y_tr, sample_weight=w_tr.to_numpy() if w_tr is not None else None)

        pred = model.predict(X_te)
        sw = w_te.to_numpy() if w_te is not None else None

        importance = sorted(
            (
                {
                    "feature": str(name),
                    "source_variable": design.feature_origin.get(str(name), str(name)),
                    "importance": num(value),
                }
                for name, value in zip(design.feature_names, model.feature_importances_, strict=True)
            ),
            key=lambda d: d["importance"] or 0.0,
            reverse=True,
        )

        if self.classifier:
            metrics = self._classification_metrics(y_te, pred, model, X_te, sw, len(classes or []))
            cm = confusion_matrix(y_te, pred, labels=list(range(len(classes or []))))
            extra: dict[str, Any] = {
                "classes": classes,
                "confusion_matrix": {"labels": classes, "matrix": cm.tolist()},
            }
        else:
            mse = mean_squared_error(y_te, pred, sample_weight=sw)
            metrics = {
                "r_squared": num(r2_score(y_te, pred, sample_weight=sw)),
                "rmse": num(np.sqrt(mse)),
                "mse": num(mse),
                "mae": num(mean_absolute_error(y_te, pred, sample_weight=sw)),
                "train_r_squared": num(model.score(X_tr, y_tr)),
                "evaluation": "holdout" if params.test_size > 0 else "in_sample",
            }
            extra = {}

        structure = tree_to_dict(model, design.feature_names, classes)
        result: dict[str, Any] = {
            "method": "의사결정나무 (분류)" if self.classifier else "의사결정나무 (회귀)",
            "target": params.target,
            "features": params.features,
            "criterion": criterion,
            "n_observations": len(X),
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "weighted": w is not None,
            "tree_depth": int(model.get_depth()),
            "n_leaves": int(model.get_n_leaves()),
            "feature_importance": importance,
            "tree": structure,
            "rules": _rules(model, design.feature_names, classes),
            "preprocessing_steps": design.steps,
            **extra,
        }

        artifacts: dict[str, Any] = {"tree": structure, "importance": importance, "design": design}
        if self.classifier:
            artifacts["confusion_matrix"] = extra["confusion_matrix"]
        else:
            artifacts["predictions"] = pd.DataFrame(
                {
                    "actual": np.asarray(y_te, dtype=float),
                    "predicted": np.asarray(pred, dtype=float),
                    "residual": np.asarray(y_te, dtype=float) - np.asarray(pred, dtype=float),
                }
            )

        recommendations = [
            rec(
                ChartKind.TREE_DIAGRAM,
                "트리 구조도",
                "분기 규칙과 노드별 예측을 직관적으로 확인",
                1,
                data_key="tree",
                encoding={"label": "label", "size": "n_samples", "color": "predicted"},
                options={"layout": "hierarchy", "node_list": "nodes", "link_list": "links"},
            ),
            rec(
                ChartKind.FEATURE_IMPORTANCE,
                "변수 중요도",
                "예측에 기여한 변수 순위 확인",
                2,
                data_key="feature_importance",
                encoding={"x": "importance", "y": "feature"},
                options={"sort": "-x", "format": "percent"},
            ),
        ]
        recommendations.append(
            rec(
                ChartKind.CONFUSION_MATRIX,
                "혼동행렬",
                "분류 오류 패턴 확인",
                3,
                data_key="confusion_matrix",
                encoding={"x": "predicted", "y": "actual", "color": "count", "label": "count"},
            )
            if self.classifier
            else rec(
                ChartKind.SCATTER_WITH_FIT,
                "실제값 대 예측값",
                "예측 정확도 시각 확인",
                3,
                data_key="predictions",
                encoding={"x": "actual", "y": "predicted"},
                options={"identity_line": True},
            )
        )
        return AnalysisOutcome(jsonable(result), jsonable(metrics), artifacts, recommendations)

    @staticmethod
    def _classification_metrics(
        y_te: pd.Series,
        pred: np.ndarray,
        model: Any,
        X_te: pd.DataFrame,
        sw: np.ndarray | None,
        n_classes: int,
    ) -> dict[str, Any]:
        binary = n_classes == 2
        average = "binary" if binary else "macro"
        metrics = {
            "accuracy": num(accuracy_score(y_te, pred, sample_weight=sw)),
            "precision": num(precision_score(y_te, pred, average=average, zero_division=0, sample_weight=sw)),
            "recall": num(recall_score(y_te, pred, average=average, zero_division=0, sample_weight=sw)),
            "f1": num(f1_score(y_te, pred, average=average, zero_division=0, sample_weight=sw)),
        }
        try:
            proba = model.predict_proba(X_te)
            metrics["roc_auc"] = num(
                roc_auc_score(y_te, proba[:, 1], sample_weight=sw)
                if binary
                else roc_auc_score(y_te, proba, multi_class="ovr", average="macro", sample_weight=sw)
            )
        except ValueError, IndexError:
            metrics["roc_auc"] = None
        return metrics


class DecisionTreeClassifierEngine(_BaseTreeEngine):
    method = AnalysisMethod.DECISION_TREE_CLASSIFIER
    classifier = True


class DecisionTreeRegressorEngine(_BaseTreeEngine):
    method = AnalysisMethod.DECISION_TREE_REGRESSOR
    classifier = False
