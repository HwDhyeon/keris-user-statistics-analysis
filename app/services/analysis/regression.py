"""선형/로지스틱 회귀분석."""

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from app.core.exceptions import AnalysisError
from app.models.enums import AnalysisMethod, ChartKind
from app.schemas.analysis import AnalysisParams, PreprocessingSpec
from app.services.analysis.base import AnalysisOutcome, jsonable, num, rec, split_train_test
from app.services.preprocessing import build_design_matrix


class LinearRegressionEngine:
    """OLS / WLS 선형회귀. 결정계수, RMSE 등 전문 지표를 함께 산출한다."""

    method = AnalysisMethod.LINEAR_REGRESSION

    def run(self, frame: pd.DataFrame, params: AnalysisParams, spec: PreprocessingSpec) -> AnalysisOutcome:
        design = build_design_matrix(
            frame,
            features=params.features,
            target=params.target,
            weight_column=params.weight_column,
            spec=spec,
        )
        X, y, w = design.X, design.y, design.weights
        if y is None:
            raise AnalysisError("종속변수가 필요합니다.")

        X_tr, X_te, y_tr, y_te, w_tr, w_te = split_train_test(
            X,
            y,
            w,
            params.test_size,
            params.random_state,
        )

        exog = sm.add_constant(X_tr, has_constant="add") if params.fit_intercept else X_tr
        model = sm.WLS(y_tr, exog, weights=w_tr.to_numpy()) if w_tr is not None else sm.OLS(y_tr, exog)
        fit = model.fit()

        exog_te = sm.add_constant(X_te, has_constant="add") if params.fit_intercept else X_te
        pred_tr = np.asarray(fit.predict(exog))
        pred_te = np.asarray(fit.predict(exog_te))

        coefficients = self._coefficients(fit, design.feature_origin)
        metrics = self._metrics(y_tr, pred_tr, y_te, pred_te, w_te, fit, params)

        result: dict[str, Any] = {
            "method": "선형 회귀분석",
            "target": params.target,
            "features": params.features,
            "n_observations": len(X),
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "weighted": w is not None,
            "coefficients": coefficients,
            "equation": self._equation(coefficients, params.target or "y"),
            "anova": {
                "f_statistic": num(fit.fvalue),
                "f_pvalue": num(fit.f_pvalue),
                "df_model": num(fit.df_model),
                "df_residual": num(fit.df_resid),
            },
            "preprocessing_steps": design.steps,
        }

        if params.include_diagnostics:
            result["diagnostics"] = self._diagnostics(fit, X_tr, np.asarray(y_tr), pred_tr)

        residuals = np.asarray(y_te) - pred_te
        artifacts = {
            "predictions": pd.DataFrame(
                {"actual": np.asarray(y_te, dtype=float), "predicted": pred_te, "residual": residuals}
            ),
            "coefficients": coefficients,
            "design": design,
            "x_label": params.target,
        }

        recommendations = [
            rec(
                ChartKind.SCATTER_WITH_FIT,
                "회귀선이 포함된 산점도",
                "적합선과 실제 관측치의 부합도를 확인",
                1,
                data_key="predictions",
                encoding={"x": "actual", "y": "predicted"},
                options={"fit_line": True, "identity_line": True, "x_label": "실제값", "y_label": "예측값"},
            ),
            rec(
                ChartKind.RESIDUAL,
                "잔차 산점도",
                "등분산성과 선형성 가정 위배 여부 점검",
                2,
                data_key="predictions",
                encoding={"x": "predicted", "y": "residual"},
                options={"reference_line": 0, "x_label": "예측값", "y_label": "잔차"},
            ),
            rec(
                ChartKind.BAR,
                "회귀계수 비교",
                "독립변수별 영향력 크기와 유의성 비교",
                3,
                data_key="coefficients",
                encoding={"x": "value", "y": "term", "color": "significant"},
                options={"sort": "-x", "error_bars": ["ci_lower", "ci_upper"]},
            ),
            rec(
                ChartKind.HISTOGRAM,
                "잔차 분포",
                "잔차 정규성 확인",
                4,
                data_key="residual_histogram",
                encoding={"x": "bin_center", "y": "count"},
            ),
        ]
        return AnalysisOutcome(jsonable(result), jsonable(metrics), artifacts, recommendations)

    @staticmethod
    def _coefficients(fit: Any, origin: dict[str, str]) -> list[dict[str, Any]]:
        conf = fit.conf_int()
        rows: list[dict[str, Any]] = []
        for name in fit.params.index:
            label = str(name)
            rows.append(
                {
                    "term": label,
                    "source_variable": origin.get(label, "(절편)" if label == "const" else label),
                    "coefficient": num(fit.params[name]),
                    "std_error": num(fit.bse[name]),
                    "t_value": num(fit.tvalues[name]),
                    "p_value": num(fit.pvalues[name]),
                    "ci_lower": num(conf.loc[name].iloc[0]),
                    "ci_upper": num(conf.loc[name].iloc[1]),
                    "significant": bool(num(fit.pvalues[name]) is not None and fit.pvalues[name] < 0.05),
                }
            )
        return rows

    @staticmethod
    def _metrics(
        y_tr: pd.Series,
        pred_tr: np.ndarray,
        y_te: pd.Series,
        pred_te: np.ndarray,
        w_te: pd.Series | None,
        fit: Any,
        params: AnalysisParams,
    ) -> dict[str, Any]:
        sw = w_te.to_numpy() if w_te is not None else None
        mse = mean_squared_error(y_te, pred_te, sample_weight=sw)
        try:
            mape = num(mean_absolute_percentage_error(y_te, pred_te, sample_weight=sw))
        except ValueError:
            mape = None
        return {
            "r_squared": num(fit.rsquared),
            "adjusted_r_squared": num(fit.rsquared_adj),
            "holdout_r_squared": num(r2_score(y_te, pred_te, sample_weight=sw)),
            "rmse": num(np.sqrt(mse)),
            "mse": num(mse),
            "mae": num(mean_absolute_error(y_te, pred_te, sample_weight=sw)),
            "mape": mape,
            "aic": num(fit.aic),
            "bic": num(fit.bic),
            "log_likelihood": num(fit.llf),
            "f_statistic": num(fit.fvalue),
            "f_pvalue": num(fit.f_pvalue),
            "durbin_watson": num(_durbin_watson(np.asarray(y_tr) - pred_tr)),
            "evaluation": "holdout" if params.test_size > 0 else "in_sample",
        }

    @staticmethod
    def _diagnostics(fit: Any, X: pd.DataFrame, y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
        from scipy import stats as sps
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        residuals = y - pred
        diagnostics: dict[str, Any] = {}

        # 다중공선성 (VIF)
        if X.shape[1] >= 2:
            exog = sm.add_constant(X, has_constant="add").to_numpy(dtype=float)
            vifs = []
            for i, name in enumerate(["const", *X.columns], start=0):
                if name == "const":
                    continue
                try:
                    vifs.append({"term": str(name), "vif": num(variance_inflation_factor(exog, i))})
                except ValueError, ZeroDivisionError:
                    vifs.append({"term": str(name), "vif": None})
            diagnostics["vif"] = vifs
            high = [v["term"] for v in vifs if v["vif"] and v["vif"] > 10]
            if high:
                diagnostics["multicollinearity_warning"] = (
                    f"VIF가 10을 초과하는 변수: {', '.join(high)} — 다중공선성이 의심됩니다."
                )

        # 잔차 정규성
        if 8 <= len(residuals) <= 5000:
            stat, p = sps.shapiro(residuals)
            diagnostics["normality"] = {
                "test": "Shapiro-Wilk",
                "statistic": num(stat),
                "p_value": num(p),
                "normal": bool(p > 0.05),
            }

        # 등분산성
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan

            lm, lm_p, _f_stat, _f_p = het_breuschpagan(residuals, sm.add_constant(X, has_constant="add"))
            diagnostics["homoscedasticity"] = {
                "test": "Breusch-Pagan",
                "lm_statistic": num(lm),
                "p_value": num(lm_p),
                "homoscedastic": bool(lm_p > 0.05),
            }
        except ValueError, np.linalg.LinAlgError:
            pass

        return diagnostics

    @staticmethod
    def _equation(coefficients: list[dict[str, Any]], target: str) -> str:
        parts: list[str] = []
        intercept = ""
        for row in coefficients:
            value = row["coefficient"]
            if value is None:
                continue
            if row["term"] == "const":
                intercept = f"{value:.4f}"
                continue
            sign = "+" if value >= 0 else "-"
            parts.append(f"{sign} {abs(value):.4f}×{row['term']}")
        body = " ".join(parts)
        return f"{target} = {intercept} {body}".strip()


class LogisticRegressionEngine:
    """이항/다항 로지스틱 회귀분석."""

    method = AnalysisMethod.LOGISTIC_REGRESSION

    def run(self, frame: pd.DataFrame, params: AnalysisParams, spec: PreprocessingSpec) -> AnalysisOutcome:
        design = build_design_matrix(
            frame,
            features=params.features,
            target=params.target,
            weight_column=params.weight_column,
            spec=spec,
            target_as_category=True,
        )
        X, y, w = design.X, design.y, design.weights
        classes = design.target_classes or []
        if y is None or len(classes) < 2:
            msg = f"종속변수 '{params.target}'의 범주가 2개 이상이어야 합니다. (현재 {len(classes)}개)"
            raise AnalysisError(msg)

        # positive_label 지정 시 해당 범주를 1이 되도록 재배치한다.
        if params.positive_label is not None and len(classes) == 2:
            label = str(params.positive_label)
            if label not in classes:
                msg = f"positive_label '{label}'이 종속변수 범주에 없습니다: {classes}"
                raise AnalysisError(msg)
            if classes.index(label) == 0:
                y = 1 - y
                classes = [classes[1], classes[0]]

        binary = len(classes) == 2
        X_tr, X_te, y_tr, y_te, w_tr, w_te = split_train_test(
            X, y, w, params.test_size, params.random_state, stratify=True
        )

        coefficients: list[dict[str, Any]] = []
        pseudo_r2: dict[str, Any] = {}

        if binary:
            exog = sm.add_constant(X_tr, has_constant="add") if params.fit_intercept else X_tr
            try:
                sm_fit = sm.Logit(y_tr.astype(float), exog.astype(float)).fit(
                    disp=False, maxiter=params.max_iter, method="newton"
                )
            except Exception as exc:
                msg = f"로지스틱 회귀 적합에 실패했습니다(완전분리 또는 수렴 실패 가능): {exc}"
                raise AnalysisError(msg) from exc
            coefficients = self._coefficients(sm_fit, design.feature_origin)
            pseudo_r2 = {
                "mcfadden": num(sm_fit.prsquared),
                "log_likelihood": num(sm_fit.llf),
                "llr_pvalue": num(sm_fit.llr_pvalue),
                "aic": num(sm_fit.aic),
                "bic": num(sm_fit.bic),
            }

        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            max_iter=params.max_iter,
            fit_intercept=params.fit_intercept,
            random_state=params.random_state,
        )
        clf.fit(X_tr, y_tr, sample_weight=w_tr.to_numpy() if w_tr is not None else None)

        proba = clf.predict_proba(X_te)
        pred = clf.predict(X_te)
        metrics = self._metrics(y_te, pred, proba, w_te, binary) | pseudo_r2

        if not binary:
            coefficients = [
                {
                    "class": classes[i],
                    "terms": [
                        {"term": str(t), "coefficient": num(c), "odds_ratio": num(np.exp(c))}
                        for t, c in zip(X.columns, row, strict=True)
                    ],
                }
                for i, row in enumerate(clf.coef_)
            ]

        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_te, pred, labels=list(range(len(classes))))
        result: dict[str, Any] = {
            "method": "로지스틱 회귀분석",
            "target": params.target,
            "classes": classes,
            "features": params.features,
            "binary": binary,
            "n_observations": len(X),
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "weighted": w is not None,
            "coefficients": coefficients,
            "class_distribution": {classes[int(k)]: int(v) for k, v in y.value_counts().items()},
            "confusion_matrix": {"labels": classes, "matrix": cm.tolist()},
            "preprocessing_steps": design.steps,
        }

        artifacts: dict[str, Any] = {
            "confusion_matrix": {"labels": classes, "matrix": cm.tolist()},
            "coefficients": coefficients,
            "design": design,
        }
        if binary:
            fpr, tpr, _ = roc_curve(y_te, proba[:, 1])
            artifacts["roc"] = pd.DataFrame({"fpr": fpr, "tpr": tpr})
            artifacts["probabilities"] = pd.DataFrame({"actual": np.asarray(y_te), "probability": proba[:, 1]})

        recommendations = [
            rec(
                ChartKind.CONFUSION_MATRIX,
                "혼동행렬",
                "분류 정확도의 구조적 오류 확인",
                1,
                data_key="confusion_matrix",
                encoding={"x": "predicted", "y": "actual", "color": "count", "label": "count"},
            ),
        ]
        if binary:
            recommendations.append(
                rec(
                    ChartKind.ROC_CURVE,
                    "ROC 곡선",
                    "임계값 전반의 판별력(AUC) 확인",
                    2,
                    data_key="roc_curve",
                    encoding={"x": "fpr", "y": "tpr"},
                    options={"diagonal_reference": True, "x_label": "거짓 양성률", "y_label": "참 양성률"},
                )
            )
        recommendations.append(
            rec(
                ChartKind.BAR,
                "오즈비 비교",
                "변수별 승산 변화량 비교",
                3,
                data_key="coefficients",
                encoding={"x": "value", "y": "term", "color": "significant"},
                options={"sort": "-x", "reference_line": 1, "error_bars": ["ci_lower", "ci_upper"]},
            )
        )

        return AnalysisOutcome(jsonable(result), jsonable(metrics), artifacts, recommendations)

    @staticmethod
    def _coefficients(fit: Any, origin: dict[str, str]) -> list[dict[str, Any]]:
        conf = fit.conf_int()
        rows: list[dict[str, Any]] = []
        for name in fit.params.index:
            label = str(name)
            coef = float(fit.params[name])
            rows.append(
                {
                    "term": label,
                    "source_variable": origin.get(label, "(절편)" if label == "const" else label),
                    "coefficient": num(coef),
                    "std_error": num(fit.bse[name]),
                    "z_value": num(fit.tvalues[name]),
                    "p_value": num(fit.pvalues[name]),
                    "odds_ratio": num(np.exp(coef)),
                    "or_ci_lower": num(np.exp(conf.loc[name].iloc[0])),
                    "or_ci_upper": num(np.exp(conf.loc[name].iloc[1])),
                    "significant": bool(fit.pvalues[name] < 0.05),
                }
            )
        return rows

    @staticmethod
    def _metrics(
        y_te: pd.Series, pred: np.ndarray, proba: np.ndarray, w_te: pd.Series | None, binary: bool
    ) -> dict[str, Any]:
        sw = w_te.to_numpy() if w_te is not None else None
        average = "binary" if binary else "macro"
        metrics: dict[str, Any] = {
            "accuracy": num(accuracy_score(y_te, pred, sample_weight=sw)),
            "precision": num(precision_score(y_te, pred, average=average, zero_division=0, sample_weight=sw)),
            "recall": num(recall_score(y_te, pred, average=average, zero_division=0, sample_weight=sw)),
            "f1": num(f1_score(y_te, pred, average=average, zero_division=0, sample_weight=sw)),
        }
        try:
            metrics["log_loss"] = num(log_loss(y_te, proba, sample_weight=sw))
        except ValueError:
            metrics["log_loss"] = None
        try:
            metrics["roc_auc"] = num(
                roc_auc_score(y_te, proba[:, 1], sample_weight=sw)
                if binary
                else roc_auc_score(y_te, proba, multi_class="ovr", average="macro", sample_weight=sw)
            )
        except ValueError:
            metrics["roc_auc"] = None
        return metrics


def _durbin_watson(residuals: np.ndarray) -> float:
    diff = np.diff(residuals)
    denominator = np.sum(residuals**2)
    return float(np.sum(diff**2) / denominator) if denominator else float("nan")
