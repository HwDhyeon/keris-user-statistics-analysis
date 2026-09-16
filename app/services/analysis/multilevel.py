"""다층모형(HLM)과 성장모형(Growth Curve). 둘 다 statsmodels의 선형 혼합모형(MixedLM)을 사용한다."""

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from app.core.exceptions import AnalysisError
from app.models.enums import AnalysisMethod, ChartKind
from app.schemas.analysis import AnalysisParams, PreprocessingSpec
from app.services.analysis.base import AnalysisOutcome, jsonable, num, rec
from app.services.preprocessing import build_design_matrix


def _fit(
    y: pd.Series,
    exog: pd.DataFrame,
    groups: pd.Series,
    exog_re: pd.DataFrame | None,
) -> Any:
    # 열 이름(exog_names)을 유지해야 fe_params/conf_int 등이 라벨 있는 Series/DataFrame으로 반환된다.
    model = sm.MixedLM(
        endog=y.to_numpy(),
        exog=exog,
        groups=groups.to_numpy(),
        exog_re=exog_re,
    )

    try:
        return model.fit(reml=True)
    except np.linalg.LinAlgError as exc:
        msg = "혼합모형 적합에 실패했습니다. 집단 수·표본 수 대비 확률효과 구조가 과도하게 복잡할 수 있습니다."
        raise AnalysisError(msg) from exc


def _fixed_effects_table(
    fit: Any,
    exog_columns: list[str],
    origin: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    conf = fit.conf_int()
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(exog_columns):
        p = num(fit.pvalues.iloc[i])
        rows.append(
            {
                "term": name,
                "source_variable": (origin or {}).get(name, "(절편)" if name == "const" else name),
                "coefficient": num(fit.fe_params.iloc[i]),
                "std_error": num(fit.bse_fe.iloc[i]),
                "p_value": p,
                "ci_lower": num(conf.iloc[i, 0]),
                "ci_upper": num(conf.iloc[i, 1]),
                "significant": bool(p is not None and p < 0.05),
            }
        )
    return rows


def _variance_components(fit: Any, re_names: list[str]) -> dict[str, Any]:
    cov_re = fit.cov_re
    sigma2 = float(fit.scale)
    tau00 = float(cov_re.iloc[0, 0])
    icc = tau00 / (tau00 + sigma2) if (tau00 + sigma2) else None
    components = {
        "group_variance": num(tau00),
        "residual_variance": num(sigma2),
        "icc": num(icc),
    }
    if len(re_names) > 1:
        components["slope_variance"] = num(cov_re.iloc[1, 1])
        components["intercept_slope_covariance"] = num(cov_re.iloc[0, 1])
        denom = np.sqrt(cov_re.iloc[0, 0] * cov_re.iloc[1, 1])
        components["intercept_slope_correlation"] = num(cov_re.iloc[0, 1] / denom) if denom else None
    return components


def _random_effects_rows(fit: Any, re_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, effects in fit.random_effects.items():
        row: dict[str, Any] = {"group": str(group)}
        for name in re_names:
            row[name] = num(effects.get(name))
        rows.append(row)
    key = re_names[0]
    return sorted(rows, key=lambda r: (r[key] is None, r[key]))


class MultilevelEngine:
    """다층모형(HLM) 엔진. 학생 within 학급/학교 같은 위계구조에서 집단 간·집단 내 변동을 분리한다."""

    method = AnalysisMethod.MULTILEVEL

    def run(self, frame: pd.DataFrame, params: AnalysisParams, spec: PreprocessingSpec) -> AnalysisOutcome:
        group_column = params.group_column
        if not group_column:
            raise AnalysisError("group_column(위계 집단 변수)이 필요합니다.")

        design = build_design_matrix(
            frame,
            features=params.features,
            target=params.target,
            weight_column=None,
            spec=spec,
            extra_columns=[group_column],
        )
        X, y = design.X, design.y
        if y is None:
            raise AnalysisError("종속변수가 필요합니다.")
        groups = design.frame[group_column].astype(str)

        n_groups = groups.nunique()
        if n_groups < 3:
            msg = f"위계 집단 수가 {n_groups}개뿐입니다. 다층모형은 최소 3개 이상의 집단이 필요합니다."
            raise AnalysisError(msg)

        const = pd.DataFrame({"const": np.ones(len(X))}, index=X.index)

        # 무조건모형(intercept-only): 집단 간/집단 내 분산 비율(ICC)의 기준선
        null_fit = _fit(y, const, groups, exog_re=None)
        null_variance = _variance_components(null_fit, ["const"])

        exog = pd.concat([const, X], axis=1)
        random_slope_on: str | None = None
        exog_re = const
        if params.random_slope and len(X.columns) > 0:
            random_slope_on = str(X.columns[0])
            exog_re = pd.concat([const, X[[random_slope_on]]], axis=1)

        fit = _fit(y, exog, groups, exog_re=exog_re)

        re_names = list(exog_re.columns)
        fixed_effects = _fixed_effects_table(fit, list(exog.columns), design.feature_origin)
        variance_components = _variance_components(fit, re_names)
        random_effects = _random_effects_rows(fit, re_names)

        result: dict[str, Any] = {
            "method": "다층모형(HLM)",
            "target": params.target,
            "features": params.features,
            "group_column": group_column,
            "n_groups": n_groups,
            "n_observations": len(X),
            "random_slope_feature": random_slope_on if params.random_slope else None,
            "fixed_effects": fixed_effects,
            "unconditional_icc": variance_components["icc"],
            "unconditional_variance": null_variance,
            "variance_components": variance_components,
            "random_effects_preview": random_effects[:200],
            "model_fit": {
                "log_likelihood": num(fit.llf),
                "aic": num(fit.aic),
                "bic": num(fit.bic),
                "converged": bool(fit.converged),
            },
            "preprocessing_steps": design.steps,
            "note": (
                "확률기울기(random_slope=true)는 독립변수 중 첫 번째 변수 하나에만 적용됩니다. "
                "unconditional_icc는 예측변수를 넣기 전 무조건모형 기준 집단 간 분산 비율입니다."
            ),
        }
        metrics = {
            "n_groups": n_groups,
            "n_observations": len(X),
            "icc": variance_components["icc"],
            "unconditional_icc": null_variance["icc"],
            "aic": num(fit.aic),
            "bic": num(fit.bic),
            "log_likelihood": num(fit.llf),
        }
        artifacts = {
            "coefficients": fixed_effects,
            "random_effects": random_effects,
            "variance_components": [{"component": k, "value": v} for k, v in variance_components.items() if k != "icc"],
        }
        recommendations = [
            rec(
                ChartKind.BAR,
                "고정효과 계수",
                "독립변수별 평균적인 영향력과 유의성 비교",
                1,
                data_key="coefficients",
                encoding={"x": "value", "y": "term", "color": "significant"},
                options={"sort": "-x", "error_bars": ["ci_lower", "ci_upper"]},
            ),
            rec(
                ChartKind.RANDOM_EFFECTS,
                "집단별 확률효과(캐터필러 플롯)",
                "학급/학교별 절편(과 기울기)이 평균에서 얼마나 벗어나는지 확인",
                2,
                data_key="random_effects",
                encoding={"x": "group", "y": "const"},
                options={"sort": "y", "slope_field": random_slope_on},
            ),
            rec(
                ChartKind.VARIANCE_COMPONENTS,
                "분산성분 분해",
                "종속변수 변동 중 집단 간(between) vs 집단 내(within) 비율 — ICC",
                3,
                data_key="variance_components",
                encoding={"x": "component", "y": "value"},
            ),
        ]
        return AnalysisOutcome(jsonable(result), jsonable(metrics), artifacts, recommendations)


class GrowthCurveEngine:
    """성장모형(확률효과 방식) 엔진. 패널 개체별 시간에 따른 변화 궤적을 추적한다."""

    method = AnalysisMethod.GROWTH_CURVE

    def run(self, frame: pd.DataFrame, params: AnalysisParams, spec: PreprocessingSpec) -> AnalysisOutcome:
        group_column = params.group_column
        time_column = params.time_column
        if not group_column or not time_column:
            raise AnalysisError("group_column(패널 개체 식별 변수)과 time_column(시점 변수)이 필요합니다.")
        if time_column in params.features:
            msg = "time_column은 features에 중복으로 포함할 수 없습니다."
            raise AnalysisError(msg)

        design = build_design_matrix(
            frame,
            features=[time_column, *params.features],
            target=params.target,
            weight_column=None,
            spec=spec,
            extra_columns=[group_column],
        )
        X, y = design.X, design.y
        if y is None:
            raise AnalysisError("종속변수가 필요합니다.")
        groups = design.frame[group_column].astype(str)
        time = X[time_column]

        n_groups = groups.nunique()
        if n_groups < 3:
            msg = f"패널 개체 수가 {n_groups}개뿐입니다. 성장모형은 최소 3개 이상의 개체가 필요합니다."
            raise AnalysisError(msg)
        if (groups.value_counts() < 2).any():
            raise AnalysisError("일부 개체의 관측 시점이 1개뿐입니다. 성장모형은 개체별 2개 이상의 시점이 필요합니다.")

        const = pd.DataFrame({"const": np.ones(len(X))}, index=X.index)
        exog = pd.concat([const, X], axis=1)
        exog_re = pd.concat([const, time.to_frame("time_slope")], axis=1) if params.random_slope else const

        fit = _fit(y, exog, groups, exog_re=exog_re if params.random_slope else None)

        re_names = ["const", "time_slope"] if params.random_slope else ["const"]
        fixed_effects = _fixed_effects_table(fit, list(exog.columns), design.feature_origin)
        variance_components = _variance_components(fit, re_names)
        random_effects = _random_effects_rows(fit, re_names)

        avg_slope = next((r["coefficient"] for r in fixed_effects if r["term"] == time_column), None)
        avg_slope_p = next((r["p_value"] for r in fixed_effects if r["term"] == time_column), None)

        trajectories = self._trajectories(
            fit, design.frame, groups, time_column, target=params.target, re_names=re_names
        )

        result: dict[str, Any] = {
            "method": "성장모형(확률효과)",
            "target": params.target,
            "time_column": time_column,
            "covariates": params.features,
            "group_column": group_column,
            "n_individuals": n_groups,
            "n_observations": len(X),
            "random_slope_included": params.random_slope,
            "fixed_effects": fixed_effects,
            "average_growth_rate": avg_slope,
            "average_growth_rate_p_value": avg_slope_p,
            "variance_components": variance_components,
            "individual_trajectories_preview": trajectories[:20],
            "model_fit": {
                "log_likelihood": num(fit.llf),
                "aic": num(fit.aic),
                "bic": num(fit.bic),
                "converged": bool(fit.converged),
            },
            "preprocessing_steps": design.steps,
            "note": "average_growth_rate는 시간(time_column) 1단위 증가에 따른 평균적인 target 변화량입니다.",
        }
        metrics = {
            "n_individuals": n_groups,
            "n_observations": len(X),
            "average_growth_rate": avg_slope,
            "average_growth_rate_p_value": avg_slope_p,
            "slope_variance": variance_components.get("slope_variance"),
            "aic": num(fit.aic),
            "bic": num(fit.bic),
        }
        artifacts = {
            "coefficients": fixed_effects,
            "random_effects": random_effects,
            "variance_components": [{"component": k, "value": v} for k, v in variance_components.items() if k != "icc"],
            "trajectories": trajectories,
        }
        recommendations = [
            rec(
                ChartKind.TRAJECTORY,
                "개체별 성장 궤적",
                "시간에 따른 개체별 변화와 평균 성장 추세 비교",
                1,
                data_key="trajectories",
                encoding={"x": "time", "y": "value", "color": "group"},
                options={"highlight": "average"},
            ),
            rec(
                ChartKind.BAR,
                "고정효과 계수",
                "시간(성장률)과 공변량의 평균적인 영향력",
                2,
                data_key="coefficients",
                encoding={"x": "value", "y": "term", "color": "significant"},
                options={"sort": "-x", "error_bars": ["ci_lower", "ci_upper"]},
            ),
            rec(
                ChartKind.VARIANCE_COMPONENTS,
                "분산성분 분해",
                "개체 간 초기값·성장률 차이와 잔차 분산 비교",
                3,
                data_key="variance_components",
                encoding={"x": "component", "y": "value"},
            ),
        ]
        return AnalysisOutcome(jsonable(result), jsonable(metrics), artifacts, recommendations)

    @staticmethod
    def _trajectories(
        fit: Any,
        frame: pd.DataFrame,
        groups: pd.Series,
        time_column: str,
        target: str | None,
        re_names: list[str],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        avg_intercept = float(fit.fe_params.get("const", 0.0))
        avg_slope = float(fit.fe_params.get(time_column, 0.0))
        for group, effects in fit.random_effects.items():
            mask = groups == str(group)
            times = frame.loc[mask, time_column]
            if times.empty:
                continue
            re_intercept = float(effects.get("const", 0.0))
            re_slope = float(effects.get("time_slope", 0.0)) if "time_slope" in re_names else 0.0
            t_min, t_max = float(times.min()), float(times.max())
            for t in (t_min, t_max):
                predicted = (avg_intercept + re_intercept) + (avg_slope + re_slope) * t
                rows.append({"group": str(group), "time": num(t), "value": num(predicted), "kind": "individual"})
            actual = frame.loc[mask, target] if target else None
            if actual is not None:
                for t, v in zip(times, actual, strict=True):
                    rows.append({"group": str(group), "time": num(t), "value": num(v), "kind": "observed"})
        t_min, t_max = float(frame[time_column].min()), float(frame[time_column].max())
        for t in (t_min, t_max):
            rows.append(
                {"group": "평균", "time": num(t), "value": num(avg_intercept + avg_slope * t), "kind": "average"}
            )
        return rows
