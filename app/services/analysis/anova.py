"""분산분석(ANOVA) / 공분산분석(ANCOVA)."""

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sps
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from app.core.exceptions import AnalysisError
from app.models.enums import AnalysisMethod, ChartKind
from app.schemas.analysis import AnalysisParams, PreprocessingSpec
from app.services.analysis.base import AnalysisOutcome, jsonable, num, rec
from app.services.preprocessing import apply_category_orders, apply_filters, handle_missing


def _quote(column: str) -> str:
    """patsy 수식에서 한글·공백 등을 포함한 변수명을 안전하게 참조한다.

    Args:
        column (str): 수식에 사용할 원본 변수 이름.

    Returns:
        str: patsy의 Q() 구문으로 감싼 변수 참조 문자열.
    """
    return f"Q('{column}')"


class AnovaEngine:
    """일원/이원 분산분석과 공변량을 포함한 공분산분석을 함께 지원하는 엔진.

    covariates가 비어 있으면 ANOVA, 1개 이상이면 ANCOVA로 동작한다.
    """

    method = AnalysisMethod.ANOVA

    def run(self, frame: pd.DataFrame, params: AnalysisParams, spec: PreprocessingSpec) -> AnalysisOutcome:
        target = params.target
        factors = params.factors
        covariates = params.covariates
        if not target:
            raise AnalysisError("종속변수(target)가 필요합니다.")
        if not factors:
            raise AnalysisError("요인(factors)이 1개 이상 필요합니다.")

        used = list(dict.fromkeys([target, *factors, *covariates]))
        missing_cols = [c for c in used if c not in frame.columns]
        if missing_cols:
            msg = f"데이터셋에 없는 변수입니다: {', '.join(missing_cols)}"
            raise AnalysisError(msg)

        work = frame[used].copy()
        work, steps = apply_filters(work, spec.filters)
        work, s = handle_missing(work, spec)
        steps += s
        work, s = apply_category_orders(work, spec.category_orders)
        steps += s

        work[target] = pd.to_numeric(work[target], errors="coerce")
        for cov in covariates:
            work[cov] = pd.to_numeric(work[cov], errors="coerce")
        work = work.dropna(subset=[target, *covariates])
        for f in factors:
            work[f] = work[f].astype(str)

        if len(work) < len(used) + 3:
            msg = f"분석 가능한 행이 {len(work)}건뿐입니다. 요인·공변량 수 대비 표본이 부족합니다."
            raise AnalysisError(msg)
        for f in factors:
            if work[f].nunique() < 2:
                msg = f"요인 '{f}'의 범주가 2개 미만이라 분산분석을 수행할 수 없습니다."
                raise AnalysisError(msg)

        two_way = len(factors) == 2 and params.include_interaction
        factor_terms = [f"C({_quote(f)})" for f in factors]
        interaction_term = [f"{factor_terms[0]}:{factor_terms[1]}"] if two_way else []
        rhs_terms = [*factor_terms, *(_quote(c) for c in covariates), *interaction_term]
        formula = f"{_quote(target)} ~ " + " + ".join(rhs_terms)

        fit = smf.ols(formula, data=work).fit()
        table = anova_lm(fit, typ=2)
        ss_total = float(table["sum_sq"].sum())

        anova_table = [self._anova_row(term, row, factors, ss_total) for term, row in table.iterrows()]

        levene_p = self._levene(work, target, factors[0])
        residuals = np.asarray(fit.resid)
        shapiro_p = None
        if 8 <= len(residuals) <= 5000:
            shapiro_p = num(sps.shapiro(residuals)[1])

        post_hoc = {f: self._tukey(work, target, f) for f in factors if work[f].nunique() > 2}

        group_means = self._group_means(work, target, factors)

        result: dict[str, Any] = {
            "method": "공분산분석(ANCOVA)" if covariates else "분산분석(ANOVA)",
            "target": target,
            "factors": factors,
            "covariates": covariates,
            "interaction_included": two_way,
            "n_observations": len(work),
            "anova_table": anova_table,
            "model_r_squared": num(fit.rsquared),
            "group_means": group_means,
            "assumption_checks": {
                "levene": {
                    "test": "Levene",
                    "factor": factors[0],
                    "p_value": levene_p,
                    "equal_variance": bool(levene_p is not None and levene_p > 0.05),
                },
                "normality": {
                    "test": "Shapiro-Wilk",
                    "p_value": shapiro_p,
                    "normal": bool(shapiro_p is not None and shapiro_p > 0.05),
                }
                if shapiro_p is not None
                else None,
            },
            "post_hoc": post_hoc,
            "preprocessing_steps": steps,
            "note": (
                "요인이 3개 이상이면 교호작용은 계산하지 않고 주효과만 산출합니다. "
                "사후검정(Tukey HSD)은 해당 요인 단독 비교이며 다른 요인·공변량을 통제하지 않습니다."
            ),
        }

        eta_values = [t["eta_squared"] for t in anova_table if t["eta_squared"] is not None]
        metrics = {
            "n_observations": len(work),
            "model_r_squared": num(fit.rsquared),
            "n_significant_terms": sum(1 for t in anova_table if t["significant"]),
            "max_eta_squared": max(eta_values, default=None),
            "levene_p_value": levene_p,
        }

        group_boxplots = self._group_boxplots(work, target, factors)
        artifacts = {
            "group_means": group_means,
            "group_boxplots": group_boxplots,
            "anova_table": anova_table,
            "frame": work,
        }
        recommendations = [
            rec(
                ChartKind.MEANS_PLOT,
                "집단별 평균 비교",
                "요인 수준 간 평균과 신뢰구간 비교",
                1,
                data_key="group_means",
                encoding={"x": "group", "y": "mean"},
                options={"error_bars": ["ci_lower", "ci_upper"], "group_by": "factor"},
            ),
            rec(
                ChartKind.BOXPLOT,
                "집단별 분포",
                "등분산성 가정과 집단 간 분포 형태 확인",
                2,
                data_key="group_boxplots",
                encoding={"x": "group", "y": "median"},
                options={"whisker_fields": ["lower_fence", "q1", "median", "q3", "upper_fence"], "group_by": "factor"},
            ),
            rec(
                ChartKind.BAR,
                "요인별 효과크기(η²)",
                "어떤 요인이 종속변수 변동을 더 많이 설명하는지 비교",
                3,
                data_key="anova_table",
                encoding={"x": "term", "y": "eta_squared"},
                options={"sort": "-y"},
            ),
        ]
        return AnalysisOutcome(jsonable(result), jsonable(metrics), artifacts, recommendations)

    @staticmethod
    def _anova_row(term: str, row: pd.Series, factors: list[str], ss_total: float) -> dict[str, Any]:
        p_value = row.get("PR(>F)")
        significant = p_value is not None and not np.isnan(p_value) and p_value < 0.05
        return {
            "term": AnovaEngine._label(str(term), factors),
            "sum_sq": num(row["sum_sq"]),
            "df": num(row["df"]),
            "f_statistic": num(row.get("F")),
            "p_value": num(p_value),
            "eta_squared": num(row["sum_sq"] / ss_total) if ss_total else None,
            "significant": bool(significant),
        }

    @staticmethod
    def _label(term: str, factors: list[str]) -> str:
        label = term
        for f in factors:
            label = label.replace(f"C(Q('{f}'))", f)
        return label.replace(":", " × ")

    @staticmethod
    def _levene(work: pd.DataFrame, target: str, factor: str) -> float | None:
        groups = [g[target].to_numpy() for _, g in work.groupby(factor) if len(g) > 1]
        if len(groups) < 2:
            return None
        return num(sps.levene(*groups)[1])

    @staticmethod
    def _tukey(work: pd.DataFrame, target: str, factor: str) -> list[dict[str, Any]]:
        result = pairwise_tukeyhsd(endog=work[target].to_numpy(), groups=work[factor].to_numpy())
        rows: list[dict[str, Any]] = []
        for row in result.summary().data[1:]:
            group1, group2, meandiff, p_adj, lower, upper, reject = row
            rows.append(
                {
                    "group1": str(group1),
                    "group2": str(group2),
                    "mean_diff": num(meandiff),
                    "p_adj": num(p_adj),
                    "ci_lower": num(lower),
                    "ci_upper": num(upper),
                    "significant": bool(reject),
                }
            )
        return rows

    @staticmethod
    def _group_means(work: pd.DataFrame, target: str, factors: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for factor in factors:
            for group, values in work.groupby(factor)[target]:
                n = len(values)
                mean = float(values.mean())
                sem = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                ci = 1.96 * sem
                rows.append(
                    {
                        "factor": factor,
                        "group": str(group),
                        "n": n,
                        "mean": num(mean),
                        "std": num(values.std(ddof=1)) if n > 1 else None,
                        "ci_lower": num(mean - ci),
                        "ci_upper": num(mean + ci),
                    }
                )
        return rows

    @staticmethod
    def _group_boxplots(work: pd.DataFrame, target: str, factors: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for factor in factors:
            for group, values in work.groupby(factor)[target]:
                q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                inside = values[(values >= lower) & (values <= upper)]
                rows.append(
                    {
                        "factor": factor,
                        "group": str(group),
                        "min": num(values.min()),
                        "q1": num(q1),
                        "median": num(values.median()),
                        "q3": num(q3),
                        "max": num(values.max()),
                        "lower_fence": num(inside.min()) if not inside.empty else num(values.min()),
                        "upper_fence": num(inside.max()) if not inside.empty else num(values.max()),
                    }
                )
        return rows
