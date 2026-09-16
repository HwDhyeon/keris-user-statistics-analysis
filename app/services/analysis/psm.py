"""성향점수매칭(PSM). 정책/프로그램 참여 효과성 같은 인과추론 수요를 지원한다."""

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps
from sklearn.neighbors import NearestNeighbors

from app.core.exceptions import AnalysisError
from app.models.enums import AnalysisMethod, ChartKind
from app.schemas.analysis import AnalysisParams, PreprocessingSpec
from app.services.analysis.base import AnalysisOutcome, jsonable, num, rec
from app.services.preprocessing import build_design_matrix


class PSMEngine:
    """성향점수매칭(PSM) 엔진. 로지스틱 회귀로 성향점수를 산출하고, 최근접이웃 매칭으로 처치/대조군 균형을 맞춘다."""

    method = AnalysisMethod.PSM

    def run(self, frame: pd.DataFrame, params: AnalysisParams, spec: PreprocessingSpec) -> AnalysisOutcome:
        treatment_column = params.treatment_column
        outcome_column = params.target
        if not treatment_column:
            raise AnalysisError("treatment_column(처치 여부 변수)이 필요합니다.")
        if not params.features:
            raise AnalysisError("features(성향점수 산출용 공변량)가 1개 이상 필요합니다.")

        design = build_design_matrix(
            frame,
            features=params.features,
            target=treatment_column,
            weight_column=None,
            spec=spec,
            target_as_category=True,
            extra_columns=[outcome_column] if outcome_column else None,
        )
        X, treat = design.X, design.y
        classes = design.target_classes or []
        if treat is None or len(classes) != 2:
            msg = (
                f"treatment_column '{treatment_column}'은 범주가 정확히 2개인 이분형이어야 합니다. "
                f"(현재 {len(classes)}개)"
            )
            raise AnalysisError(msg)

        # positive_label 지정 시 해당 범주를 처치군(1)으로 재배치한다.
        if params.positive_label is not None:
            label = str(params.positive_label)
            if label not in classes:
                msg = f"positive_label '{label}'이 treatment_column 범주에 없습니다: {classes}"
                raise AnalysisError(msg)
            if classes.index(label) == 0:
                treat = 1 - treat
                classes = [classes[1], classes[0]]

        n_treated_total = int((treat == 1).sum())
        n_control_total = int((treat == 0).sum())
        if n_treated_total < 5 or n_control_total < 5:
            raise AnalysisError("처치군·대조군 각각 최소 5건 이상 필요합니다.")

        exog = sm.add_constant(X, has_constant="add")
        try:
            ps_fit = sm.Logit(treat.astype(float), exog.astype(float)).fit(disp=False, maxiter=params.max_iter)
        except Exception as exc:
            msg = f"성향점수 모형(로지스틱 회귀) 적합에 실패했습니다: {exc}"
            raise AnalysisError(msg) from exc

        propensity = np.clip(np.asarray(ps_fit.predict(exog)), 1e-6, 1 - 1e-6)
        logit_ps = np.log(propensity / (1 - propensity))

        caliper = params.caliper
        if caliper is None:
            caliper = 0.2 * float(np.std(logit_ps))

        treated_mask = (treat == 1).to_numpy()
        pairs = self._match(
            logit_ps=logit_ps,
            treated_mask=treated_mask,
            n_neighbors=params.n_neighbors,
            caliper=caliper,
            with_replacement=params.matching_replacement,
        )

        outcome = design.frame[outcome_column] if outcome_column else None
        balance = self._balance(X, treated_mask, pairs)
        att_result = self._att(outcome, pairs) if outcome is not None else None

        matched_treated = len({p["treated_index"] for p in pairs})
        result: dict[str, Any] = {
            "method": "성향점수매칭(PSM)",
            "treatment_column": treatment_column,
            "treated_label": classes[1],
            "control_label": classes[0],
            "outcome": outcome_column,
            "covariates": params.features,
            "n_treated": n_treated_total,
            "n_control": n_control_total,
            "n_matched_treated": matched_treated,
            "match_rate": round(matched_treated / n_treated_total, 6) if n_treated_total else None,
            "caliper": num(caliper),
            "n_neighbors": params.n_neighbors,
            "with_replacement": params.matching_replacement,
            "propensity_model": {
                "pseudo_r_squared": num(ps_fit.prsquared),
                "coefficients": self._coefficients(ps_fit, design.feature_origin),
            },
            "covariate_balance": balance,
            "att": att_result,
            "preprocessing_steps": design.steps,
            "note": (
                "SMD(표준화 평균차)는 절대값 0.1 미만이면 해당 공변량의 균형이 양호하다고 판단합니다. "
                "att는 outcome(target)을 지정했을 때만 산출되며, 매칭된 처치군-대조군 쌍의 평균 결과 차이입니다."
            ),
        }
        metrics = {
            "n_treated": n_treated_total,
            "n_matched_treated": matched_treated,
            "match_rate": result["match_rate"],
            "mean_abs_smd_before": balance["mean_abs_smd_before"],
            "mean_abs_smd_after": balance["mean_abs_smd_after"],
            "att": att_result["att"] if att_result else None,
            "att_p_value": att_result["p_value"] if att_result else None,
        }

        propensity_distribution = self._ps_distribution(propensity, treated_mask, pairs)
        artifacts = {
            "propensity_distribution": propensity_distribution,
            "balance": balance["covariates"],
            "coefficients": result["propensity_model"]["coefficients"],
        }
        if att_result:
            artifacts["att"] = [att_result]
        recommendations = [
            rec(
                ChartKind.PS_DISTRIBUTION,
                "성향점수 분포(매칭 전/후)",
                "처치군·대조군의 성향점수 겹침(공통지지영역) 확인",
                1,
                data_key="propensity_distribution",
                encoding={"x": "propensity", "color": "group"},
                options={"facet": "matched"},
            ),
            rec(
                ChartKind.LOVE_PLOT,
                "공변량 균형(Love Plot)",
                "매칭 전/후 표준화 평균차(SMD) 비교 — 매칭 후 0에 가까울수록 균형이 좋음",
                2,
                data_key="balance",
                encoding={"y": "covariate", "x": "smd_after"},
                options={"reference_lines": [-0.1, 0.1], "compare_field": "smd_before"},
            ),
        ]
        if att_result:
            recommendations.append(
                rec(
                    ChartKind.BAR,
                    "처치효과(ATT)",
                    "매칭된 표본에서 추정한 평균처치효과와 신뢰구간",
                    3,
                    data_key="att",
                    encoding={"y": "att"},
                    options={"error_bars": ["ci_lower", "ci_upper"]},
                )
            )
        return AnalysisOutcome(jsonable(result), jsonable(metrics), artifacts, recommendations)

    @staticmethod
    def _match(
        *,
        logit_ps: np.ndarray,
        treated_mask: np.ndarray,
        n_neighbors: int,
        caliper: float,
        with_replacement: bool,
    ) -> list[dict[str, Any]]:
        treated_idx = np.where(treated_mask)[0]
        control_idx = np.where(~treated_mask)[0]
        if control_idx.size == 0:
            return []

        pairs: list[dict[str, Any]] = []
        if with_replacement:
            nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(control_idx)))
            nn.fit(logit_ps[control_idx].reshape(-1, 1))
            distances, neighbors = nn.kneighbors(logit_ps[treated_idx].reshape(-1, 1))
            for row, t in enumerate(treated_idx):
                for col in range(distances.shape[1]):
                    dist = float(distances[row, col])
                    if dist > caliper:
                        continue
                    pairs.append(
                        {
                            "treated_index": int(t),
                            "control_index": int(control_idx[neighbors[row, col]]),
                            "distance": dist,
                        }
                    )
            return pairs

        # 비복원 그리디 매칭: 성향점수 거리가 가까운 처치 단위부터 순서대로 배정한다.
        available = set(control_idx.tolist())
        order = sorted(treated_idx.tolist(), key=lambda t: logit_ps[t])
        for t in order:
            if not available:
                break
            candidates = np.array(sorted(available))
            dist = np.abs(logit_ps[candidates] - logit_ps[t])
            k = min(n_neighbors, len(candidates))
            nearest_pos = np.argsort(dist)[:k]
            for pos in nearest_pos:
                if dist[pos] > caliper:
                    continue
                c = int(candidates[pos])
                pairs.append({"treated_index": int(t), "control_index": c, "distance": float(dist[pos])})
                available.discard(c)
        return pairs

    @staticmethod
    def _balance(X: pd.DataFrame, treated_mask: np.ndarray, pairs: list[dict[str, Any]]) -> dict[str, Any]:
        matched_treated_idx = [p["treated_index"] for p in pairs]
        matched_control_idx = [p["control_index"] for p in pairs]
        covariates: list[dict[str, Any]] = []
        for col in X.columns:
            values = X[col].to_numpy(dtype=float)
            smd_before = _smd(values[treated_mask], values[~treated_mask])
            smd_after = (
                _smd(values[matched_treated_idx], values[matched_control_idx]) if pairs else None
            )
            covariates.append(
                {
                    "covariate": str(col),
                    "smd_before": num(smd_before),
                    "smd_after": num(smd_after),
                    "balanced": bool(smd_after is not None and abs(smd_after) < 0.1),
                }
            )
        before_vals = [abs(c["smd_before"]) for c in covariates if c["smd_before"] is not None]
        after_vals = [abs(c["smd_after"]) for c in covariates if c["smd_after"] is not None]
        return {
            "covariates": covariates,
            "mean_abs_smd_before": num(np.mean(before_vals)) if before_vals else None,
            "mean_abs_smd_after": num(np.mean(after_vals)) if after_vals else None,
        }

    @staticmethod
    def _att(outcome: pd.Series, pairs: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not pairs:
            return None
        values = pd.to_numeric(outcome, errors="coerce").to_numpy()
        by_treated: dict[int, list[float]] = {}
        for p in pairs:
            by_treated.setdefault(p["treated_index"], []).append(values[p["control_index"]])

        diffs = []
        for t, controls in by_treated.items():
            controls_clean = [c for c in controls if not np.isnan(c)]
            if not controls_clean or np.isnan(values[t]):
                continue
            diffs.append(values[t] - float(np.mean(controls_clean)))
        if len(diffs) < 2:
            return None

        diffs_arr = np.array(diffs)
        att = float(diffs_arr.mean())
        se = float(diffs_arr.std(ddof=1) / np.sqrt(len(diffs_arr)))
        t_stat, p_value = sps.ttest_1samp(diffs_arr, 0.0)
        ci = sps.t.ppf(0.975, df=len(diffs_arr) - 1) * se
        return {
            "att": num(att),
            "std_error": num(se),
            "t_statistic": num(t_stat),
            "p_value": num(p_value),
            "ci_lower": num(att - ci),
            "ci_upper": num(att + ci),
            "n_pairs": len(diffs_arr),
        }

    @staticmethod
    def _ps_distribution(
        propensity: np.ndarray, treated_mask: np.ndarray, pairs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        matched = {p["treated_index"] for p in pairs} | {p["control_index"] for p in pairs}
        return [
            {
                "propensity": num(p),
                "group": "처치군" if treated_mask[i] else "대조군",
                "matched": i in matched,
            }
            for i, p in enumerate(propensity)
        ]

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
                    "p_value": num(fit.pvalues[name]),
                    "ci_lower": num(conf.loc[name].iloc[0]),
                    "ci_upper": num(conf.loc[name].iloc[1]),
                    "significant": bool(fit.pvalues[name] < 0.05),
                }
            )
        return rows


def _smd(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return None
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)
