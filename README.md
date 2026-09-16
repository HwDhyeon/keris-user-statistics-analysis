# 이용자 통계분석 서비스 — 백엔드 API

사용자가 보유한 로컬 파일(CSV·Excel 등)을 시스템으로 직접 반입해 **즉시 분석**하고,
기초 통계부터 회귀·의사결정나무·군집분석까지 수행한 뒤, 그 조건과 결과를
**재현 가능한 패키지**로 보존·공유하는 FastAPI 기반 백엔드입니다.

- Python 3.14 / uv
- FastAPI · Pydantic v2 · SQLModel
- pandas · numpy · scipy · scikit-learn · statsmodels
- **차트 렌더링은 포함하지 않습니다.** 서버는 차트 *추천*과 *차트용 데이터*만 제공하고, 그리는 일은 웹 클라이언트가 맡습니다.

## 빠른 시작

```bash
uv sync                       # 의존성 설치
cp .env.example .env          # 설정 파일 (선택)
uv run fastapi dev app/main.py
```

- API 문서(Swagger): http://127.0.0.1:8000/docs
- 헬스 체크: http://127.0.0.1:8000/health

```bash
uv run pytest         # 테스트
uv run ruff check .   # 린트
```

## 기능 매핑

### ① 로컬 파일 반입 및 즉시 분석

| 기능 | 엔드포인트 |
| --- | --- |
| 파일 업로드(CSV/TSV/Excel/JSON/Parquet) | `POST /api/v1/datasets/upload` |
| 데이터 미리보기 | `GET /api/v1/datasets/{id}/preview` |
| 변수 목록·역할·범주 | `GET /api/v1/datasets/{id}/columns` |

인코딩(utf-8 / cp949 / euc-kr)과 구분자를 자동 판별하고, 각 변수의 역할
(`numeric`·`categorical`·`datetime`·`identifier` 등)을 추론해 저장합니다.
반입된 데이터는 Parquet으로 보관되어 이후 분석에서 빠르게 재사용됩니다.

### ② 기초 통계량 · 결측치/이상치 자동 탐지 · 시각화 리포트

| 기능 | 엔드포인트 |
| --- | --- |
| 기초 통계량(평균·중앙값·사분위수·왜도·첨도) + 결측·이상치 | `GET /api/v1/datasets/{id}/profile` |
| 이상치 탐지 (IQR / Z-score / Modified Z / Isolation Forest) | `GET /api/v1/datasets/{id}/outliers` |
| 상관관계 행렬 + 유의확률 + 히트맵 셀 | `GET /api/v1/datasets/{id}/correlation` |
| 프로파일 + 상관 + 차트 추천/데이터 통합 리포트 | `GET /api/v1/datasets/{id}/report` |

리포트에는 히트맵용 `(x, y, r)` 셀과 상위 상관 쌍의 산점도 좌표가 함께 담기므로,
분석 모델을 세우기 전에 변수 간 관계를 미리 파악할 수 있습니다.

### ③ 고급 통계 및 예측 분석

`POST /api/v1/analyses` 하나로 아래 기법을 실행합니다. (`GET /api/v1/methods`로 목록 조회)

| 기법 | `method` | 주요 성능 지표 |
| --- | --- | --- |
| 기술통계 | `descriptive` | 결측률, 중복 행 |
| 상관분석 | `correlation` | 최대 \|r\|, 강한 상관 쌍 수 |
| 선형 회귀 | `linear_regression` | **결정계수(R²)**, 수정 R², **RMSE**, MAE, MAPE, AIC/BIC, Durbin-Watson |
| 로지스틱 회귀 | `logistic_regression` | 정확도, 정밀도, 재현율, F1, ROC-AUC, McFadden 유사 R² |
| 의사결정나무(회귀/분류) | `decision_tree_regressor` / `decision_tree_classifier` | R²·RMSE / 정확도·F1·AUC |
| K-평균 군집 | `kmeans` | 실루엣, 이너셔, Davies-Bouldin, Calinski-Harabasz |
| 계층적 군집 | `hierarchical` | 실루엣, 덴드로그램 |
| DBSCAN | `dbscan` | 군집 수, 잡음 비율 |
| 주성분분석 | `pca` | 설명분산 비율 |
| 분산분석(ANOVA/ANCOVA) | `anova` | 요인별 F·p·η², Levene 등분산성, Tukey HSD 사후검정 |
| 다층모형(HLM) | `multilevel` | ICC(집단 간/내 분산비), 고정효과, AIC/BIC |
| 성장모형(Growth Curve) | `growth_curve` | 평균 성장률, 개체별 확률효과(절편·기울기), AIC/BIC |
| 성향점수매칭(PSM) | `psm` | 매칭률, 매칭 전/후 SMD, 평균처치효과(ATT) |

선형회귀는 다중공선성(VIF), 잔차 정규성(Shapiro-Wilk), 등분산성(Breusch-Pagan)
진단을 함께 제공하고, 로지스틱 회귀는 오즈비와 신뢰구간을 산출합니다.

**교육 데이터 위계구조·인과추론 특화 기법**

- **분산분석(ANOVA/ANCOVA)** — 지역별·학교유형별 등 범주형 요인(`params.factors`) 간 평균 차이를
  검정합니다. 공변량(`params.covariates`)을 추가하면 ANCOVA로 동작합니다. 요인이 2개면 교호작용항을
  기본 포함하고, 등분산성(Levene)·잔차 정규성(Shapiro-Wilk) 진단과 요인이 3수준 이상이면
  Tukey HSD 사후검정을 함께 제공합니다.
- **다층모형(HLM)** — 학생 within 학급 within 학교 같은 위계구조에서 `params.group_column`으로 지정한
  집단별 확률효과를 추정합니다(`statsmodels.MixedLM`). 예측변수를 넣기 전 무조건모형 기준
  집단 간 분산 비율(`unconditional_icc`)과, 예측변수 투입 후 조건부 ICC를 함께 보여줘
  "학교 차이로 설명되는 부분"과 "학생 개인차로 설명되는 부분"을 분리합니다.
- **성장모형(Growth Curve, 확률효과)** — 패널 데이터에서 `params.time_column`(시점)에 따른
  `params.group_column`(패널 개체, 예: 학생 ID)별 변화 궤적을 추적합니다. 평균 성장률과
  개체별 확률기울기(초기값·성장속도의 개인차)를 함께 산출합니다. 고정효과(fixed-effects) 패널모형은
  제공하지 않습니다 — 확률효과(random-effects) 방식만 지원합니다.
- **성향점수매칭(PSM)** — `params.treatment_column`(정책/프로그램 참여 여부)에 대해 로지스틱
  회귀로 성향점수를 산출한 뒤, 로짓 성향점수 기준 최근접이웃 매칭(캘리퍼 적용)으로 처치군·대조군의
  공변량 분포를 맞춥니다. 매칭 전/후 표준화평균차(SMD)로 균형을 진단하고, `params.target`(결과변수)을
  지정하면 매칭 표본에서 평균처치효과(ATT)를 추정합니다.

**정교한 분석 파라미터**

- **변수 값 순서 유지** — `preprocessing.category_orders`에 순서를 지정하면 ordered
  Categorical로 고정되고, `encoding: "ordinal"`에서 그 순서가 그대로 코드값이 됩니다.
- **다중 변수 선택** — `params.features`에 독립변수를 배열로 전달합니다.
- **가중치 부여** — `params.weight_column` 지정 시 선형회귀는 WLS로, 나머지 기법은
  `sample_weight`로 반영됩니다.
- 그 밖에 필터, 결측 대치 전략(열별 지정 가능), 이상치 제거, 스케일링,
  홀드아웃 비율(`test_size`), 트리 깊이·가지치기, 군집 수 자동 탐색(`auto_k_range`) 등.

요청 예시:

```jsonc
POST /api/v1/analyses
{
  "dataset_id": "…",
  "method": "linear_regression",
  "params": {
    "target": "성취도점수",
    "features": ["학습시간", "출석률", "학년"],
    "weight_column": "가중치",
    "test_size": 0.25
  },
  "preprocessing": {
    "missing_strategy": "median",
    "remove_outliers": true,
    "encoding": "ordinal",
    "category_orders": { "학년": ["1학년", "2학년", "3학년"] }
  },
  "charts": { "enabled": true }
}
```

### ④ 결과 제시: 차트 추천 (렌더링은 웹에서)

| 기능 | 엔드포인트 |
| --- | --- |
| 기법별 차트 추천 | `GET /api/v1/analyses/{id}/chart-recommendations` |
| 차트용 데이터 조회 (`?key=`로 선택) | `GET /api/v1/analyses/{id}/chart-data` |
| 차트용 데이터 재생성 | `POST /api/v1/analyses/{id}/chart-data` |

분석을 수행하면 응답에 `chart_recommendations`와 `chart_data`가 함께 들어옵니다.
서버는 이미지를 만들지 않고, **무엇을 그릴지**와 **그릴 재료**만 넘깁니다.

```jsonc
// chart_recommendations[0]
{
  "kind": "scatter_with_fit",
  "title": "회귀선이 포함된 산점도",
  "reason": "적합선과 실제 관측치의 부합도를 확인",
  "priority": 1,
  "data_key": "predictions",                        // chart_data에서 꺼낼 키
  "encoding": { "x": "actual", "y": "predicted" },  // 축에 대응하는 필드명
  "options": { "fit_line": true, "identity_line": true }
}
```

추천은 기법에 맞춰 자동 생성됩니다 — 회귀분석이면 **회귀선이 포함된 산점도**·잔차도·
계수 막대그래프, 의사결정나무면 **트리 구조도**·변수 중요도·혼동행렬,
군집분석이면 군집 산점도·엘보 곡선. `priority` 순으로 정렬되어 있으니 상위 몇 개만
골라 그려도 됩니다.

`chart_data`는 원시 데이터를 그대로 흘리지 않고 **차트 단위로 집계·표본추출**해
응답 크기를 제한합니다.

| 키 | 내용 |
| --- | --- |
| `predictions` | `{actual, predicted, residual}` 좌표 (max_points까지 표본추출) |
| `residual_histogram` | `{bin_start, bin_end, bin_center, count}` |
| `roc_curve` | `{fpr, tpr}` (500점으로 균등 축소) |
| `confusion_matrix` | `{labels, cells: [{actual, predicted, count}]}` |
| `coefficients` | `{term, value, measure, p_value, significant, ci_lower, ci_upper}` (절편 제외) |
| `feature_importance` | `{feature, source_variable, importance}` |
| `tree` | `{nodes, links, hierarchy}` — 그래프/계층 레이아웃 양쪽 지원 |
| `projection` | `{x, y, cluster, x_label, y_label}` (변수 3개 이상이면 PCA 축약) |
| `cluster_sizes`, `elbow`, `scree`, `loadings`, `dendrogram` | 각 기법 전용 |
| `histograms`, `boxplots`, `category_counts`, `missing_ratios` | 기술통계·탐색용 분포 |
| `correlation_matrix`, `scatter_pairs` | 히트맵 셀, 상위 상관 쌍 좌표 |

상자그림은 클라이언트가 다시 계산할 필요 없도록 `lower_fence`·`q1`·`median`·`q3`·
`upper_fence`와 이상치 목록을 미리 담아 보냅니다.

산출 여부와 양은 요청에서 조절합니다.

```jsonc
"chart_data": {
  "include": true,        // false면 추천만 받고 데이터는 생략
  "max_points": 5000,     // 산점도 점 개수 상한
  "histogram_bins": 30,
  "max_categories": 20
}
```

이미 수행한 분석에 대해 `POST /analyses/{id}/chart-data`로 옵션만 바꿔
데이터를 다시 만들 수도 있습니다.

### ⑤ 분석 패키지 — 보존 · 재현 · 공유

| 기능 | 엔드포인트 |
| --- | --- |
| 패키지 생성 | `POST /api/v1/packages` |
| ZIP 내려받기 | `GET /api/v1/packages/{id}/download` |
| **재현 실행** | `POST /api/v1/packages/{id}/reproduce` |
| 공유 링크 발급 / 해제 | `POST` / `DELETE /api/v1/packages/{id}/share` |
| 공유된 패키지 열람 | `GET /api/v1/shared/{token}` |

ZIP 구성:

```
manifest.json                    분석 조건·전처리 로직·데이터 체크섬·실행 환경
README.md                        패키지 개요와 재현 절차
analyses/{id}/analysis.json      파라미터·전처리·결과·지표 전문
analyses/{id}/report.md          사람이 읽는 분석 리포트
analyses/{id}/chart-recommendations.json   추천 차트 목록
analyses/{id}/chart-data.json              차트용 집계 데이터
data/{name}.parquet              (선택) 원본 데이터 스냅샷
```

재현 API는 보존된 조건으로 분석을 다시 실행한 뒤 원본 지표와 대조해,
일치 여부(`matched`)와 차이 항목(`comparisons`)을 돌려줍니다. 데이터셋 체크섬이
다르면 그 사실도 함께 알려 결과 차이의 원인을 구분할 수 있게 합니다.

## 구조

```
app/
├── main.py                 FastAPI 앱
├── core/                   설정 · DB · 예외
├── models/                 SQLModel 테이블, 열거형
├── schemas/                요청/응답 Pydantic 모델
├── api/v1/                 datasets · profiling · analyses · packages 라우터
└── services/
    ├── ingest.py           파일 반입, 변수 역할 추론
    ├── profiling.py        기초 통계, 결측·이상치, 상관관계
    ├── preprocessing.py    필터·대치·인코딩·스케일링 (적용 내역 기록)
    ├── analysis/           기법별 엔진 + 레지스트리
    ├── recommendations.py  차트 추천과 차트용 데이터 집계
    ├── packaging.py        패키지 생성·재현·공유
    └── runner.py           분석 실행 오케스트레이션
```

## 고려 중인 기능

아래 두 기법은 교육 데이터 분석 수요(문항 분석, 잠재변수 인과경로)와 관련이 있으나,
현재는 구현하지 않고 검토 단계에 있습니다.

| 기법 | 제약사항 |
| --- | --- |
| **문항반응이론(IRT)** | 문항 난이도·변별도 분석에 쓰이는 표준 Python 라이브러리가 아직 성숙하지 않습니다(`girth`·`py-irt` 등은 유지보수 활발도가 낮거나 무거운 의존성 필요). 또한 입력 데이터가 "문항 × 응답자" wide-format이라, 이 서비스가 전제하는 "행 = 독립 관측치, target+features" 구조와 데이터 셰이프 자체가 달라 별도 엔드포인트 설계가 필요합니다. |
| **구조방정식모형(SEM)** | 순수 Python SEM 라이브러리는 `semopy`가 사실상 유일한 선택지이며, R의 `lavaan` 대비 기능·안정성이 떨어집니다. 잠재변수·경로모형을 API 파라미터로 어떻게 표현할지(현재의 `target`/`features` 평면 구조로는 표현 불가)부터 새로 설계해야 해서, 현재 지원하는 기법들과 아키텍처 격차가 가장 큽니다. |

우선순위와 구현 가능성 검토는 필요 시 별도로 진행합니다.

## 설정

환경변수는 `USA_` 접두사를 사용합니다 (`.env` 지원).

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `USA_DATABASE_URL` | `sqlite:///storage/app.db` | PostgreSQL 등으로 교체 가능 |
| `USA_STORAGE_DIR` | `./storage` | 데이터셋·차트·패키지 저장 경로 |
| `USA_MAX_UPLOAD_MB` | `200` | 업로드 크기 상한 |
| `USA_DEBUG` | `false` | 오류 응답에 스택트레이스 포함 |
| `USA_CORS_ORIGINS` | `["*"]` | 허용 오리진 |

## 알아둘 점

- 분석은 **동기 실행**됩니다. 대용량 데이터에서 장시간 걸리는 작업은 Celery/ARQ 등
  작업 큐로 옮기는 것을 권장합니다 (`Analysis.status`에 `pending`/`running` 상태가
  이미 마련되어 있습니다).
- 인증·권한은 포함되어 있지 않습니다. 공유 토큰은 URL을 아는 사람이면 접근 가능한
  방식이므로, 실제 운영에서는 기관 인증 체계와 연동하세요.
