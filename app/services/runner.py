"""분석 실행 오케스트레이션."""

import time
import traceback
from datetime import UTC, datetime

from sqlmodel import Session

from app.core.config import settings
from app.core.exceptions import AnalysisError, NotFoundError
from app.models.enums import AnalysisStatus
from app.models.tables import Analysis, Dataset
from app.schemas.analysis import AnalysisCreate, ChartDataOptions
from app.services import recommendations as rec_service
from app.services import storage
from app.services.analysis.base import AnalysisOutcome
from app.services.analysis.registry import METHOD_INFO, get_engine


def create_analysis(session: Session, payload: AnalysisCreate) -> Analysis:
    """분석을 동기 실행하고 결과·차트 추천·차트용 데이터를 저장한다.

    Args:
        session (Session): DB 세션.
        payload (AnalysisCreate): 분석 대상 데이터셋, 기법, 파라미터, 전처리·차트 옵션을 담은 요청.

    Raises:
        NotFoundError: 대상 데이터셋을 찾을 수 없는 경우.

    Returns:
        Analysis: 실행 결과가 반영된 분석 레코드. 실행 중 오류가 발생해도 실패 상태로 저장되어 반환된다.
    """
    dataset = session.get(Dataset, payload.dataset_id)
    if dataset is None:
        msg = f"데이터셋을 찾을 수 없습니다: {payload.dataset_id}"
        raise NotFoundError(msg)

    info = METHOD_INFO[payload.method]
    analysis = Analysis(
        dataset_id=dataset.id,
        name=payload.name or f"{info['label']} - {dataset.name}",
        method=payload.method,
        status=AnalysisStatus.RUNNING,
        params=payload.params.model_dump(mode="json"),
        preprocessing=payload.preprocessing.model_dump(mode="json"),
    )
    session.add(analysis)
    session.commit()
    session.refresh(analysis)

    started = time.perf_counter()
    try:
        frame = storage.load_frame(dataset.id)
        outcome = get_engine(payload.method).run(frame, payload.params, payload.preprocessing)
        _persist_success(session, analysis, outcome, payload.chart_data, started)
    except (AnalysisError, FileNotFoundError) as exc:
        _persist_failure(session, analysis, str(exc), started)
    except Exception as exc:  # noqa: BLE001 - 예기치 못한 오류도 분석 레코드에 남긴다
        detail = str(exc) if not settings.debug else traceback.format_exc()
        _persist_failure(session, analysis, f"분석 중 오류가 발생했습니다: {detail}", started)

    session.refresh(analysis)
    return analysis


def _persist_success(
    session: Session,
    analysis: Analysis,
    outcome: AnalysisOutcome,
    chart_options: ChartDataOptions,
    started: float,
) -> None:
    analysis.result = outcome.result
    analysis.metrics = outcome.metrics
    analysis.chart_recommendations = [
        r.model_dump(mode="json")
        for r in rec_service.recommend(
            outcome,
            analysis.method,
        )
    ]
    analysis.chart_data = rec_service.build_chart_data(
        outcome,
        analysis.method,
        chart_options,
    )
    analysis.status = AnalysisStatus.SUCCEEDED
    analysis.duration_ms = int((time.perf_counter() - started) * 1000)
    analysis.completed_at = datetime.now(UTC)
    session.add(analysis)
    session.commit()


def _persist_failure(
    session: Session,
    analysis: Analysis,
    message: str,
    started: float,
) -> None:
    analysis.status = AnalysisStatus.FAILED
    analysis.error = message
    analysis.duration_ms = int((time.perf_counter() - started) * 1000)
    analysis.completed_at = datetime.now(UTC)
    session.add(analysis)
    session.commit()


def rebuild_chart_data(
    session: Session,
    analysis: Analysis,
    options: ChartDataOptions,
) -> Analysis:
    """분석을 다시 실행해 차트용 데이터를 새 옵션(표본 수, 구간 수 등)으로 만든다.

    Args:
        session (Session): DB 세션.
        analysis (Analysis): 차트 데이터를 다시 만들 대상 분석. 성공 상태여야 한다.
        options (ChartDataOptions): 표본 점 개수·히스토그램 구간 수 등 새 차트 데이터 산출 옵션.

    Raises:
        AnalysisError: 대상 분석이 성공 상태가 아닌 경우.

    Returns:
        Analysis: 차트 추천·차트 데이터가 갱신된 분석 레코드.
    """

    if analysis.status is not AnalysisStatus.SUCCEEDED:
        msg = "성공한 분석에 대해서만 차트 데이터를 다시 만들 수 있습니다."
        raise AnalysisError(msg)

    from app.schemas.analysis import AnalysisParams, PreprocessingSpec

    frame = storage.load_frame(analysis.dataset_id)
    params = AnalysisParams.model_validate(analysis.params)
    spec = PreprocessingSpec.model_validate(analysis.preprocessing)
    outcome = get_engine(analysis.method).run(frame, params, spec)

    analysis.chart_recommendations = [
        r.model_dump(mode="json") for r in rec_service.recommend(outcome, analysis.method)
    ]
    analysis.chart_data = rec_service.build_chart_data(outcome, analysis.method, options)
    session.add(analysis)
    session.commit()
    session.refresh(analysis)
    return analysis
