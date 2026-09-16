"""분석 패키지 생성·재현·공유.

패키지는 분석 조건, 전처리 로직, 결과 리포트, 차트를 하나의 ZIP으로 묶어
향후 재현과 공유가 가능하도록 보존한다.
"""

import json
import platform
import secrets
import sys
import zipfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from app.core.config import settings
from app.core.exceptions import AnalysisError, NotFoundError
from app.models.enums import AnalysisStatus
from app.models.tables import Analysis, AnalysisPackage, Dataset, DatasetColumn
from app.schemas.analysis import AnalysisCreate, AnalysisParams, ChartDataOptions, PreprocessingSpec
from app.schemas.package import PackageCreate, ReproduceResult
from app.services import storage

MANIFEST_VERSION = "1.0"
_TRACKED = ("pandas", "numpy", "scikit-learn", "statsmodels", "scipy", "fastapi")


def _environment() -> dict[str, Any]:
    """재현 시 결과 차이를 설명할 수 있도록 실행 환경을 기록한다.

    Returns:
        dict[str, Any]: 파이썬 버전, 플랫폼, 주요 라이브러리 버전 정보.
    """
    packages: dict[str, str] = {}
    for name in _TRACKED:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover
            packages[name] = "unknown"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }


def build_manifest(dataset: Dataset, columns: list[DatasetColumn], analyses: list[Analysis]) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "environment": _environment(),
        "dataset": {
            "id": dataset.id,
            "name": dataset.name,
            "original_filename": dataset.original_filename,
            "source_format": dataset.source_format,
            "checksum": dataset.checksum,
            "n_rows": dataset.n_rows,
            "n_cols": dataset.n_cols,
            "ingest_options": dataset.ingest_options,
            "columns": [
                {
                    "name": c.name, "position": c.position, "dtype": c.dtype, "role": c.role.value,
                    "n_missing": c.n_missing, "n_unique": c.n_unique, "categories": c.categories,
                }
                for c in sorted(columns, key=lambda c: c.position)
            ],
        },
        "analyses": [
            {
                "id": a.id,
                "name": a.name,
                "method": a.method.value,
                # 재현에 필요한 3요소: 분석 조건 / 전처리 로직 / 산출 지표
                "params": a.params,
                "preprocessing": a.preprocessing,
                "metrics": a.metrics,
                "created_at": a.created_at.isoformat(),
                "duration_ms": a.duration_ms,
                # 차트는 클라이언트가 그리므로 추천 내역만 보존한다.
                "chart_recommendations": a.chart_recommendations,
            }
            for a in analyses
        ],
    }


def create_package(session: Session, payload: PackageCreate) -> AnalysisPackage:
    analyses = list(
        session.exec(select(Analysis).where(Analysis.id.in_(payload.analysis_ids))).all()
    )
    found = {a.id for a in analyses}
    missing = [i for i in payload.analysis_ids if i not in found]
    if missing:
        msg = f"분석을 찾을 수 없습니다: {', '.join(missing)}"
        raise NotFoundError(msg)

    failed = [a.name for a in analyses if a.status is not AnalysisStatus.SUCCEEDED]
    if failed:
        msg = f"성공하지 않은 분석은 패키징할 수 없습니다: {', '.join(failed)}"
        raise AnalysisError(msg)

    dataset_ids = {a.dataset_id for a in analyses}
    if len(dataset_ids) > 1:
        msg = "하나의 패키지에는 동일한 데이터셋의 분석만 담을 수 있습니다."
        raise AnalysisError(msg)

    dataset_id = dataset_ids.pop()
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        msg = f"데이터셋을 찾을 수 없습니다: {dataset_id}"
        raise NotFoundError(msg)

    columns = list(session.exec(select(DatasetColumn).where(DatasetColumn.dataset_id == dataset_id)).all())
    # 요청 순서를 보존한다.
    order = {aid: i for i, aid in enumerate(payload.analysis_ids)}
    analyses.sort(key=lambda a: order[a.id])

    manifest = build_manifest(dataset, columns, analyses)
    package = AnalysisPackage(
        name=payload.name,
        description=payload.description,
        dataset_id=dataset_id,
        analysis_ids=[a.id for a in analyses],
        manifest=manifest,
    )

    archive = _write_archive(package, dataset, analyses, payload)
    package.archive_path = str(archive)
    package.archive_bytes = archive.stat().st_size

    session.add(package)
    session.commit()
    session.refresh(package)
    return package


def _write_archive(
    package: AnalysisPackage, dataset: Dataset, analyses: list[Analysis], payload: PackageCreate
) -> Path:
    path = settings.package_dir / f"{package.id}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", _dump(package.manifest))
        zf.writestr("README.md", _readme(package, dataset, analyses))

        for analysis in analyses:
            base = f"analyses/{analysis.id}"
            zf.writestr(f"{base}/analysis.json", _dump({
                "id": analysis.id,
                "name": analysis.name,
                "method": analysis.method.value,
                "params": analysis.params,
                "preprocessing": analysis.preprocessing,
                "metrics": analysis.metrics,
                "result": analysis.result,
                "chart_recommendations": analysis.chart_recommendations,
            }))
            zf.writestr(f"{base}/report.md", _report(analysis))
            if payload.include_chart_data:
                zf.writestr(
                    f"{base}/chart-recommendations.json", _dump(analysis.chart_recommendations)
                )
                zf.writestr(f"{base}/chart-data.json", _dump(analysis.chart_data))

        if payload.include_data_snapshot:
            source = storage.dataset_path(dataset.id)
            if source.exists():
                zf.write(source, f"data/{dataset.name}.parquet")
    return path


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _readme(package: AnalysisPackage, dataset: Dataset, analyses: list[Analysis]) -> str:
    lines = [
        f"# {package.name}",
        "",
        package.description or "",
        "",
        "## 데이터셋",
        f"- 이름: {dataset.name}",
        f"- 원본 파일: {dataset.original_filename}",
        f"- 규모: {dataset.n_rows:,}행 × {dataset.n_cols}열",
        f"- 체크섬(SHA-256): `{dataset.checksum}`",
        "",
        "## 포함된 분석",
    ]
    for analysis in analyses:
        metrics = ", ".join(f"{k}={v}" for k, v in list(analysis.metrics.items())[:4])
        lines.append(f"- **{analysis.name}** ({analysis.method.value}) — {metrics}")
    lines += [
        "",
        "## 재현 방법",
        "1. `manifest.json`의 `dataset.checksum`이 동일한 파일을 반입합니다.",
        "2. `POST /api/v1/packages/{package_id}/reproduce`를 호출하면",
        "   기록된 조건과 전처리 로직으로 분석이 재실행됩니다.",
        "3. 재현 결과의 지표가 `analyses[].metrics`와 일치하는지 응답의 `comparisons`에서 확인합니다.",
        "",
        f"패키지 ID: `{package.id}`",
    ]
    return "\n".join(lines)


def _report(analysis: Analysis) -> str:
    """사람이 읽는 분석 리포트.

    Args:
        analysis (Analysis): 리포트를 생성할 분석 객체.

    Returns:
        str: 분석 개요·성능 지표·전처리 로직·분석 조건·계수를 담은 마크다운 문자열.
    """
    lines = [
        f"# {analysis.name}",
        "",
        f"- 분석 기법: {analysis.result.get('method', analysis.method.value)}",
        f"- 수행 시각: {analysis.created_at.isoformat()}",
        f"- 소요 시간: {analysis.duration_ms}ms",
        "",
        "## 모델 성능 지표",
    ]
    lines += [f"- {k}: {v}" for k, v in analysis.metrics.items()] or ["- (없음)"]

    steps = analysis.result.get("preprocessing_steps") or []
    lines += ["", "## 전처리 로직"]
    lines += [f"{i}. `{json.dumps(s, ensure_ascii=False)}`" for i, s in enumerate(steps, 1)] or ["- (없음)"]

    lines += ["", "## 분석 조건", "```json", _dump(analysis.params), "```"]

    if coefficients := analysis.result.get("coefficients"):
        lines += ["", "## 계수", "| 항 | 계수 | 표준오차 | p-value |", "| --- | ---: | ---: | ---: |"]
        for row in coefficients:
            if not isinstance(row, dict) or "term" not in row:
                continue
            lines.append(
                f"| {row['term']} | {row.get('coefficient')} | {row.get('std_error')} | {row.get('p_value')} |"
            )
    return "\n".join(lines)


def issue_share_token(session: Session, package: AnalysisPackage) -> AnalysisPackage:
    if not package.share_token:
        package.share_token = secrets.token_urlsafe(24)
        session.add(package)
        session.commit()
        session.refresh(package)
    return package


def revoke_share_token(session: Session, package: AnalysisPackage) -> AnalysisPackage:
    package.share_token = None
    session.add(package)
    session.commit()
    session.refresh(package)
    return package


def reproduce(
    session: Session, package: AnalysisPackage, *, dataset_id: str | None, name_suffix: str
) -> ReproduceResult:
    """패키지에 보존된 조건으로 분석을 재실행하고 원본 지표와 대조한다.

    Args:
        session (Session): DB 세션.
        package (AnalysisPackage): 재현할 대상 패키지.
        dataset_id (str | None): 재현에 사용할 데이터셋 식별자. 미지정 시 패키지에 기록된 원본 데이터셋을 사용.
        name_suffix (str): 재현된 분석 이름에 덧붙일 접미사.

    Raises:
        AnalysisError: 재현할 데이터셋을 지정할 수 없는 경우(dataset_id와 패키지 기록 모두 없음).
        NotFoundError: 지정된 데이터셋을 찾을 수 없는 경우.

    Returns:
        ReproduceResult: 재현된 분석 ID 목록과 원본 지표와의 비교 결과.
    """
    from app.services.runner import create_analysis

    target_id = dataset_id or package.dataset_id
    if target_id is None:
        raise AnalysisError("재현할 데이터셋을 지정하세요.")

    dataset = session.get(Dataset, target_id)
    if dataset is None:
        msg = f"데이터셋을 찾을 수 없습니다: {target_id}"
        raise NotFoundError(msg)

    recorded_checksum = package.manifest.get("dataset", {}).get("checksum")
    comparisons: list[dict[str, Any]] = []
    new_ids: list[str] = []
    all_matched = dataset.checksum == recorded_checksum

    for entry in package.manifest.get("analyses", []):
        payload = AnalysisCreate(
            dataset_id=dataset.id,
            method=entry["method"],
            name=f"{entry['name']}{name_suffix}",
            params=AnalysisParams.model_validate(entry["params"]),
            preprocessing=PreprocessingSpec.model_validate(entry["preprocessing"]),
            chart_data=ChartDataOptions(include=False),
        )
        analysis = create_analysis(session, payload)
        new_ids.append(analysis.id)

        original = entry.get("metrics", {})
        diffs = _compare_metrics(original, analysis.metrics)
        matched = analysis.status is AnalysisStatus.SUCCEEDED and not diffs
        all_matched = all_matched and matched
        comparisons.append(
            {
                "original_analysis_id": entry["id"],
                "new_analysis_id": analysis.id,
                "method": entry["method"],
                "status": analysis.status.value,
                "matched": matched,
                "differences": diffs,
                "error": analysis.error,
            }
        )

    if dataset.checksum != recorded_checksum:
        comparisons.append(
            {
                "note": "데이터셋 체크섬이 패키지 기록과 다릅니다. 결과 차이는 데이터 변경 때문일 수 있습니다.",
                "recorded_checksum": recorded_checksum,
                "current_checksum": dataset.checksum,
            }
        )

    return ReproduceResult(
        package_id=package.id,
        dataset_id=dataset.id,
        analysis_ids=new_ids,
        matched=all_matched,
        comparisons=comparisons,
    )


def _compare_metrics(original: dict[str, Any], current: dict[str, Any], tol: float = 1e-6) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    for key, old in original.items():
        new = current.get(key)
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            if abs(float(old) - float(new)) > tol:
                diffs.append({"metric": key, "original": old, "reproduced": new})
        elif old != new:
            diffs.append({"metric": key, "original": old, "reproduced": new})
    return diffs
