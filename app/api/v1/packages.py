"""분석 패키지 보존·재현·공유 API."""

from typing import Annotated

from fastapi import APIRouter, Query, Request, status
from fastapi.responses import FileResponse
from sqlmodel import func, select

from app.api.deps import PackageDep, SessionDep
from app.core.exceptions import NotFoundError
from app.models.tables import AnalysisPackage
from app.schemas.common import Message, Page
from app.schemas.package import PackageCreate, PackageRead, ReproduceRequest, ReproduceResult, ShareInfo
from app.services import packaging

router = APIRouter(prefix="/packages", tags=["분석 패키지 (보존·재현·공유)"])


@router.post(
    "",
    response_model=PackageRead,
    status_code=status.HTTP_201_CREATED,
    summary="분석 패키지 생성",
    description="분석 조건·전처리 로직·결과 리포트·차트를 ZIP으로 묶어 보존한다.",
)
def create_package(payload: PackageCreate, session: SessionDep) -> PackageRead:
    package = packaging.create_package(session, payload)
    return PackageRead.model_validate(package, from_attributes=True)


@router.get("", response_model=Page[PackageRead], summary="패키지 목록")
def list_packages(
    session: SessionDep,
    dataset_id: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[PackageRead]:
    statement = select(AnalysisPackage)
    count_statement = select(func.count()).select_from(AnalysisPackage)
    if dataset_id:
        statement = statement.where(AnalysisPackage.dataset_id == dataset_id)
        count_statement = count_statement.where(AnalysisPackage.dataset_id == dataset_id)

    total = session.exec(count_statement).one()
    rows = session.exec(
        statement.order_by(AnalysisPackage.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return Page(
        items=[PackageRead.model_validate(r, from_attributes=True) for r in rows],
        total=int(total), limit=limit, offset=offset,
    )


@router.get("/{package_id}", response_model=PackageRead, summary="패키지 상세")
def get_package_detail(package: PackageDep) -> PackageRead:
    return PackageRead.model_validate(package, from_attributes=True)


@router.get("/{package_id}/download", summary="패키지 ZIP 내려받기")
def download_package(package: PackageDep) -> FileResponse:
    if not package.archive_path:
        raise NotFoundError("패키지 아카이브가 없습니다.")
    return FileResponse(
        package.archive_path, media_type="application/zip", filename=f"{package.name}.zip"
    )


@router.post(
    "/{package_id}/reproduce",
    response_model=ReproduceResult,
    summary="패키지 재현 실행",
    description="보존된 조건과 전처리 로직으로 분석을 재실행하고 원본 지표와 대조한다.",
)
def reproduce_package(
    package: PackageDep, payload: ReproduceRequest, session: SessionDep
) -> ReproduceResult:
    return packaging.reproduce(
        session, package, dataset_id=payload.dataset_id, name_suffix=payload.name_suffix
    )


@router.post("/{package_id}/share", response_model=ShareInfo, summary="공유 링크 발급")
def share_package(package: PackageDep, session: SessionDep, request: Request) -> ShareInfo:
    shared = packaging.issue_share_token(session, package)
    return ShareInfo(
        package_id=shared.id,
        share_token=shared.share_token or "",
        share_url=str(request.base_url).rstrip("/") + f"/api/v1/shared/{shared.share_token}",
    )


@router.delete("/{package_id}/share", response_model=Message, summary="공유 해제")
def unshare_package(package: PackageDep, session: SessionDep) -> Message:
    packaging.revoke_share_token(session, package)
    return Message(message="공유 링크를 해제했습니다.")


@router.delete("/{package_id}", response_model=Message, summary="패키지 삭제")
def delete_package(package: PackageDep, session: SessionDep) -> Message:
    from pathlib import Path

    if package.archive_path:
        Path(package.archive_path).unlink(missing_ok=True)
    session.delete(package)
    session.commit()
    return Message(message=f"패키지 '{package.name}'을(를) 삭제했습니다.")


shared_router = APIRouter(prefix="/shared", tags=["분석 패키지 (보존·재현·공유)"])


def _by_token(session: SessionDep, token: str) -> AnalysisPackage:
    package = session.exec(
        select(AnalysisPackage).where(AnalysisPackage.share_token == token)
    ).first()
    if package is None:
        raise NotFoundError("유효하지 않은 공유 링크입니다.")
    return package


@shared_router.get("/{token}", response_model=PackageRead, summary="공유된 패키지 조회")
def get_shared_package(token: str, session: SessionDep) -> PackageRead:
    return PackageRead.model_validate(_by_token(session, token), from_attributes=True)


@shared_router.get("/{token}/download", summary="공유된 패키지 내려받기")
def download_shared_package(token: str, session: SessionDep) -> FileResponse:
    package = _by_token(session, token)
    if not package.archive_path:
        raise NotFoundError("패키지 아카이브가 없습니다.")
    return FileResponse(
        package.archive_path, media_type="application/zip", filename=f"{package.name}.zip"
    )
