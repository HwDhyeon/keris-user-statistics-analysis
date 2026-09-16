"""데이터셋 반입·조회 API."""

import json
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Query, UploadFile, status
from sqlmodel import func, select

from app.api.deps import DatasetDep, SessionDep
from app.core.config import settings
from app.core.exceptions import IngestError
from app.models.tables import Dataset, DatasetColumn
from app.schemas.common import Message, Page
from app.schemas.dataset import DatasetDetail, DatasetPreview, DatasetRead, DatasetUpdate
from app.services import ingest, storage
from app.services.ingest import _clean

router = APIRouter(prefix="/datasets", tags=["데이터셋 반입"])


@router.post(
    "/upload",
    response_model=DatasetDetail,
    status_code=status.HTTP_201_CREATED,
    summary="로컬 파일 반입",
    description="CSV/TSV/Excel/JSON/Parquet 파일을 업로드해 즉시 분석 가능한 데이터셋으로 등록한다.",
)
async def upload_dataset(
    session: SessionDep,
    file: Annotated[UploadFile, File(description="업로드할 데이터 파일")],
    name: Annotated[str | None, Form()] = None,
    description: Annotated[str | None, Form()] = None,
    sheet: Annotated[str | None, Form(description="Excel 시트명 또는 인덱스")] = None,
    delimiter: Annotated[str | None, Form(description="CSV 구분자. 미지정 시 자동 추론")] = None,
    encoding: Annotated[str | None, Form(description="미지정 시 utf-8/cp949 순으로 자동 시도")] = None,
    header_row: Annotated[int | None, Form(description="헤더 행 인덱스. 헤더가 없으면 비워둔다")] = 0,
    na_values: Annotated[str | None, Form(description="결측 처리할 값 목록(JSON 배열)")] = None,
) -> DatasetDetail:
    content = await file.read()
    if not content:
        raise IngestError("빈 파일입니다.")

    parsed_na = _parse_na_values(na_values)
    dataset, columns = ingest.ingest_file(
        content=content,
        filename=file.filename or "upload",
        name=name,
        description=description,
        sheet=_parse_sheet(sheet),
        delimiter=delimiter,
        encoding=encoding,
        header_row=header_row,
        na_values=parsed_na,
    )
    session.add(dataset)
    for column in columns:
        session.add(column)
    session.commit()
    session.refresh(dataset)
    return DatasetDetail.model_validate(dataset, from_attributes=True)


def _parse_sheet(sheet: str | None) -> str | int | None:
    if sheet is None or sheet == "":
        return None
    return int(sheet) if sheet.isdigit() else sheet


def _parse_na_values(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [v.strip() for v in raw.split(",") if v.strip()]
    return [str(v) for v in parsed] if isinstance(parsed, list) else None


@router.get("", response_model=Page[DatasetRead], summary="데이터셋 목록")
def list_datasets(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    q: Annotated[str | None, Query(description="이름 검색어")] = None,
) -> Page[DatasetRead]:
    statement = select(Dataset)
    count_statement = select(func.count()).select_from(Dataset)
    if q:
        statement = statement.where(Dataset.name.contains(q))
        count_statement = count_statement.where(Dataset.name.contains(q))

    total = session.exec(count_statement).one()
    rows = session.exec(
        statement.order_by(Dataset.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return Page(
        items=[DatasetRead.model_validate(r, from_attributes=True) for r in rows],
        total=int(total), limit=limit, offset=offset,
    )


@router.get("/{dataset_id}", response_model=DatasetDetail, summary="데이터셋 상세")
def get_dataset_detail(dataset: DatasetDep) -> DatasetDetail:
    return DatasetDetail.model_validate(dataset, from_attributes=True)


@router.patch("/{dataset_id}", response_model=DatasetDetail, summary="데이터셋 정보 수정")
def update_dataset(dataset: DatasetDep, payload: DatasetUpdate, session: SessionDep) -> DatasetDetail:
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(dataset, key, value)
    session.add(dataset)
    session.commit()
    session.refresh(dataset)
    return DatasetDetail.model_validate(dataset, from_attributes=True)


@router.get("/{dataset_id}/preview", response_model=DatasetPreview, summary="데이터 미리보기")
def preview_dataset(
    dataset: DatasetDep,
    limit: Annotated[int, Query(ge=1, le=1000)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    columns: Annotated[list[str] | None, Query(description="조회할 열 이름")] = None,
) -> DatasetPreview:
    frame = storage.load_frame(dataset.id, columns=columns)
    window = frame.iloc[offset : offset + limit]
    rows: list[dict[str, Any]] = [
        {str(k): _clean(v) for k, v in record.items()} for record in window.to_dict("records")
    ]
    return DatasetPreview(
        dataset_id=dataset.id,
        columns=[str(c) for c in frame.columns],
        rows=rows,
        total_rows=len(frame),
        returned_rows=len(rows),
    )


@router.get("/{dataset_id}/columns", summary="변수 목록")
def list_columns(dataset: DatasetDep, session: SessionDep) -> list[dict[str, Any]]:
    columns = session.exec(
        select(DatasetColumn)
        .where(DatasetColumn.dataset_id == dataset.id)
        .order_by(DatasetColumn.position)
    ).all()
    return [
        {
            "name": c.name, "position": c.position, "dtype": c.dtype, "role": c.role.value,
            "n_missing": c.n_missing, "missing_ratio": c.missing_ratio, "n_unique": c.n_unique,
            "categories": c.categories, "stats": c.stats,
        }
        for c in columns
    ]


@router.delete("/{dataset_id}", response_model=Message, summary="데이터셋 삭제")
def delete_dataset(dataset: DatasetDep, session: SessionDep) -> Message:
    storage.delete_frame(dataset.id)
    session.delete(dataset)
    session.commit()
    return Message(message=f"데이터셋 '{dataset.name}'을(를) 삭제했습니다.")


@router.get("/meta/limits", summary="반입 제한 조회", tags=["데이터셋 반입"])
def ingest_limits() -> dict[str, Any]:
    return {
        "max_upload_mb": settings.max_upload_mb,
        "allowed_extensions": list(settings.allowed_extensions),
        "max_preview_rows": settings.max_preview_rows,
    }
