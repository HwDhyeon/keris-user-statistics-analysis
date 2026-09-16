"""로컬 파일(CSV/Excel 등) 반입 및 변수 프로파일링."""

import io
import math
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api import types as ptypes

from app.core.config import settings
from app.core.exceptions import IngestError
from app.models.enums import ColumnRole
from app.models.tables import Dataset, DatasetColumn
from app.services import storage

_TEXT_FORMATS = {".csv", ".tsv", ".txt"}
_EXCEL_FORMATS = {".xlsx", ".xls"}
_DEFAULT_ENCODINGS = ("utf-8-sig", "cp949", "euc-kr", "latin-1")


def read_any(
    content: bytes,
    filename: str,
    *,
    sheet: str | int | None = None,
    delimiter: str | None = None,
    encoding: str | None = None,
    header_row: int | None = 0,
    na_values: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """업로드된 바이트를 DataFrame으로 읽고, 실제 적용된 반입 옵션을 함께 반환한다."""
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_extensions:
        msg = f"지원하지 않는 파일 형식입니다: {suffix or '(확장자 없음)'}"
        raise IngestError(msg, detail={"allowed": list(settings.allowed_extensions)})

    options: dict[str, Any] = {"format": suffix, "header_row": header_row}
    try:
        if suffix in _TEXT_FORMATS:
            frame, used = _read_text(content, suffix, delimiter, encoding, header_row, na_values)
            options |= used
        elif suffix in _EXCEL_FORMATS:
            frame = pd.read_excel(
                io.BytesIO(content),
                sheet_name=sheet if sheet is not None else 0,
                header=header_row,
                na_values=na_values,
            )
            options["sheet"] = sheet if sheet is not None else 0
        elif suffix == ".json":
            frame = pd.read_json(io.BytesIO(content))
        elif suffix == ".parquet":
            frame = pd.read_parquet(io.BytesIO(content))
        else:  # pragma: no cover - allowed_extensions와 동기화됨
            frame = _unsupported(suffix)
    except IngestError:
        raise
    except Exception as exc:
        msg = f"파일을 읽는 중 오류가 발생했습니다: {exc}"
        raise IngestError(msg) from exc

    if frame.empty:
        raise IngestError("반입된 데이터에 행이 없습니다.")

    frame = _normalise_columns(frame)
    options["n_rows"] = len(frame)
    options["n_cols"] = int(frame.shape[1])
    return frame, options


def _unsupported(suffix: str) -> pd.DataFrame:  # pragma: no cover
    msg = f"처리기가 정의되지 않은 형식입니다: {suffix}"
    raise IngestError(msg)


def _read_text(
    content: bytes,
    suffix: str,
    delimiter: str | None,
    encoding: str | None,
    header_row: int | None,
    na_values: list[str] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """인코딩과 구분자를 자동 판별하며 텍스트 표를 읽는다."""
    encodings = (encoding,) if encoding else _DEFAULT_ENCODINGS
    sep = delimiter if delimiter else ("\t" if suffix == ".tsv" else None)
    last_error: Exception | None = None

    for enc in encodings:
        try:
            frame = pd.read_csv(
                io.BytesIO(content),
                sep=sep,
                engine="python" if sep is None else "c",
                encoding=enc,
                header=header_row,
                na_values=na_values,
                skipinitialspace=True,
            )
        except Exception as exc:  # noqa: BLE001 - 다음 인코딩으로 재시도
            last_error = exc
            continue
        return frame, {"encoding": enc, "delimiter": sep or ","}

    msg = f"인코딩/구분자를 판별하지 못했습니다: {last_error}"
    raise IngestError(msg, detail={"tried_encodings": list(encodings)})


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """열 이름을 문자열로 정리하고 중복 이름에 접미사를 붙인다."""
    seen: dict[str, int] = {}
    names: list[str] = []
    for raw in frame.columns:
        name = str(raw).strip() or "column"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    frame = frame.copy()
    frame.columns = names
    return frame


def infer_role(series: pd.Series) -> ColumnRole:
    """변수의 분석상 역할을 추론한다."""
    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return ColumnRole.CONSTANT
    n_unique = int(non_null.nunique())
    if n_unique <= 1:
        return ColumnRole.CONSTANT
    if ptypes.is_bool_dtype(series):
        return ColumnRole.BOOLEAN
    if ptypes.is_datetime64_any_dtype(series):
        return ColumnRole.DATETIME
    if ptypes.is_numeric_dtype(series):
        # 정수이면서 모든 값이 고유하면 식별자로 본다.
        if n_unique == n and n > 20 and ptypes.is_integer_dtype(series):
            return ColumnRole.IDENTIFIER
        if n_unique <= 2:
            return ColumnRole.BOOLEAN
        return ColumnRole.NUMERIC
    # 객체형: 날짜 문자열인지 확인
    if _looks_like_datetime(non_null):
        return ColumnRole.DATETIME
    if n_unique == n and n > 20:
        return ColumnRole.IDENTIFIER
    if n_unique <= max(settings.max_categories_for_profile, int(n * 0.5)):
        return ColumnRole.CATEGORICAL
    return ColumnRole.TEXT


def _looks_like_datetime(series: pd.Series) -> bool:
    sample = series.astype(str).head(50)
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() > 0.9)


def build_column_metadata(dataset_id: str, frame: pd.DataFrame) -> list[DatasetColumn]:
    """열별 메타데이터와 요약 통계를 생성한다."""
    total = len(frame)
    columns: list[DatasetColumn] = []

    for position, name in enumerate(frame.columns):
        series = frame[name]
        role = infer_role(series)
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))

        categories: list[Any] | None = None
        if role in {ColumnRole.CATEGORICAL, ColumnRole.BOOLEAN} and n_unique <= settings.max_categories_for_profile:
            # 원본 등장 순서를 그대로 보존한다 (변수 값 순서 유지 요구사항).
            categories = [_clean(v) for v in series.dropna().unique().tolist()]

        stats: dict[str, Any] = {}
        if role is ColumnRole.NUMERIC:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                stats = {
                    "mean": _clean(numeric.mean()),
                    "std": _clean(numeric.std(ddof=1)),
                    "min": _clean(numeric.min()),
                    "q1": _clean(numeric.quantile(0.25)),
                    "median": _clean(numeric.median()),
                    "q3": _clean(numeric.quantile(0.75)),
                    "max": _clean(numeric.max()),
                }
        elif role in {ColumnRole.CATEGORICAL, ColumnRole.BOOLEAN, ColumnRole.TEXT}:
            counts = series.value_counts(dropna=True).head(10)
            stats = {"top_values": [{"value": _clean(k), "count": int(v)} for k, v in counts.items()]}
        elif role is ColumnRole.DATETIME:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            if parsed.notna().any():
                stats = {"min": str(parsed.min()), "max": str(parsed.max())}

        columns.append(
            DatasetColumn(
                dataset_id=dataset_id,
                name=str(name),
                position=position,
                dtype=str(series.dtype),
                role=role,
                n_missing=n_missing,
                missing_ratio=round(n_missing / total, 6) if total else 0.0,
                n_unique=n_unique,
                categories=categories,
                stats=stats,
            )
        )
    return columns


def ingest_file(
    *,
    content: bytes,
    filename: str,
    name: str | None = None,
    description: str | None = None,
    sheet: str | int | None = None,
    delimiter: str | None = None,
    encoding: str | None = None,
    header_row: int | None = 0,
    na_values: list[str] | None = None,
) -> tuple[Dataset, list[DatasetColumn]]:
    """파일을 반입해 Dataset + 열 메타데이터를 만든다 (DB 커밋은 호출자 책임)."""
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        msg = f"파일이 너무 큽니다. 최대 {settings.max_upload_mb}MB까지 허용됩니다."
        raise IngestError(msg, detail={"size_bytes": len(content)})

    frame, options = read_any(
        content,
        filename,
        sheet=sheet,
        delimiter=delimiter,
        encoding=encoding,
        header_row=header_row,
        na_values=na_values,
    )

    dataset = Dataset(
        name=name or Path(filename).stem,
        description=description,
        original_filename=filename,
        source_format=Path(filename).suffix.lower().lstrip("."),
        storage_path="",
        checksum=storage.compute_checksum(content),
        size_bytes=len(content),
        n_rows=len(frame),
        n_cols=int(frame.shape[1]),
        ingest_options=options,
    )
    path = storage.save_frame(dataset.id, frame)
    dataset.storage_path = str(path)
    columns = build_column_metadata(dataset.id, frame)
    return dataset, columns


def _clean(value: Any) -> Any:
    """JSON 직렬화 가능한 파이썬 기본형으로 변환한다."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
