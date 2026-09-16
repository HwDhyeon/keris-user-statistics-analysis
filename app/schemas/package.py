"""분석 패키지(보존·재현·공유) 스키마."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PackageCreate(BaseModel):
    name: str
    description: str | None = None
    analysis_ids: list[str] = Field(min_length=1)
    include_data_snapshot: bool = Field(default=False, description="원본 데이터 사본 포함 여부")
    include_chart_data: bool = Field(default=True, description="차트 추천·차트용 데이터 포함 여부")


class PackageRead(BaseModel):
    id: str
    name: str
    description: str | None = None
    dataset_id: str | None = None
    analysis_ids: list[str]
    manifest: dict[str, Any]
    archive_bytes: int
    share_token: str | None = None
    created_at: datetime


class ShareInfo(BaseModel):
    package_id: str
    share_token: str
    share_url: str


class ReproduceRequest(BaseModel):
    dataset_id: str | None = Field(
        default=None, description="미지정 시 패키지에 기록된 원본 데이터셋으로 재현"
    )
    name_suffix: str = " (재현)"


class ReproduceResult(BaseModel):
    package_id: str
    dataset_id: str
    analysis_ids: list[str]
    matched: bool = Field(description="원본 결과 지표와 일치하는지 여부")
    comparisons: list[dict[str, Any]] = Field(default_factory=list)
