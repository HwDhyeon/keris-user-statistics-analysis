"""API 공통 의존성."""

from typing import Annotated

from fastapi import Depends, Path
from sqlmodel import Session

from app.core.database import get_session
from app.core.exceptions import NotFoundError
from app.models.tables import Analysis, AnalysisPackage, Dataset

SessionDep = Annotated[Session, Depends(get_session)]


def get_dataset(dataset_id: Annotated[str, Path()], session: SessionDep) -> Dataset:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        msg = f"데이터셋을 찾을 수 없습니다: {dataset_id}"
        raise NotFoundError(msg)
    return dataset


def get_analysis(analysis_id: Annotated[str, Path()], session: SessionDep) -> Analysis:
    analysis = session.get(Analysis, analysis_id)
    if analysis is None:
        msg = f"분석을 찾을 수 없습니다: {analysis_id}"
        raise NotFoundError(msg)
    return analysis


def get_package(package_id: Annotated[str, Path()], session: SessionDep) -> AnalysisPackage:
    package = session.get(AnalysisPackage, package_id)
    if package is None:
        msg = f"패키지를 찾을 수 없습니다: {package_id}"
        raise NotFoundError(msg)
    return package


DatasetDep = Annotated[Dataset, Depends(get_dataset)]
AnalysisDep = Annotated[Analysis, Depends(get_analysis)]
PackageDep = Annotated[AnalysisPackage, Depends(get_package)]
