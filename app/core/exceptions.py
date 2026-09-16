"""도메인 예외 및 핸들러."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


class AppError(Exception):
    """모든 도메인 예외가 상속하는 애플리케이션 기본 예외.

    Attributes:
        status_code (int): 이 예외를 HTTP 응답으로 변환할 때 사용할 상태 코드.
        code (str): 클라이언트에 전달되는 오류 식별 코드.
        message (str): 사람이 읽을 수 있는 오류 메시지.
        detail (dict): 오류에 대한 부가 정보.
    """

    status_code = status.HTTP_400_BAD_REQUEST
    code = "app_error"

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class ValidationError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "validation_error"


class IngestError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "ingest_error"


class AnalysisError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "analysis_error"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message, "detail": exc.detail},
        )
