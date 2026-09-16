"""DB 엔진 및 세션."""

from collections.abc import Generator

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.core.config import settings

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine_kwargs: dict = {"echo": settings.debug, "connect_args": connect_args}
if settings.database_url.endswith(":memory:"):
    engine_kwargs["poolclass"] = StaticPool

engine = create_engine(settings.database_url, **engine_kwargs)


def init_db() -> None:
    """테이블 생성. 모델 모듈을 먼저 임포트해야 메타데이터에 등록된다."""
    import app.models  # noqa: F401

    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session]:
    with Session(engine) as session:
        yield session
