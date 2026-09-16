"""애플리케이션 설정."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="USA_", extra="ignore"
    )

    app_name: str = "이용자 통계분석 서비스"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False

    # 저장소
    storage_dir: Path = BASE_DIR / "storage"
    database_url: str = f"sqlite:///{BASE_DIR / 'storage' / 'app.db'}"

    # 반입 제한
    max_upload_mb: int = 200
    max_preview_rows: int = 200
    allowed_extensions: tuple[str, ...] = (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".parquet")

    # 분석 기본값
    default_outlier_method: str = "iqr"
    default_correlation_method: str = "pearson"
    max_categories_for_profile: int = 50

    cors_origins: list[str] = ["*"]

    @property
    def upload_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def dataset_dir(self) -> Path:
        return self.storage_dir / "datasets"

    @property
    def package_dir(self) -> Path:
        return self.storage_dir / "packages"

    def ensure_dirs(self) -> None:
        for path in (
            self.storage_dir,
            self.upload_dir,
            self.dataset_dir,
            self.package_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings = get_settings()
