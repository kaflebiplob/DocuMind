from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "DocuMind Backend"
    ENV: str = "dev"
    LOG_LEVEL: str = "INFO"
    TIMEZONE: str = "UTC"

    DATABASE_URL: str
    DATABASE_POOL_SIZE: int = 20

    class Config:
        env_file = (".env",)
        extra = "ignore"


settings = Settings()
