from pydantic_settings import BaseSettings as _BaseSettings
from pydantic_settings import SettingsConfigDict
from pydantic import SecretStr, Field


class BaseSettings(_BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore", env_file=".env", env_file_encoding="utf-8"
    )


class SFTPConfig(BaseSettings):
    """Настройки SFTP сервера."""

    host: str = "127.0.0.1"
    port: int = 22
    user: str = "sftp"
    passwd: SecretStr = SecretStr("default_pass")
    remote_dir: str = "/"

    class Config:
        env_prefix = "SFTP_"  # Префикс для переменных окружения


class OpenAIConfig(BaseSettings):
    """Настройки OpenAI API."""

    base_url: str = "https://api.openai.com/v1"
    api_key: SecretStr = SecretStr("default_key")

    class Config:
        env_prefix = "OPENAI_"  # Префикс для переменных окружения


class AppSettings(BaseSettings):
    """Основные настройки приложения."""

    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    sftp: SFTPConfig = Field(default_factory=SFTPConfig)


settings = AppSettings()
