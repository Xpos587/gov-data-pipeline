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
    port: int = 21
    user: str = "sftp"
    passwd: SecretStr
    remote_dir: str = "/"


class OpenAIConfig(BaseSettings):
    """Настройки OpenAI API."""

    api_key: SecretStr
    model: str = "gpt-4o-mini"
    max_tokens: int = 768


class AppSettings(BaseSettings):
    """Основные настройки приложения."""

    openai: OpenAIConfig = Field(default=OpenAIConfig(api_key=SecretStr("default_key")))
    sftp: SFTPConfig = Field(default=SFTPConfig(passwd=SecretStr("default_pass")))


settings = AppSettings()
