from typing import Optional, Tuple, Dict, Any
from aiohttp import BasicAuth
from pydantic import SecretStr, Field, model_validator
from pydantic_settings import BaseSettings as _BaseSettings
from pydantic_settings import SettingsConfigDict


class BaseSettings(_BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore", env_file=".env", env_file_encoding="utf-8"
    )


class FTPConfig(BaseSettings):
    """Настройки FTP сервера."""

    host: str = "127.0.0.1"
    port: int = 21
    user: str = "sftp"
    passwd: SecretStr = SecretStr("default_pass")
    remote_dir: str = "/"

    class Config:
        env_prefix = "FTP_"  # Префикс для переменных окружения


class OpenAIConfig(BaseSettings):
    """Настройки OpenAI API."""

    base_url: str = "https://api.openai.com/v1"
    api_key: SecretStr = SecretStr("default_key")
    image_model: str = "gpt-4o"
    brand_model: str = "gpt-4o-mini"
    correct_model: str = "gpt-4o-mini"

    class Config:
        env_prefix = "OPENAI_"  # Префикс для переменных окружения


def parse_proxy_string(proxy_value: str) -> Tuple[str, Optional[BasicAuth]]:
    """
    Разбирает строку прокси вида:
       HTTP://IP:PORT@USER:PSWD
    Если символ '@' присутствует, то всё, что после последнего '@'
    считается частью авторизации.
    """
    if "@" in proxy_value:
        # Разбиваем по последнему вхождению '@'
        proxy_without_auth, credentials = proxy_value.rsplit("@", 1)
        if ":" in credentials:
            username, password = credentials.split(":", 1)
        else:
            # Если пароль не указан, оставляем его пустым
            username, password = credentials, ""
        auth = BasicAuth(login=username, password=password)
        return proxy_without_auth, auth
    return proxy_value, None


class ProxyConfigMixin:
    """Миксин для автоматического разбора прокси."""

    proxy: Optional[str] = None
    proxy_auth: Optional[BasicAuth] = None

    @model_validator(mode="before")
    @classmethod
    def split_proxy(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        proxy_value = values.get("proxy")
        if isinstance(proxy_value, str) and proxy_value:
            proxy, auth = parse_proxy_string(proxy_value)
            values["proxy"] = proxy
            values["proxy_auth"] = auth
        return values


class BelarusConfig(BaseSettings, ProxyConfigMixin):
    """Настройки для Беларуси."""

    enabled: bool = True
    user_agent: str = "BelarusHandler/1.0"
    correction: bool = False

    class Config:
        env_prefix = "BELARUS_"  # Префикс для переменных окружения


class KazakhstanConfig(BaseSettings, ProxyConfigMixin):
    """Настройки для Казахстана."""

    enabled: bool = True
    user_agent: str = "KazakhstanHandler/1.0"
    correction: bool = False

    class Config:
        env_prefix = "KAZAKHSTAN_"  # Префикс для переменных окружения


class KyrgyzstanConfig(BaseSettings, ProxyConfigMixin):
    """Настройки для Кыргызстана."""

    enabled: bool = True
    user_agent: str = "KyrgyzstanHandler/1.0"
    correction: bool = False

    class Config:
        env_prefix = "KYRGYZSTAN_"  # Префикс для переменных окружения


class AppSettings(BaseSettings):
    """Основные настройки приложения."""

    ftp: FTPConfig = Field(default_factory=FTPConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    belarus: BelarusConfig = Field(default_factory=BelarusConfig)
    kazakhstan: KazakhstanConfig = Field(default_factory=KazakhstanConfig)
    kyrgyzstan: KyrgyzstanConfig = Field(default_factory=KyrgyzstanConfig)


settings = AppSettings()

if __name__ == "__main__":
  print(settings.model_dump_json(indent=2))
