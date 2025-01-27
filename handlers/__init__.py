import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, Type
from polars import DataFrame
import aiohttp

from .base import BaseHandler
from .belarus import BelarusHandler
from .kazakhstan import KazakhstanHandler
from .kyrgyzstan import KyrgyzstanHandler

logger = logging.getLogger(__name__)

# Сопоставление строковых имен классов с реальными классами обработчиков
HANDLER_CLASS_MAPPING: Dict[str, Type[BaseHandler]] = {
    "BelarusHandler": BelarusHandler,
    "KazakhstanHandler": KazakhstanHandler,
    "KyrgyzstanHandler": KyrgyzstanHandler,
}


class HandlerConfig:
    """
    Конфигурация для отдельного обработчика.
    """

    def __init__(
        self,
        handler_class: Type[BaseHandler],
        user_agent: Optional[str] = None,
        proxy: Optional[str] = None,
        proxy_auth: Optional[aiohttp.BasicAuth] = None,
        enabled: bool = True,
    ):
        self.handler_class: Type[BaseHandler] = handler_class
        self.user_agent: Optional[str] = user_agent
        self.proxy: Optional[str] = proxy
        self.proxy_auth: Optional[aiohttp.BasicAuth] = proxy_auth
        self.enabled: bool = enabled


class HandlersManager:
    """
    Менеджер для управления и запуска обработчиков.
    """

    def __init__(self, configs: Dict[str, HandlerConfig]) -> None:
        """
        Инициализация менеджера обработчиков.

        Args:
            configs (Dict[str, HandlerConfig]): Словарь конфигураций обработчиков.
        """
        self.configs: Dict[str, HandlerConfig] = configs
        self.handlers: Dict[str, BaseHandler] = self._initialize_handlers()

    def _initialize_handlers(self) -> Dict[str, BaseHandler]:
        """
        Инициализация всех обработчиков из конфигураций.
        """
        handlers: Dict[str, BaseHandler] = {}
        for name, config in self.configs.items():
            if config.enabled:
                handler_instance = config.handler_class()
                handlers[name] = handler_instance
                logger.info(f"Обработчик '{name}' инициализирован.")
            else:
                logger.info(f"Обработчик '{name}' отключен.")
        return handlers

    async def process_all(self) -> Dict[str, Optional[DataFrame]]:
        """
        Запускает обработку данных для всех включенных обработчиков.

        Returns:
            Dict[str, Optional[DataFrame]]: Результаты обработки для каждого обработчика.
        """
        results: Dict[str, Optional[DataFrame]] = {}
        for name, handler in self.handlers.items():
            config = self.configs[name]
            async with handler:
                logger.info(f"Запуск обработчика '{name}'...")
                result = await handler.process(
                    {
                        "proxy": config.proxy,
                        "proxy_auth": config.proxy_auth,
                        "user_agent": config.user_agent,
                    }
                )
                results[name] = result
                logger.info(f"Обработчик '{name}' завершил работу.")
        return results


def load_configs_from_file(config_path: str) -> Dict[str, HandlerConfig]:
    """
    Загружает конфигурации обработчиков из YAML файла.

    Args:
        config_path (str): Путь к конфигурационному файлу.

    Returns:
        Dict[str, HandlerConfig]: Словарь конфигураций обработчиков.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if not raw_config or "handlers" not in raw_config:
        raise ValueError(
            "Конфигурационный файл пуст или не содержит секцию 'handlers'."
        )

    configs: Dict[str, HandlerConfig] = {}
    for name, conf in raw_config["handlers"].items():
        handler_class_name = conf.get("handler_class")
        if not handler_class_name:
            raise ValueError(
                f"В конфигурации обработчика '{name}' не указано 'handler_class'."
            )

        handler_class = HANDLER_CLASS_MAPPING.get(handler_class_name)
        if not handler_class:
            raise ValueError(f"Неизвестный класс обработчика: {handler_class_name}")

        # Обработка proxy_auth, если он указан
        proxy_auth_config = conf.get("proxy_auth")
        if proxy_auth_config and isinstance(proxy_auth_config, dict):
            try:
                proxy_auth = aiohttp.BasicAuth(**proxy_auth_config)
            except TypeError as e:
                logger.error(
                    f"Неверная конфигурация proxy_auth для обработчика '{name}': {e}"
                )
                proxy_auth = None
        else:
            proxy_auth = None

        configs[name] = HandlerConfig(
            handler_class=handler_class,
            user_agent=conf.get("user_agent"),
            proxy=conf.get("proxy"),
            proxy_auth=proxy_auth,
            enabled=conf.get("enabled", True),
        )
    return configs


# Загрузка конфигурации обработчиков
CONFIG_FILE = Path(__file__).parent.parent / "config.yml"

# Преобразуем Path в str
configs = load_configs_from_file(str(CONFIG_FILE))

handlers_manager = HandlersManager(configs)

__all__ = ["handlers_manager"]
