import aiohttp
import logging
from typing import Dict, Optional, Type
from polars import DataFrame
import pkgutil
import importlib
import inspect

from .base import BaseHandler
from utils.settings import settings

logger = logging.getLogger(__name__)


def load_handler_classes() -> Dict[str, Type[BaseHandler]]:
    """
    Динамически загружает все классы-наследники BaseHandler внутри пакета `handlers`.

    Returns:
        Dict[str, Type[BaseHandler]]: Словарь, где ключ — имя класса, а значение — сам класс.
    """
    handlers_dict: Dict[str, Type[BaseHandler]] = {}

    # __path__ — специальная переменная модуля __init__.py,
    # указывающая на директорию пакета
    package_path = __path__
    package_name = __package__  # например, "handlers"

    for module_info in pkgutil.iter_modules(package_path):
        if not module_info.ispkg:
            module_name = module_info.name
            full_module_name = f"{package_name}.{module_name}"

            try:
                module = importlib.import_module(full_module_name)
            except ImportError as e:
                logger.error(
                    f"Не удалось импортировать модуль '{full_module_name}': {e}"
                )
                continue

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseHandler) and obj is not BaseHandler:
                    handlers_dict[name] = obj
                    logger.debug(f"Найден обработчик: {name}")

    return handlers_dict


# Динамическое сопоставление
HANDLER_CLASS_MAPPING: Dict[str, Type[BaseHandler]] = load_handler_classes()


class HandlerConfig:
    """
    Конфигурация для отдельного обработчика.
    """

    def __init__(
        self,
        handler_class: Type[BaseHandler],
        user_agent: str,
        proxy: Optional[str],
        proxy_auth: Optional[aiohttp.BasicAuth],
        enabled: bool,
        correction: bool,
    ) -> None:
        self.handler_class: Type[BaseHandler] = handler_class
        self.user_agent: str = user_agent
        self.proxy: Optional[str] = proxy
        self.proxy_auth: Optional[aiohttp.BasicAuth] = proxy_auth
        self.enabled: bool = enabled
        self.correction: bool = correction


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
                try:
                    handler_instance = config.handler_class()
                    handlers[name] = handler_instance
                    logger.info(f"Обработчик '{name}' инициализирован.")
                except Exception as e:
                    logger.error(f"Ошибка при инициализации обработчика '{name}': {e}")
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
                try:
                    df_result: Optional[DataFrame] = await handler.process(
                        user_agent=config.user_agent,
                        proxy=config.proxy,
                        proxy_auth=config.proxy_auth,
                        correction=config.correction,
                    )
                    results[name] = df_result
                    logger.info(f"Обработчик '{name}' завершил работу.")
                except Exception as e:
                    logger.error(f"Ошибка в обработчике '{name}': {e}")
                    results[name] = None
        return results


def load_configs_from_settings() -> Dict[str, HandlerConfig]:
    """
    Загружает конфигурации обработчиков из `settings`.

    Returns:
        Dict[str, HandlerConfig]: Словарь конфигураций обработчиков.
    """
    configs: Dict[str, HandlerConfig] = {}

    for country, config in {
        "Belarus": settings.belarus,
        "Kazakhstan": settings.kazakhstan,
        "Kyrgyzstan": settings.kyrgyzstan,
    }.items():
        handler_class = HANDLER_CLASS_MAPPING.get(f"{country}Handler")

        if not handler_class:
            logger.warning(f"Не найден обработчик для {country}, пропускаем.")
            continue

        configs[country] = HandlerConfig(
            handler_class=handler_class,
            user_agent=config.user_agent,
            proxy=config.proxy,
            proxy_auth=config.proxy_auth,
            enabled=config.enabled,
            correction=config.correction,
        )
        logger.debug(f"Загружена конфигурация для {country}.")

    return configs


# Загрузка конфигурации обработчиков из `settings.py`
configs = load_configs_from_settings()

handlers_manager = HandlersManager(configs)

__all__ = ["handlers_manager"]
