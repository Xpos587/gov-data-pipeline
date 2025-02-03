import aiohttp
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, Type, Any
from polars import DataFrame
import pkgutil
import importlib
import inspect

from .base import BaseHandler

logger = logging.getLogger(__name__)


def load_handler_classes() -> Dict[str, Type[BaseHandler]]:
    """
    Динамически загружает все классы-наследники BaseHandler внутри пакета `handlers`.

    Returns
    -------
    Dict[str, Type[BaseHandler]]
        Словарь, где ключ — имя класса, а значение — сам класс.
    """
    handlers_dict: Dict[str, Type[BaseHandler]] = {}

    # __path__ — специальная переменная модуля __init__.py,
    # указывающая на директорию пакета
    package_path = __path__
    package_name = __package__  # например, "handlers"

    for module_info in pkgutil.iter_modules(package_path):
        # Пропускаем пакеты (ispkg==True), если они есть
        if not module_info.ispkg:
            module_name = module_info.name
            full_module_name = f"{package_name}.{module_name}"

            try:
                # Импортируем модуль
                module = importlib.import_module(full_module_name)
            except ImportError as e:
                logger.error(
                    f"Не удалось импортировать модуль '{full_module_name}': {e}"
                )
                continue

            # Перебираем все объекты внутри модуля
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Проверяем, является ли obj наследником BaseHandler (и не самим BaseHandler)
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
        user_agent: str = "DefaultUserAgent",
        proxy: Optional[str] = None,
        proxy_auth: Optional[aiohttp.BasicAuth] = None,
        enabled: bool = True,
        correction: bool = False,
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


def load_configs_from_file(config_path: str) -> Dict[str, HandlerConfig]:
    """
    Загружает конфигурации обработчиков из YAML файла.

    Args:
        config_path (str): Путь к конфигурационному файлу.

    Returns:
        Dict[str, HandlerConfig]: Словарь конфигураций обработчиков.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            raw_config: Dict[str, Any] = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Файл конфигурации не найден: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Ошибка при разборе YAML файла: {e}")
        raise

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

        # Обработка proxy_auth
        proxy_auth_config: Optional[Dict[str, str]] = conf.get("proxy_auth")
        proxy_auth: Optional[str] = None
        if isinstance(proxy_auth_config, dict):
            login: str = proxy_auth_config.get("login", "")
            password: str = proxy_auth_config.get("password", "")
            if login and password:
                proxy_auth = f"{login}:{password}"

        # Обработка user_agent, если он не указан, устанавливаем значение по умолчанию
        user_agent = conf.get("user_agent")
        if not isinstance(user_agent, str):
            logger.warning(
                f"user_agent для обработчика '{name}' должен быть строкой. Используется значение по умолчанию."
            )
            user_agent = "DefaultUserAgent"

        # Обработка enabled и correction с типом bool
        enabled = bool(conf.get("enabled", True))
        correction = bool(conf.get("correction", False))

        # Обработка proxy, если она указана
        proxy = conf.get("proxy")
        if proxy is not None and not isinstance(proxy, str):
            logger.warning(
                f"proxy для обработчика '{name}' должен быть строкой. Игнорируется."
            )
            proxy = None

        configs[name] = HandlerConfig(
            handler_class=handler_class,
            user_agent=user_agent,
            proxy=proxy,
            proxy_auth=proxy_auth,
            enabled=enabled,
            correction=correction,
        )
        logger.debug(f"Загружена конфигурация для обработчика '{name}'.")

    return configs


# Загрузка конфигурации обработчиков
CONFIG_FILE = Path(__file__).parent.parent / "config.yml"

# Преобразуем Path в str
configs = load_configs_from_file(str(CONFIG_FILE))

handlers_manager = HandlersManager(configs)

__all__ = ["handlers_manager"]
