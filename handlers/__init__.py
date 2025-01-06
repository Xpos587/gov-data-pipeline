from typing import Dict, Optional, Type
from .base import BaseHandler
from .belarus import BelarusHandler
from .kazakhstan import KazakhstanHandler
from .kyrgyzstan import KyrgyzstanHandler
from polars import DataFrame
import aiohttp
import logging

logger = logging.getLogger(__name__)


class HandlerConfig:
    """
    Конфигурация для отдельного обработчика.
    """

    def __init__(
        self,
        handler_class: Type[BaseHandler],
        user_agent: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        proxy: Optional[str] = None,
        proxy_auth: Optional[aiohttp.BasicAuth] = None,
        enabled: bool = True,
    ) -> None:
        self.handler_class: Type[BaseHandler] = handler_class
        self.user_agent: Optional[str] = user_agent
        self.headers: Dict[str, str] = headers or {}
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
                result = await handler.process({
                    "headers": config.headers,
                    "proxy": config.proxy,
                    "proxy_auth": config.proxy_auth,
                    "user_agent": config.user_agent,
                })
                results[name] = result
                logger.info(f"Обработчик '{name}' завершил работу.")

        return results

# Конфигурация обработчиков
configs: Dict[str, HandlerConfig] = {
    "belarus": HandlerConfig(
        handler_class=BelarusHandler,
        user_agent="BelarusHandler/1.0",
        headers={},
        # proxy="http://proxy.example.com:8080",
        # proxy_auth=aiohttp.BasicAuth("user", "pass"),
        enabled=True,
    ),
    "kazakhstan": HandlerConfig(
        handler_class=KazakhstanHandler,
        user_agent="KazakhstanHandler/1.0",
        headers={},
        enabled=True,
    ),
    "kyrgyzstan": HandlerConfig(
        handler_class=KyrgyzstanHandler,
        user_agent="KyrgyzstanHandler/1.0",
        headers={},
        enabled=True,
    ),
}

# Экземпляр менеджера обработчиков
handlers_manager: HandlersManager = HandlersManager(configs)

__all__ = ["handlers_manager"]
