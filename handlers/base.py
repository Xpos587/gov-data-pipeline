import aiohttp
import logging
from types import TracebackType
from typing import Optional, Union, Dict, Type, Any
from polars import DataFrame
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

ExcType = Optional[Type[BaseException]]
ExcVal = Optional[BaseException]


class BaseHandler(ABC):
    """
    Базовый класс для хендлеров, работающих с внешними сервисами.
    Предоставляет общую асинхронную логику и структуру.
    """

    IMAGE_COLUMN_NAME: str
    TRADEMARK_COLUMN_NAME: str
    DESCRIPTION_COLUMN_NAME: str
    ROW_OFFSET: int = 0

    def __init__(self) -> None:
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "BaseHandler":
        """
        Инициализирует aiohttp сессию при входе в контекст.
        """
        self.session = aiohttp.ClientSession()
        logger.info("aiohttp ClientSession инициализирована")
        return self

    async def __aexit__(
        self, exc_type: ExcType, exc_val: ExcVal, exc_tb: Optional[TracebackType]
    ) -> None:
        """
        Закрывает aiohttp сессию при выходе из контекста.
        """
        if self.session:
            await self.session.close()
            logger.info("aiohttp ClientSession закрыта")

    async def fetch(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        proxy: Optional[str] = None,
        proxy_auth: Optional[aiohttp.BasicAuth] = None,
        cookies: Optional[Dict[str, str]] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Выполняет GET-запрос к указанному URL с возможностью настройки заголовков, прокси, cookies и User-Agent.

        Parameters:
        -----------
        url : str
            URL для запроса.
        headers : Optional[Dict[str, str]]
            Заголовки HTTP-запроса.
        proxy : Optional[str]
            URL прокси-сервера (например, 'http://proxy.example.com:8080').
        proxy_auth : Optional[aiohttp.BasicAuth]
            Объект авторизации для прокси-сервера.
        cookies : Optional[Dict[str, str]]
            Cookies для запроса.
        user_agent : Optional[str]
            Значение User-Agent для заголовков.

        Returns:
        --------
        Optional[bytes]
            Байты содержимого ответа, если запрос успешен, иначе None.
        """
        if not self.session:
            raise RuntimeError(
                "Сессия не инициализирована. Используйте 'async with' для управления контекстом."
            )

        combined_headers: Dict[str, str] = headers.copy() if headers else {}
        if user_agent:
            combined_headers["User-Agent"] = user_agent

        try:
            async with self.session.get(
                url,
                headers=combined_headers,
                proxy=proxy,
                proxy_auth=proxy_auth,
                cookies=cookies,
            ) as response:
                response.raise_for_status()
                content = await response.read()
                logger.info(f"Запрос к {url} выполнен успешно: {response.status}")
                return content
        except aiohttp.ClientError as e:
            logger.error(f"Ошибка при выполнении запроса к {url}: {e}")
            return None

    async def post(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Union[Dict[str, Any], aiohttp.FormData]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        proxy: Optional[str] = None,
        proxy_auth: Optional[aiohttp.BasicAuth] = None,
        cookies: Optional[Dict[str, str]] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Выполняет POST-запрос к указанному URL с возможностью настройки заголовков, данных, прокси, cookies и User-Agent.

        Parameters:
        -----------
        url : str
            URL для запроса.
        headers : Optional[Dict[str, str]]
            Заголовки HTTP-запроса.
        data : Optional[Union[Dict[str, Any], aiohttp.FormData]]
            Данные для отправки в теле запроса. Может быть словарем или aiohttp.FormData для multipart/form-data.
        json_data : Optional[Dict[str, Any]]
            Данные в формате JSON для отправки в теле запроса.
        proxy : Optional[str]
            URL прокси-сервера (например, 'http://proxy.example.com:8080').
        proxy_auth : Optional[aiohttp.BasicAuth]
            Объект авторизации для прокси-сервера.
        cookies : Optional[Dict[str, str]]
            Cookies для запроса.
        user_agent : Optional[str]
            Значение User-Agent для заголовков.

        Returns:
        --------
        Optional[bytes]
            Байты содержимого ответа, если запрос успешен, иначе None.
        """
        if not self.session:
            raise RuntimeError(
                "Сессия не инициализирована. Используйте 'async with' для управления контекстом."
            )

        combined_headers: Dict[str, str] = headers.copy() if headers else {}
        if user_agent:
            combined_headers["User-Agent"] = user_agent

        try:
            async with self.session.post(
                url,
                headers=combined_headers,
                data=data,
                json=json_data,
                proxy=proxy,
                proxy_auth=proxy_auth,
                cookies=cookies,
            ) as response:
                response.raise_for_status()
                content = await response.read()
                logger.info(f"POST-запрос к {url} выполнен успешно: {response.status}")
                return content
        except aiohttp.ClientError as e:
            logger.error(f"Ошибка при выполнении POST-запроса к {url}: {e}")
            return None

    @abstractmethod
    async def process(
        self,
        user_agent: str,
        proxy: Optional[str] = None,
        proxy_auth: Optional[aiohttp.BasicAuth] = None,
        correction: bool = False,
    ) -> Optional[DataFrame]:
        """
        Абстрактный метод, который должен быть реализован в дочерних классах.

        Parameters
        ----------
        user_agent : Optional[str]
            Заголовок User-Agent, если нужно.
        proxy : Optional[str]
            Адрес прокси-сервера.
        proxy_auth : Optional[aiohttp.BasicAuth]
            Учетные данные для прокси (логин/пароль).
        correction : bool
            Флаг включения/выключения коррекции.

        Returns
        -------
        Optional[DataFrame]
            Обработанные данные (DataFrame) или None, если что-то пошло не так.
        """
        pass
