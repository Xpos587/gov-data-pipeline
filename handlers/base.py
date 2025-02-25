import aiohttp
import logging
from io import BytesIO
from types import TracebackType
from typing import Optional, Union, Dict, Type, Any, Tuple, List
from abc import ABC, abstractmethod
from openpyxl import load_workbook
from utils.gpt import image_to_base64
import polars as pl

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
    ) -> Optional[pl.DataFrame]:
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

    async def process_excel_images(
        self,
        df: pl.DataFrame,
        byte_stream: BytesIO,
    ) -> pl.DataFrame:
        """
        Предобрабатывает изображения в Excel, извлекая Base64-строки и помещая их в новый столбец.
        Оригинальный столбец с текстом остается без изменений.

        :param df: Исходный DataFrame.
        :param byte_stream: Поток байтов Excel-файла.
        :param column_name: Название столбца, по которому определяется расположение строк для изображений.
        :return: Обновленный DataFrame с дополнительным столбцом, содержащим Base64-строки изображений.
        """
        logger.info(
            f"Начинаем предобработку изображений для столбца: {self.IMAGE_COLUMN_NAME}..."
        )

        if self.IMAGE_COLUMN_NAME not in df.columns:
            logger.warning(f"Столбец '{self.IMAGE_COLUMN_NAME}' не найден.")
            return df

        try:
            workbook = load_workbook(byte_stream, data_only=True)
            sheet = workbook.active
        except Exception as e:
            logger.error(f"Ошибка загрузки книги Excel: {e}")
            return df

        images_info: Dict[Tuple[int, int], List[str]] = {}

        for image in getattr(sheet, "_images", []):
            try:
                base_row: int = image.anchor._from.row  # 0-based
                base_col: int = image.anchor._from.col  # 0-based
                row_off: int = image.anchor._from.rowOff
                additional_row: int = 1 if row_off > 10000 else 0

                excel_row: int = base_row + 1 + additional_row
                excel_col: int = base_col + 1

                img_data = BytesIO(image._data())
                base64_image: str = await image_to_base64(img_data.getvalue())

                key: Tuple[int, int] = (excel_row, excel_col)
                images_info.setdefault(key, []).append(base64_image)
            except Exception as e:
                logger.warning(f"Ошибка при обработке изображения: {e}")
                continue

        if not images_info:
            logger.info("Изображений не обнаружено или они отсутствуют.")
            return df

        logger.info(f"Найдено {len(images_info)} встраиваемых изображений.")

        # Создаем новый столбец для Base64-строк изображений.
        new_column: List[str] = ["" for _ in range(df.height)]

        for (excel_row, excel_col), base64_list in images_info.items():
            df_row: int = (
                excel_row - self.ROW_OFFSET - 1
            )  # Преобразуем Excel-индекс в индекс DataFrame
            if df_row < 0 or df_row >= df.height:
                logger.warning(
                    f"Изображение (Excel row={excel_row}, col={excel_col}) -> df_row={df_row} вне диапазона."
                )
                continue
            # В новом столбце только Base64-строки, без дополнительного текста.
            new_text: str = " ".join(base64_list).strip()
            new_column[df_row] = new_text

        df = df.with_columns(pl.Series("Изображение", new_column))
        logger.info("Предобработка изображений завершена.")
        return df
