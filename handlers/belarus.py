import logging
import re
import aiohttp
from typing import Optional, Match
from io import BytesIO
from urllib.parse import urlparse

import polars as pl
from polars import DataFrame

from .base import BaseHandler
from utils.gpt import process_table

logger = logging.getLogger(__name__)


class BelarusHandler(BaseHandler):
    """
    Асинхронный хендлер для получения данных с сайта Беларуси.
    """

    IMAGE_COLUMN_NAME: str = "Вид объекта интеллектуальной собственности, его наименование (описание, изображение)"  # Название столбца, где потенциально может быть ТМ с изображением
    TRADEMARK_COLUMN_NAME: str = "Вид объекта интеллектуальной собственности, его наименование (описание, изображение)"
    DESCRIPTION_COLUMN_NAME: str = "Наименование (описание) товаров, содержащих объект интеллектуальной собственности"
    ROW_OFFSET: int = 2

    async def retrieve(
        self,
        user_agent: Optional[str],
        proxy: Optional[str] = None,
        proxy_auth: Optional[aiohttp.BasicAuth] = None,
    ) -> Optional[bytes]:
        """
        Загружает страницу, извлекает ссылку на файл с помощью регулярного выражения и возвращает содержимое файла.
        """
        page_url: str = "https://www.customs.gov.by/zashchita-prav-na-obekty-intellektualnoy-sobstvennosti"

        logger.info(f"Запрашиваем страницу: {page_url}")

        page_content: Optional[bytes] = await self.fetch(
            url=page_url,
            proxy=proxy,
            proxy_auth=proxy_auth,
            user_agent=user_agent,
        )

        if page_content is None:
            logger.error("Не удалось загрузить страницу для поиска файла.")
            return None

        try:
            page_text: str = page_content.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.error(f"Ошибка декодирования страницы: {e}")
            return None

        # Ищем ссылку на .xlsx
        pattern: str = r"/uploads/reestr-is/Reestr\d{6}\.xlsx"
        match: Optional[Match[str]] = re.search(pattern, page_text)

        if not match:
            logger.warning("Ссылка на файл не найдена на странице.")
            return None

        parsed_url = urlparse(page_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        file_url: str = base_url + match.group(0)
        logger.info(f"Найдена ссылка на файл: {file_url}")

        # Загружаем Excel-файл
        file_content: Optional[bytes] = await self.fetch(
            url=file_url,
            headers={
                "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            },
            proxy=proxy,
            proxy_auth=proxy_auth,
            user_agent=user_agent,
        )

        if file_content is not None:
            logger.info(f"Файл успешно загружен: {file_url}")
        else:
            logger.error(f"Не удалось загрузить файл по ссылке: {file_url}")

        return file_content

    async def process(
        self,
        user_agent: str,
        proxy: Optional[str] = None,
        proxy_auth: Optional[aiohttp.BasicAuth] = None,
        correction: bool = False,
    ) -> Optional[DataFrame]:
        """
        Получает последний доступный файл и обрабатывает его с помощью polars и GPT.
        """
        data_bytes: Optional[bytes] = await self.retrieve(user_agent, proxy, proxy_auth)
        if data_bytes is None:
            logger.error("Не удалось получить данные последнего файла.")
            return None

        byte_stream: BytesIO = BytesIO(data_bytes)
        # Чтение Excel
        df: DataFrame = pl.read_excel(
            byte_stream,
            engine="calamine",
            read_options={"skip_rows": 1},  # При необходимости корректируем
        )
        # Считываем первую строку как будущие заголовки
        new_columns = [str(val) if val is not None else "UNKNOWN" for val in df.row(0)]

        # Применяем заголовки и убираем первую строку
        df = df.slice(2).rename({old: new for old, new in zip(df.columns, new_columns)})

        # Удаляем лишние пробелы в строковых столбцах
        string_columns = [col for col, dtype in df.schema.items() if dtype == pl.Utf8]
        df = df.with_columns([pl.col(col).str.strip_chars() for col in string_columns])

        # Предобработка сырых изображений
        df = await self.process_excel_images(df, byte_stream)

        logger.info(f"Данные успешно загружены и предобработаны: {df.shape}")

        df = await process_table(
            df,
            brand_column=self.TRADEMARK_COLUMN_NAME,
            description_column=self.DESCRIPTION_COLUMN_NAME,
            correction=correction,
        )

        logger.info(f"Формирование таблицы завершено: {df.shape}")
        return df
