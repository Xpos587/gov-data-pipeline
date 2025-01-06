import logging
import re
from typing import Optional, Match, Any, Dict
from io import BytesIO
from urllib.parse import urlparse

import polars as pl
from polars import DataFrame

from .base import BaseHandler

logger = logging.getLogger(__name__)


class BelarusHandler(BaseHandler):
    """
    Асинхронный хендлер для получения данных с сайта Беларуси.
    """

    async def retrieve(self, options: Dict[str, Any]) -> Optional[bytes]:
        """
        Загружает страницу, извлекает ссылку на файл с помощью регулярного выражения и возвращает содержимое файла.

        Args:
            options (Dict[str, Any]): Параметры, переданные из конфигурации.

        Returns:
            Optional[bytes]: Содержимое файла в байтах, если файл найден и доступен, иначе None.
        """
        page_url: str = "https://www.customs.gov.by/zashchita-prav-na-obekty-intellektualnoy-sobstvennosti"

        headers = options.get("headers", {})
        proxy = options.get("proxy")
        proxy_auth = options.get("proxy_auth")
        user_agent = options.get("user_agent")

        logger.info(f"Запрашиваем страницу: {page_url}")

        page_content: Optional[bytes] = await self.fetch(
            url=page_url,
            headers=headers,
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

        # Регулярное выражение для поиска ссылки на файл .xlsx
        pattern: str = r"/uploads/reestr-is/Reestr\d{6}\.xlsx"
        match: Optional[Match[str]] = re.search(pattern, page_text)

        if not match:
            logger.warning("Ссылка на файл не найдена на странице.")
            return None

        # Извлечение поддомена из page_url
        parsed_url = urlparse(page_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # Формирование полного URL файла
        file_url: str = base_url + match.group(0)
        logger.info(f"Найдена ссылка на файл: {file_url}")

        # Загружаем файл по найденной ссылке
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

    async def process(self, options: Dict[str, Any]) -> Optional[DataFrame]:
        """
        Получает последний доступный файл и обрабатывает его с помощью polars.

        Args:
            options (Dict[str, Any]): Параметры, переданные из конфигурации.

        Returns:
            Optional[DataFrame]: DataFrame с данными из файла, если файл найден и обработан успешно, иначе None.
        """
        data_bytes: Optional[bytes] = await self.retrieve(options)
        if data_bytes is None:
            logger.error("Не удалось получить данные последнего файла.")
            return None

        try:
            byte_stream: BytesIO = BytesIO(data_bytes)
            # Чтение данных с использованием Polars
            df: DataFrame = pl.read_excel(
                byte_stream,
                engine="xlsx2csv",
                read_options={
                    "skip_rows": 3,
                    "schema_overrides": {"№ п/п": pl.Utf8},
                },
            )[4:]

            # Убираем пробелы из строковых колонок
            string_columns = [
                col for col, dtype in df.schema.items() if dtype == pl.Utf8
            ]
            df = df.with_columns(
                [pl.col(col).str.strip_chars() for col in string_columns]
            )

            logger.info(
                f"Данные успешно загружены и обработаны с помощью polars: {df.shape}"
            )
            return df
        except Exception as e:
            logger.error(f"Ошибка при обработке данных с помощью polars: {e}")
            return None
