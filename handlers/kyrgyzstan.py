import logging
import re
from typing import Optional, Match, Any, Dict
from io import BytesIO

import polars as pl
from polars import DataFrame

from .base import BaseHandler

logger = logging.getLogger(__name__)


class KyrgyzstanHandler(BaseHandler):
    """
    Асинхронный хендлер для получения данных с сайта Киргизии.
    """

    async def retrieve(self, options: Dict[str, Any]) -> Optional[bytes]:
        """
        Загружает страницу, извлекает ссылку на файл с помощью регулярного выражения и возвращает содержимое файла.

        Args:
            options (Dict[str, Any]): Параметры, переданные из конфигурации.

        Returns:
            Optional[bytes]: Содержимое файла в байтах, если файл найден и доступен, иначе None.
        """
        page_url: str = "https://www.customs.gov.kg/article/get?id=46&lang=ru"

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

        # Регулярное выражение для поиска ссылки на файл .pdf
        pattern: re.Pattern[str] = re.compile(
            r"href=&quot;([^&]+\.pdf)&quot;.*?Таможенный реестр объектов интеллектуальной собственности ГТС",
            re.IGNORECASE,
        )
        match: Optional[Match[str]] = re.search(pattern, page_text)

        if not match:
            logger.warning("Ссылка на файл не найдена на странице.")
            return None

        file_url: str = match.group(1)  # Извлекаем первую группу (ссылка на файл)
        logger.info(f"Найдена ссылка на файл: {file_url}")

        # Загружаем файл по найденной ссылке
        file_content: Optional[bytes] = await self.fetch(
            url=file_url,
            headers=headers,
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
            df: DataFrame = pl.read_excel(byte_stream)
            logger.info(
                f"Данные успешно загружены и обработаны с помощью polars: {df.shape}"
            )
            return df
        except Exception as e:
            logger.error(f"Ошибка при обработке данных с помощью polars: {e}")
            return None
