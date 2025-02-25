import re
import logging
import unicodedata
from typing import Optional, Match

from io import BytesIO

import aiohttp
import polars as pl
from polars import DataFrame

from .base import BaseHandler
from utils.gpt import process_table

logger = logging.getLogger(__name__)


class KazakhstanHandler(BaseHandler):
    """
    Асинхронный хендлер для получения данных с сайта Казахстана.
    """

    IMAGE_COLUMN_NAME: str = "Наименование (вид, описание, изображение) объекта интеллектуальной собственности"
    TRADEMARK_COLUMN_NAME: str = "Наименование (вид, описание, изображение) объекта интеллектуальной собственности"
    DESCRIPTION_COLUMN_NAME: str = (
        "Наименование товаров, класс товаров по МКТУ или код товаров по ТН ВЭД"
    )
    # Предположим, что реально пропущено 5 строк (4 через skip_rows + первая строка с заголовками)
    ROW_OFFSET = 5

    async def retrieve(
        self,
        user_agent: Optional[str],
        proxy: Optional[str] = None,
        proxy_auth: Optional[aiohttp.BasicAuth] = None,
    ) -> Optional[bytes]:
        """
        Загружает страницу, извлекает ссылку на файл с помощью регулярного выражения и возвращает содержимое файла.

        Args:
            user_agent (str): Заголовок User-Agent.
            proxy (Optional[str]): URL прокси-сервера.
            proxy_auth (Optional[aiohttp.BasicAuth]): Учетные данные для прокси.

        Returns:
            Optional[bytes]: Содержимое файла, иначе None.
        """
        page_url: str = "https://kgd.gov.kz/ru/content/tamozhennyy-reestr-obektov-intellektualnoy-sobstvennosti-1"

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

        # Регулярное выражение для поиска ссылки на файл .xlsx
        pattern: str = r"https://kgd\.gov\.kz/sites/default/files/pages.*?\.(xlsx|xls)"
        match: Optional[Match[str]] = re.search(pattern, page_text)

        if not match:
            logger.warning("Ссылка на файл не найдена на странице.")
            return None

        file_url: str = match.group(0)
        file_format: str = match.group(1)

        logger.info(f"Найдена ссылка на файл: {file_url} (расширение: {file_format})")

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

        if file_content is None:
            logger.error(f"Не удалось загрузить файл по ссылке: {file_url}")
            return None

        logger.info(f"Файл успешно загружен: {file_url}")
        return file_content

    async def process(
        self,
        user_agent: str,
        proxy: Optional[str] = None,
        proxy_auth: Optional[aiohttp.BasicAuth] = None,
        correction: bool = False,
    ) -> Optional[DataFrame]:
        """
        Получает последний доступный файл и обрабатывает его с помощью polars.

        Args:
            user_agent (str): Заголовок User-Agent.
            proxy (Optional[str]): URL прокси-сервера.
            proxy_auth (Optional[aiohttp.BasicAuth]): Учетные данные для прокси.
            correction (bool): Флаг включения/выключения коррекции.

        Returns:
            Optional[DataFrame]: DataFrame с данными из файла, если файл найден и обработан успешно, иначе None.
        """
        data_bytes = await self.retrieve(user_agent, proxy, proxy_auth)
        if data_bytes is None:
            logger.error("Не удалось получить данные последнего файла.")
            return None

        byte_stream = BytesIO(data_bytes)

        # ВАЖНО: используется skip_rows=4, а потом [1:] => примерно 5 строк пропущено
        df: DataFrame = pl.read_excel(
            byte_stream,
            engine="calamine",
            read_options={"skip_rows": 3},
        )
        # Берем первую строку как заголовки и приводим все значения к строкам
        new_columns = [str(val) if val is not None else "UNKNOWN" for val in df.row(0)]

        # Применяем новые заголовки и убираем первую строку
        df = df.slice(2).rename({old: new for old, new in zip(df.columns, new_columns)})

        # Функция для рефакторинга названий столбцов
        def clean_column_name(name: str) -> str:
            name = name.strip()
            name = re.sub(r"Наименова\s*ние", "Наименование", name)
            name = name.replace("/", " или ")
            name = name.replace("\n", " ")
            name = re.sub(r"\s{2,}", " ", name)
            name = "".join(
                char for char in name if char.isprintable()
            )  # удаляет невидимые символы
            return name

        # Приводим названия столбцов в порядок
        df = df.rename({col: clean_column_name(col) for col in df.columns})

        # Функция очистки текста от не-ASCII символов и приведения к нормальной форме
        def clean_text(text: Optional[str]) -> str:
            if text is None:
                return ""

            # Убираем пробелы, переносы строк и нормализуем Unicode
            text = text.strip().replace("\n", " ").replace("\r", "")
            text = re.sub(r"\s{2,}", " ", text)  # удаляем двойные пробелы

            # Нормализуем текст (NFKC: убирает странные комбинации символов)
            text = unicodedata.normalize("NFKC", text)

            # Удаляем все не-ASCII символы, оставляя кириллицу, латиницу и цифры
            text = re.sub(r"[^\w\s\.,;:№\-]", "", text)

            return text

        # Очищаем все текстовые столбцы
        string_columns = [col for col, dtype in df.schema.items() if dtype == pl.Utf8]
        df = df.with_columns(
            [
                pl.col(col).map_elements(
                    clean_text, return_dtype=pl.Utf8, skip_nulls=False
                )
                for col in string_columns
            ]
        )

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
