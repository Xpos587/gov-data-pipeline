import re
import logging
import unicodedata
from typing import Optional, Match, Dict, List, Tuple

from openpyxl import load_workbook
from io import BytesIO

import aiohttp
import polars as pl
from polars import DataFrame

from .base import BaseHandler
from utils.gpt import image_to_base64, process_table

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

        try:
            # ВАЖНО: используется skip_rows=4, а потом [1:] => примерно 5 строк пропущено
            df: DataFrame = pl.read_excel(
                byte_stream,
                engine="calamine",
                read_options={"skip_rows": 3},
            )
            # Берем первую строку как заголовки и приводим все значения к строкам
            new_columns = [
                str(val) if val is not None else "UNKNOWN" for val in df.row(0)
            ]

            # Применяем новые заголовки и убираем первую строку
            df = df.slice(2).rename(
                {old: new for old, new in zip(df.columns, new_columns)}
            )

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
            string_columns = [
                col for col, dtype in df.schema.items() if dtype == pl.Utf8
            ]
            df = df.with_columns(
                [
                    pl.col(col).map_elements(
                        clean_text, return_dtype=pl.Utf8, skip_nulls=False
                    )
                    for col in string_columns
                ]
            )

            logger.info(
                f"Данные успешно загружены и обработаны с помощью polars: {df.shape}"
            )

            # Предобработка сырых изображений
            df = await self.preprocess_images(df, byte_stream)

            # Обработка изображений при помощи gpt
            df = await process_table(
                df,
                brand_column=self.TRADEMARK_COLUMN_NAME,
                description_column=self.DESCRIPTION_COLUMN_NAME,
                correction=correction,
            )

            return df

        except Exception as e:
            logger.error(f"Ошибка при обработке данных с помощью polars: {e}")
            return None

    async def preprocess_images(self, df: DataFrame, byte_stream: BytesIO) -> DataFrame:
        """
        Предобрабатывает сырые изображения, учитывая смещение строк (ROW_OFFSET),
        а также возможное частичное смещение в пределах ячейки.
        """
        logger.info("Начинаем предобработку сырых изображений.")

        if self.IMAGE_COLUMN_NAME not in df.columns:
            logger.warning(f"Столбец '{self.IMAGE_COLUMN_NAME}' не найден в DataFrame.")
            logger.warning(df.columns)
            return df

        logger.info(f"Обработка изображений в столбце: {self.IMAGE_COLUMN_NAME}")

        # Загружаем книгу и лист
        try:
            workbook = load_workbook(byte_stream, data_only=True)
            sheet = workbook.active
        except Exception as e:
            logger.error(f"Ошибка загрузки книги Excel: {e}")
            return df

        images_info: Dict[Tuple[int, int], List[str]] = {}

        for image in getattr(sheet, "_images", []):
            try:
                # Основные координаты (целое число)
                base_row = image.anchor._from.row  # 0-based
                base_col = image.anchor._from.col  # 0-based

                # Частичное смещение внутри ячейки (кол-во пикселей)
                rowOff = image.anchor._from.rowOff  # int (пиксели)
                _colOff = image.anchor._from.colOff  # int (пиксели)

                # Ниже — очень упрощённая логика,
                # можно подбирать пороги и корректнее учитывать высоту строки
                # или брать anchor._to, но для примера:
                # Если rowOff > 10000, допустим, сдвигаемся на 1 строку вниз
                additional_row = 1 if rowOff > 10000 else 0

                # Преобразуем openpyxl индексы к "человеческим"
                # +1 потому что row,col 0-based в anchor, а Excel обычно 1-based
                excel_row = base_row + 1 + additional_row
                excel_col = base_col + 1

                # Конвертируем в base64
                img_data = BytesIO(image._data())
                base64_image = await image_to_base64(img_data.getvalue())

                # Сохраняем
                if (excel_row, excel_col) not in images_info:
                    images_info[(excel_row, excel_col)] = []
                images_info[(excel_row, excel_col)].append(base64_image)

            except Exception as e:
                logger.warning(f"Ошибка обработки изображения: {e}")
                continue

        if not images_info:
            logger.warning("Изображения не найдены в файле.")
            return df

        logger.info(f"Найдено {len(images_info)} изображений для обработки.")
        updated_column: List[str] = df[self.IMAGE_COLUMN_NAME].to_list()

        for (excel_row, excel_col), base64_list in images_info.items():
            # Учитываем смещение строк
            df_row = excel_row - self.ROW_OFFSET
            df_row -= 1  # если хотим, чтобы строка 6 Excel -> индекс 0 DataFrame

            if df_row < 0 or df_row >= df.height:
                logger.warning(
                    f"Картинка (Excel row={excel_row}, col={excel_col}) -> df_row={df_row}, вне диапазона DataFrame."
                )
                continue

            # Получаем текущий текст из столбца
            current_text = updated_column[df_row]

            # Дополняем Base64-строками
            new_text = (current_text + " " + " ".join(base64_list)).strip()
            updated_column[df_row] = new_text

            logger.debug(
                f"Обновлена ячейка DataFrame (row={df_row}) для Excel-строки={excel_row} col={excel_col}."
            )

        df = df.with_columns(pl.Series(self.IMAGE_COLUMN_NAME, updated_column))
        logger.info("Предобработка изображений завершена (учтено смещение строк).")

        return df
