import re
import logging
from typing import Optional, Match, Any, Dict, List, Tuple

from openpyxl import load_workbook
from io import BytesIO

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
        "Наименование товаров, класс товаров по МКТУ или  код товаров по ТН ВЭД"
    )
    # Предположим, что реально пропущено 5 строк (4 через skip_rows + первая строка с заголовками)
    ROW_OFFSET = 5

    async def retrieve(self, options: Dict[str, Any]) -> Optional[bytes]:
        """
        Загружает страницу, извлекает ссылку на файл с помощью регулярного выражения и возвращает содержимое файла.

        Args:
            options (Dict[str, Any]): Параметры, переданные из конфигурации.

        Returns:
            Optional[bytes]: Содержимое файла в байтах, если файл найден и доступен, иначе None.
        """
        page_url: str = "https://kgd.gov.kz/ru/content/tamozhennyy-reestr-obektov-intellektualnoy-sobstvennosti-1"

        proxy = options.get("proxy")
        proxy_auth = options.get("proxy_auth")
        user_agent = options.get("user_agent")

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
        pattern: str = r"https://kgd\.gov\.kz/sites/default/files/pages.*?\.xlsx"
        match: Optional[Match[str]] = re.search(pattern, page_text)

        if not match:
            logger.warning("Ссылка на файл не найдена на странице.")
            return None

        file_url: str = match.group(0)
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
            # ВАЖНО: используется skip_rows=4, а потом [1:] => примерно 5 строк пропущено
            df: DataFrame = pl.read_excel(
                byte_stream,
                engine="xlsx2csv",
                read_options={"skip_rows": 4},
            )[1:]

            # Функция для рефакторинга названий столбцов
            def clean_column_name(name: str) -> str:
                return (
                    name.strip()
                    .replace("\n", " ")
                    .replace("  ", " ")
                    .replace("/", " или ")
                    .replace("Наименова ние", "Наименование")
                )

            # Применяем функцию ко всем названиям столбцов
            df = df.rename({col: clean_column_name(col) for col in df.columns})

            # Убираем пробелы по краям во всех строковых значениях таблицы
            string_columns = [
                col for col, dtype in df.schema.items() if dtype == pl.Utf8
            ]
            df = df.with_columns(
                [pl.col(col).str.strip_chars() for col in string_columns]
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
            df_row = excel_row - self.ROW_OFFSET  # Попробуйте менять ROW_OFFSET
            df_row -= 1  # если хотим, чтобы строка 6 Excel -> индекс 0 DataFrame

            # Проверяем, не вышли ли мы за пределы DataFrame
            if df_row < 0 or df_row >= df.height:
                logger.warning(
                    f"Картинка (Excel row={excel_row}, col={excel_col}) -> df_row={df_row}, вне диапазона DataFrame."
                )
                continue

            current_text = str(updated_column[df_row])

            # Добавляем base64 строки
            new_text = (current_text + " " + " ".join(base64_list)).strip()
            updated_column[df_row] = new_text

            logger.debug(
                f"Обновлена ячейка DataFrame (row={df_row}) для Excel-строки={excel_row} col={excel_col}."
            )

        df = df.with_columns(pl.Series(self.IMAGE_COLUMN_NAME, updated_column))
        logger.info("Предобработка изображений завершена (учтено смещение строк).")

        return df
