import logging
import re
from typing import Optional, Match, Any, Dict, List, Tuple
from io import BytesIO

import polars as pl
from polars import DataFrame

from .base import BaseHandler
from utils.ocr import image_to_base64, ocr_base64_image

logger = logging.getLogger(__name__)


class KazakhstanHandler(BaseHandler):
    """
    Асинхронный хендлер для получения данных с сайта Казахстана.
    """

    async def retrieve(self, options: Dict[str, Any]) -> Optional[bytes]:
        """
        Загружает страницу, извлекает ссылку на файл с помощью регулярного выражения и возвращает содержимое файла.

        Args:
            options (Dict[str, Any]): Параметры, переданные из конфигурации.

        Returns:
            Optional[bytes]: Содержимое файла в байтах, если файл найден и доступен, иначе None.
        """
        page_url: str = "https://kgd.gov.kz/ru/content/tamozhennyy-reestr-obektov-intellektualnoy-sobstvennosti-1"

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

            # Извлекаем изображения
            images_info = self.extract_images(byte_stream)
            if not images_info:
                logger.warning("Изображения не найдены или не обработаны.")

            # Обновляем DataFrame с учетом изображений
            target_column = "Наименование (вид, описание, изображение) объекта интеллектуальной собственности"
            updated_column: List[Optional[str]] = [
                self.update_column_with_images(
                    row_index=i,
                    text_value=df[target_column][i],
                    images_dict=images_info,
                )
                for i in range(len(df))
            ]
            df = df.with_columns([pl.Series(target_column, updated_column)])

            return df

        except Exception as e:
            logger.error(f"Ошибка при обработке данных с помощью polars: {e}")
            return None

    @staticmethod
    def extract_images(byte_stream: BytesIO) -> Dict[Tuple[int, int], List[str]]:
        """
        Извлекает изображения из Excel файла, возвращает их в Base64 формате с привязкой к ячейкам.

        Args:
            byte_stream (BytesIO): Поток байтов с данными Excel файла.

        Returns:
            Dict[Tuple[int, int], List[str]]: Словарь с координатами ячеек и списком изображений в Base64 формате.
        """
        from openpyxl import load_workbook

        workbook = load_workbook(byte_stream, data_only=True)
        sheet = workbook.active
        images_info: Dict[Tuple[int, int], List[str]] = {}

        for image in getattr(sheet, "_images", []):
            try:
                # Определяем координаты изображения в Excel
                row = image.anchor._from.row + 1
                col = image.anchor._from.col + 1

                # Получаем данные изображения
                img_data = BytesIO(image._data())
                base64_image = image_to_base64(img_data.getvalue())

                # Сохраняем изображение в словарь
                if (row, col) not in images_info:
                    images_info[(row, col)] = []
                images_info[(row, col)].append(base64_image)

            except Exception as e:
                logger.warning(f"Ошибка обработки изображения: {e}")
                continue

        return images_info

    def update_column_with_images(
        self,
        row_index: int,
        text_value: Optional[str],
        images_dict: Dict[Tuple[int, int], List[str]],
    ) -> Optional[str]:
        """
        Заменяет в тексте base64-изображения на распознанный OCR-текст
        (если ничего не распознано – удаляет изображение).
        """
        base64_images = images_dict.get((row_index + 6, 2), [])
        ocr_texts: List[str] = []
        for base64_image in base64_images:
            recognized = ocr_base64_image(base64_image)  # вызываем OCR
            if recognized:
                ocr_texts.append(recognized)

        # Если OCR что-то распознал, добавляем в исходный текст, иначе пропускаем
        if ocr_texts:
            return f"{text_value or ''} {'; '.join(ocr_texts)}".strip()
        return text_value
