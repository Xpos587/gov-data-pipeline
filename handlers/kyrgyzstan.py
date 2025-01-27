import logging
import re
import json
from io import BytesIO
from typing import Optional, Any, Dict, List

import aiohttp
import polars as pl
from polars import DataFrame
from docx import Document
from pydantic import BaseModel

from .base import BaseHandler

logger = logging.getLogger(__name__)


class UploadResponse(BaseModel):
    server_filename: str
    scanned: str


class ProcessResponse(BaseModel):
    download_filename: str
    filesize: int
    output_filesize: int
    output_filenumber: int
    output_extensions: str
    timer: str
    status: str


class FileInfo(BaseModel):
    filename: str
    server_filename: str
    filesize: int
    output_filesize: int
    status: int
    timer: str


class TaskStatusResponse(BaseModel):
    tool: str
    process_start: str
    custom_int: Optional[int]
    custom_string: Optional[str]
    status: str
    status_message: str
    timer: str
    filesize: int
    output_filesize: int
    output_filenumber: int
    output_extensions: List[str]
    server: str
    task: str
    file_number: int
    download_filename: str
    files: List[FileInfo]


def clean_cell(cell: Optional[str]) -> str:
    return re.sub(r"\s+", " ", cell.strip()) if cell else ""


def is_new_record(val: str) -> bool:
    return bool(re.match(r"^(?:№?\d{4,})(/ТЗ.*)?", val.strip()))


def merge_continued_rows(df: pl.DataFrame, key_col: str) -> pl.DataFrame:
    rows: List[Dict[str, Any]] = df.to_dicts()
    merged_rows: List[Dict[str, Any]] = []
    prev: Optional[Dict[str, Any]] = None
    for row in rows:
        current_val = str(row[key_col]).strip()
        if is_new_record(current_val):
            if prev is not None:
                merged_rows.append(prev)
            prev = row
        else:
            if prev:
                for col in df.columns:
                    cur_val = str(row[col]).strip()
                    if cur_val:
                        prev[col] = (
                            (prev[col] + " " + cur_val).strip()
                            if prev[col]
                            else cur_val
                        )
            else:
                prev = row
    if prev is not None:
        merged_rows.append(prev)
    return pl.DataFrame(merged_rows)


def process_tables(all_data: List[List[str]]) -> pl.DataFrame:
    if not all_data:
        raise ValueError("Список таблиц пуст.")

    # Выравниваем длину строк
    max_len = max(len(r) for r in all_data)
    padded = [r + [""] * (max_len - len(r)) for r in all_data]

    # Первая строка — имена столбцов, вторая строка пропускается как служебная
    raw_columns = padded[0]
    # Очистка имён столбцов: убираем переводы строки и лишние пробелы
    columns = [
        re.sub(r"\s+", " ", col).strip() or f"Unnamed_{i}"
        for i, col in enumerate(raw_columns)
    ]

    if len(padded) < 3:
        raise ValueError("Недостаточно строк для формирования данных.")
    data_rows = padded[2:]
    df = pl.DataFrame(data_rows, schema=columns, orient="row").with_columns(
        [pl.col(c).cast(pl.Utf8).map_elements(clean_cell).alias(c) for c in columns]
    )
    # Пример переименования/предобработки (расширьте при необходимости)
    if "Рег. №" in df.columns:
        df = df.with_columns(pl.col("Рег. №").str.replace(r"^№\s*", ""))
        df = merge_continued_rows(df, "Рег. №")
    return df


class KyrgyzstanHandler(BaseHandler):
    """
    Асинхронный хендлер для обработки данных Киргизии.

    Функция retrieve:
      1. Скачивает PDF с сайта Киргизии.
      2. Получает token и taskId с ilovepdf.com.
      3. Загружает PDF на ilovepdf и инициирует конвертацию в DOCX.
      4. Проверяет статус задачи и скачивает DOCX.
      5. Возвращает байты DOCX.
    """

    async def retrieve(self, options: Dict[str, Any]) -> Optional[bytes]:
        # 1. Скачиваем PDF с сайта Киргизии
        page_url: str = "https://www.customs.gov.kg/article/get?id=46&lang=ru"
        logger.info(f"Запрашиваем страницу с PDF: {page_url}")
        pdf_page = await self.fetch(
            url=page_url,
            proxy=options.get("proxy"),
            proxy_auth=options.get("proxy_auth"),
            user_agent=options.get("user_agent"),
        )
        if not pdf_page:
            logger.error("Не удалось загрузить страницу с PDF.")
            return None

        try:
            page_text = pdf_page.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.error(f"Ошибка декодирования страницы: {e}")
            return None

        pdf_match = re.search(
            r"href=&quot;([^&]+\.pdf)&quot;.*?Таможенный реестр.*?интеллектуальной собственности",
            page_text,
            re.IGNORECASE,
        )
        if not pdf_match:
            logger.warning("Ссылка на PDF не найдена на странице.")
            return None

        pdf_url = pdf_match.group(1)
        if not pdf_url.startswith("http"):
            pdf_url = "https://www.customs.gov.kg" + pdf_url
        logger.info(f"Найдена ссылка на PDF: {pdf_url}")

        pdf_content = await self.fetch(
            url=pdf_url,
            proxy=options.get("proxy"),
            proxy_auth=options.get("proxy_auth"),
            user_agent=options.get("user_agent"),
        )

        if not pdf_content:
            logger.error("Не удалось скачать PDF.")
            return None

        # 2. Получаем token и taskId с ilovepdf
        home_url: str = "https://www.ilovepdf.com/pdf_to_word"
        logger.info(f"Запрашиваем главную страницу ilovepdf: {home_url}")
        home_content = await self.fetch(url=home_url)
        if not home_content:
            logger.error("Не удалось загрузить страницу ilovepdf.")
            return None

        try:
            home_text = home_content.decode("utf-8", "ignore")
        except UnicodeDecodeError as e:
            logger.error(f"Ошибка декодирования страницы ilovepdf: {e}")
            return None

        token_match = re.search(r'"token":\s*"([^"]+)"', home_text)
        task_match = re.search(r"ilovepdfConfig\.taskId\s*=\s*'([^']+)'", home_text)
        if not token_match or not task_match:
            logger.error("Не удалось извлечь token или taskId с ilovepdf.")
            return None

        bearer = token_match.group(1)
        task_id = task_match.group(1)

        logger.info(f"Получен token: {bearer} и taskId: {task_id}")

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {bearer}",
        }

        # 3. Загружаем PDF (upload) на ilovepdf
        form_data = aiohttp.FormData()
        form_data.add_field("name", "kyrgyzstan.pdf")
        form_data.add_field("chunk", "0")
        form_data.add_field("chunks", "1")
        form_data.add_field("task", task_id)
        form_data.add_field("preview", "1")
        form_data.add_field(
            "file",
            pdf_content,
            filename="kyrgyzstan.pdf",
            content_type="application/pdf",
        )
        upload_resp = await self.post(
            url="https://api85o.ilovepdf.com/v1/upload",
            headers=headers,
            data=form_data,
        )
        if not upload_resp:
            logger.error("Ошибка загрузки PDF (upload).")
            return None

        try:
            upload_json = json.loads(upload_resp.decode("utf-8"))
            upload_data = UploadResponse(**upload_json)
        except Exception as e:
            logger.error(f"Ошибка парсинга UploadResponse: {e}")
            return None

        # 4. Конвертируем PDF -> DOCX
        process_resp = await self.post(
            url="https://api85o.ilovepdf.com/v1/process",
            headers=headers,
            data={
                "convert_to": "docx",
                "output_filename": "{filename}",
                "packaged_filename": "ilovepdf_converted",
                "ocr": "0",
                "task": task_id,
                "tool": "pdfoffice",
                "files[0][server_filename]": upload_data.server_filename,
                "files[0][filename]": "kyrgyzstan.pdf",
            },
        )
        if not process_resp:
            logger.error("Ошибка конвертации PDF в DOCX (process).")
            return None
        try:
            proc_json = json.loads(process_resp.decode("utf-8"))
            proc_data = ProcessResponse(**proc_json)
            if proc_data.status.lower() != "tasksuccess":
                logger.error(f"Конвертация вернула статус: {proc_data.status}")
                return None
        except Exception as e:
            logger.error(f"Ошибка парсинга ProcessResponse: {e}")
            return None

        # 5. Проверяем статус задачи
        status_content = await self.fetch(
            url=f"https://api85o.ilovepdf.com/v1/task/{task_id}",
            headers=headers,
            user_agent=options.get("user_agent"),
        )
        if not status_content:
            logger.error("Ошибка получения статуса задачи.")
            return None
        try:
            status_json = json.loads(status_content.decode("utf-8"))
            status_data = TaskStatusResponse(**status_json)
            if status_data.status.lower() != "tasksuccess":
                logger.error(f"Статус задачи: {status_data.status}")
                return None
        except Exception as e:
            logger.error(f"Ошибка парсинга TaskStatusResponse: {e}")
            return None

        # 6. Скачиваем DOCX и возвращаем его байты
        docx_data = await self.fetch(
            url=f"https://api85o.ilovepdf.com/v1/download/{task_id}",
            user_agent=options.get("user_agent"),
        )
        if not docx_data:
            logger.error("Не удалось скачать DOCX.")
            return None

        logger.info(f"DOCX успешно загружен, размер: {len(docx_data)} байт")
        return docx_data

    async def process(self, options: Dict[str, Any]) -> Optional[DataFrame]:
        """
        Скачивает PDF, конвертирует его в DOCX, извлекает таблицы и формирует Polars DataFrame.
        """
        docx_bytes = await self.retrieve(options)
        if not docx_bytes:
            logger.error("DOCX не получен, прерываем обработку.")
            return None

        try:
            document = Document(BytesIO(docx_bytes))
            all_data: List[List[str]] = []
            for table in document.tables:
                for row in table.rows:
                    all_data.append([cell.text.strip() for cell in row.cells])
            df = process_tables(all_data)
            logger.info(f"Формирование DataFrame завершено: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Ошибка при чтении DOCX: {e}")
            return None
