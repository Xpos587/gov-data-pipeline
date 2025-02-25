import re
import json
import logging
import aiohttp
from typing import Optional
from pydantic import BaseModel

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
    output_extensions: list[str]
    server: str
    task: str
    file_number: int
    download_filename: str
    files: list[FileInfo]


async def convert_to_docx(
    pdf_content: bytes,
    user_agent: str,
    session: aiohttp.ClientSession,
    proxy: Optional[str] = None,
    proxy_auth: Optional[aiohttp.BasicAuth] = None,
) -> Optional[bytes]:
    """
    Конвертирует PDF в DOCX с использованием API ilovepdf.

    :param pdf_content: Байты PDF-файла.
    :param user_agent: Заголовок User-Agent.
    :param session: aiohttp.ClientSession для запросов.
    :param proxy: URL прокси (если требуется).
    :param proxy_auth: Аутентификация для прокси (если требуется).
    :return: Байты DOCX или None при ошибке.
    """
    # Получаем token и taskId с ilovepdf
    home_url: str = "https://www.ilovepdf.com/pdf_to_word"
    logger.info(f"Запрашиваем главную страницу ilovepdf: {home_url}")
    try:
        async with session.get(
            home_url,
            proxy=proxy,
            proxy_auth=proxy_auth,
            headers={"User-Agent": user_agent},
        ) as response:
            home_content = await response.read()
    except Exception as e:
        logger.error(f"Ошибка при загрузке страницы ilovepdf: {e}")
        return None

    try:
        home_text: str = home_content.decode("utf-8", "ignore")
    except Exception as e:
        logger.error(f"Ошибка декодирования страницы ilovepdf: {e}")
        return None

    token_match = re.search(r'"token":\s*"([^"]+)"', home_text)
    task_match = re.search(r"ilovepdfConfig\.taskId\s*=\s*'([^']+)'", home_text)
    if not token_match or not task_match:
        logger.error("Не удалось извлечь token или taskId с ilovepdf.")
        return None

    bearer: str = token_match.group(1)
    task_id: str = task_match.group(1)
    logger.info(f"Получен token: {bearer} и taskId: {task_id}")

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {bearer}",
        "User-Agent": user_agent,
    }

    # Загружаем PDF (upload)
    form_data = aiohttp.FormData()
    form_data.add_field("name", "kyrgyzstan.pdf")
    form_data.add_field("chunk", "0")
    form_data.add_field("chunks", "1")
    form_data.add_field("task", task_id)
    form_data.add_field("preview", "1")
    form_data.add_field(
        "file", pdf_content, filename="kyrgyzstan.pdf", content_type="application/pdf"
    )

    try:
        async with session.post(
            "https://api85o.ilovepdf.com/v1/upload",
            headers=headers,
            data=form_data,
            proxy=proxy,
            proxy_auth=proxy_auth,
        ) as response:
            upload_resp = await response.read()
    except Exception as e:
        logger.error(f"Ошибка загрузки PDF (upload): {e}")
        return None

    try:
        upload_json = json.loads(upload_resp.decode("utf-8"))
        upload_data = UploadResponse(**upload_json)
    except Exception as e:
        logger.error(f"Ошибка парсинга UploadResponse: {e}")
        return None

    # Конвертируем PDF -> DOCX
    process_data = {
        "convert_to": "docx",
        "output_filename": "{filename}",
        "packaged_filename": "ilovepdf_converted",
        "ocr": "0",
        "task": task_id,
        "tool": "pdfoffice",
        "files[0][server_filename]": upload_data.server_filename,
        "files[0][filename]": "kyrgyzstan.pdf",
    }

    logger.info("Отправляем запрос на конвертацию PDF в DOCX")
    try:
        async with session.post(
            "https://api85o.ilovepdf.com/v1/process",
            headers=headers,
            data=process_data,
            proxy=proxy,
            proxy_auth=proxy_auth,
        ) as response:
            process_resp = await response.read()
    except Exception as e:
        logger.error(f"Ошибка конвертации PDF в DOCX (process): {e}")
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

    # Проверяем статус задачи
    try:
        async with session.get(
            f"https://api85o.ilovepdf.com/v1/task/{task_id}",
            headers=headers,
            proxy=proxy,
            proxy_auth=proxy_auth,
        ) as response:
            status_content = await response.read()
    except Exception as e:
        logger.error(f"Ошибка получения статуса задачи: {e}")
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

    # Скачиваем DOCX
    try:
        async with session.get(
            f"https://api85o.ilovepdf.com/v1/download/{task_id}",
            headers={"User-Agent": user_agent},
            proxy=proxy,
            proxy_auth=proxy_auth,
        ) as response:
            docx_data = await response.read()
    except Exception as e:
        logger.error(f"Ошибка скачивания DOCX: {e}")
        return None

    logger.info(f"DOCX успешно загружен, размер: {len(docx_data)} байт")
    return docx_data
