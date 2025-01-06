import logging
from typing import List, Dict, Union
import base64
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import pytesseract
import polars as pl
import openai
from pydantic import BaseModel
from utils.settings import settings

logger = logging.getLogger(__name__)


# Функция для преобразования изображения в Base64 PNG
def image_to_base64(image_data: bytes) -> str:
    """
    Конвертирует изображение в формат Base64 PNG.

    Args:
        image_data (bytes): Байтовые данные изображения.

    Returns:
        str: Base64-строка изображения.
    """
    try:
        with BytesIO(image_data) as buffer:
            image = Image.open(buffer)
            if image.format != "PNG":
                output_buffer = BytesIO()
                image.save(output_buffer, format="PNG")
                image_data = output_buffer.getvalue()

            base64_image = base64.b64encode(image_data).decode("utf-8")
            return f"data:image/png;base64,{base64_image}"
    except UnidentifiedImageError as e:
        logger.error(f"Формат изображения не поддерживается: {e}")
        raise ValueError("Формат изображения не поддерживается.") from e
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}")
        raise RuntimeError("Ошибка при обработке изображения.") from e


# Функция OCR для Base64 изображений
def ocr_base64_image(base64_image: str, langs: str = "eng+rus") -> Union[str, None]:
    """
    Извлекает текст из Base64 PNG изображения с помощью Tesseract OCR.

    Args:
        base64_image (str): Base64 PNG изображение.
        langs (str): Языки для OCR (например, "eng+rus").

    Returns:
        Union[str, None]: Извлечённый текст или None, если текст не найден.
    """
    try:
        image_data: bytes = base64.b64decode(base64_image.split(",")[1])
        image: Image.Image = Image.open(BytesIO(image_data))
        extracted_text: str = pytesseract.image_to_string(image, lang=langs).strip()  # type: ignore
        if not extracted_text:
            logger.debug("Текст не обнаружен на изображении.")
        return extracted_text if extracted_text else None  # type: ignore
    except Exception as e:
        logger.error(f"Ошибка OCR: {e}")
        return None


# Pydantic модель для структурирования строки таблицы
class TableRow(BaseModel):
    """
    Модель строки таблицы для LLM.
    """

    columns: List[str]
    row: Dict[str, Union[str, None]]


# Pydantic модель для ответа от GPT
class GPTResponse(BaseModel):
    """
    Модель ответа от GPT для корректировки строки.
    """

    corrected_row: Dict[str, Union[str, None]]


# Асинхронная функция для корректировки строки таблицы
async def correct_row(row: TableRow) -> GPTResponse:
    """
    Корректирует строку с помощью OpenAI GPT.

    Args:
        row (TableRow): Строка для корректировки.

    Returns:
        GPTResponse: Корректированная строка.
    """
    if not settings.openai:
        raise ValueError("Настройки OpenAI не найдены.")

    openai.api_key = settings.openai.api_key.get_secret_value()
    system_prompt = (
        "You are a data processor for tabular data. "
        "Your task is to ensure that all data in the row is aligned with the columns, "
        "remove irrelevant content, and fix any formatting or alignment issues. "
        "Provide the corrected row as JSON."
    )

    try:
        async with openai.AsyncOpenAI() as client:
            response = await client.chat.completions.create(
                model=settings.openai.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row.model_dump_json()},
                ],
                temperature=0.2,
                max_tokens=settings.openai.max_tokens,
            )

        parsed_content = response.choices[0].message.content
        if parsed_content is None:
            logger.error("Получен пустой ответ от GPT.")
            return GPTResponse(corrected_row=row.row)

        return GPTResponse.model_validate_json(parsed_content)

    except Exception as e:
        logger.error(f"Ошибка GPT: {e}")
        return GPTResponse(corrected_row=row.row)


# Асинхронная функция обработки таблицы
async def process_table(df: pl.DataFrame) -> pl.DataFrame:
    """
    Обрабатывает таблицу, заменяя изображения на текст и корректируя строки с помощью LLM.

    Args:
        df (pl.DataFrame): Исходная таблица.

    Returns:
        pl.DataFrame: Обработанная таблица.
    """
    if not settings.openai:
        raise ValueError("Настройки OpenAI не найдены.")

    processed_rows: List[Dict[str, Union[str, None]]] = []

    for row in df.iter_rows(named=True):
        row_dict = row.copy()

        # Обработка изображений в ячейках
        for col, value in row_dict.items():
            if isinstance(value, str) and value.startswith("data:image/png;base64,"):
                ocr_text = ocr_base64_image(value)
                if ocr_text:
                    logger.info(f"OCR успешно извлёк текст: {ocr_text}")
                row_dict[col] = f"{value}\n{ocr_text}" if ocr_text else value

        # Коррекция строки через LLM
        structured_row = TableRow(columns=list(df.columns), row=row_dict)
        corrected_row = await correct_row(structured_row)
        processed_rows.append(corrected_row.corrected_row)

    logger.info("Таблица успешно обработана.")
    return pl.DataFrame(processed_rows)


__all__ = ["image_to_base64", "ocr_base64_image", "correct_row", "process_table"]
