from __future__ import annotations

import re
import base64
import json
import logging
import asyncio
from io import BytesIO
from typing import Dict, List, Optional, Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionMessageParam
from openai import RateLimitError, AuthenticationError, APIError
from pydantic import BaseModel, Field
from utils.settings import settings
import polars as pl
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_MODEL = "llama-3.2-11b-vision-preview"
BRAND_MODEL = "llama-3.1-8b-instant"
CORRECT_MODEL = "llama-3.1-8b-instant"


class BrandRecognitionResponse(BaseModel):
    english_samples: List[str] = Field(
        ..., description="Samples of brand names in English (max 6)."
    )
    russian_samples: List[str] = Field(
        ..., description="Samples of brand names in Russian (max 6)."
    )
    original_text: str = Field(
        ..., description="Original text where the brand is searched for."
    )


class RowCorrectionResponse(BaseModel):
    corrected_row: Dict[str, str] = Field(
        ..., description="Corrected row (column -> value dictionary)."
    )


def is_excluded(row_text: str) -> bool:
    """
    Проверяет, содержит ли вся строка ключевые слова "исключён" или "исключен".
    Учитывает вариации букв "е" и "ё".
    """
    # Нормализуем текст: заменяем "ё" на "е", приводим к нижнему регистру
    normalized_text = row_text.lower().replace("ё", "е")
    # Ищем слово "исключен" как отдельное слово
    return bool(re.search(r"\bисключен\b", normalized_text))


async def image_to_base64(image_data: bytes) -> str:
    """
    Конвертирует байты изображения в base64-строку формата data:image/png;base64,....
    """
    buf = BytesIO(image_data)
    try:
        img = Image.open(buf)
        if img.format != "PNG":
            out_buf = BytesIO()
            img.save(out_buf, format="PNG")
            image_data = out_buf.getvalue()
    except Exception as exc:
        logger.error(f"Ошибка при обработке изображения: {exc}")
        return ""

    encoded = base64.b64encode(image_data).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


async def call_openai(
    client: AsyncOpenAI,
    model: str,
    messages: List[ChatCompletionMessageParam],
    max_retries: int = 5,
    initial_delay: float = 2,
    temperature: float = 0.1,
    max_tokens: int = 64,
    **kwargs: Any,
) -> ChatCompletion:
    """
    Универсальная функция для вызова OpenAI ChatCompletion с повторными попытками
    при возникновении RateLimitError (429), AuthenticationError (401) и других API-ошибках.

    Аргументы:
      - client: Экземпляр AsyncOpenAI.
      - model: Название модели.
      - messages: Список сообщений (role, content).
      - max_retries: Максимальное число повторов при ошибках.
      - initial_delay: Начальная задержка перед повтором (сек).
      - temperature, max_tokens: Параметры генерации ответа.
      - kwargs: Прочие параметры, передаваемые в create().

    Возвращает ответ (response), либо выбрасывает исключение при неудаче.
    """
    delay: float = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            response: ChatCompletion = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response
        except RateLimitError:
            if attempt < max_retries:
                logger.warning(
                    f"Превышен лимит (429). Попытка {attempt}/{max_retries}. "
                    f"Повтор через {delay} сек..."
                )
                await asyncio.sleep(delay)
                delay *= 2
            else:
                logger.error("Превышен лимит. Достигнуто макс. попыток!")
                raise
        except AuthenticationError:
            logger.error("Ошибка аутентификации (401). Проверьте API ключ!")
            raise
        except APIError as err:
            logger.error(f"API Ошибка: {err}. Попытка {attempt}/{max_retries}.")
            if attempt < max_retries:
                await asyncio.sleep(delay)
                delay *= 2
            else:
                logger.error("Достигнут максимум попыток API.")
                raise

    # Если цикл завершился без успешного возврата, выбрасываем исключение
    raise RuntimeError("Не удалось получить ответ от OpenAI после всех попыток.")


async def process_table(
    df: pl.DataFrame, brand_column: str, description_column: Optional[str] = None
) -> pl.DataFrame:
    """
    Обрабатывает DataFrame, распознаёт бренды и корректирует каждую строку.

    1. Проверяет, не содержит ли строка ключевые слова "исключён"/"исключен".
       Если содержит — помечает строку, добавляя поле "Исключено = 'Да'" и пропускает дальнейшую обработку.

    2. Если строка не исключена, вызывает `recognize_brand` для указанного столбца с брендом и, при наличии, столбца с описанием.
       Заменяет значение бренда на (original_text + english_samples + russian_samples).

    3. Вызывает `correct_row` для итоговых данных и сохраняет результат.

    Args:
        df (pl.DataFrame): Исходный DataFrame с данными.
        brand_column (str): Название столбца, содержащего торговую марку.
        description_column (Optional[str]): Название столбца с описанием фирмы для улучшения распознавания.

    Returns:
        pl.DataFrame: Обработанный DataFrame с полем 'Исключено' и/или обновлёнными данными.
    """
    client = AsyncOpenAI(
        api_key=settings.openai.api_key.get_secret_value(),
        base_url=settings.openai.base_url,
    )

    brand_system_prompt = (
        "Identify the brand name in the input data. "
        "Respond strictly in JSON format following the provided schema. "
        "For each brand, provide no more than 6 items in the following categories:\n"
        "- Brand name samples in English.\n"
        "- Brand name samples in Russian.\n"
        "Ensure that each list contains no more than 6 items.\n"
        "JSON schema: "
        f"{json.dumps(BrandRecognitionResponse.model_json_schema(), indent=2)}"
    )

    row_system_prompt: str = (
        "Correct the table row. Respond strictly in JSON format with the key 'corrected_row' following the provided schema: "
        f"{json.dumps(RowCorrectionResponse.model_json_schema(), indent=2)}"
    )

    async def recognize_image(base64_image: str) -> str:
        """
        Использует GPT Vision для распознавания текста из изображения.
        """
        response = await call_openai(
            client,
            model=IMAGE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You must return ONLY the text found in the image."
                                "No descriptions, no explanations, no formatting."
                                "Just the raw text."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_image},
                        },
                    ],
                },
            ],
            temperature=0.1,
            max_tokens=64,
        )
        return response.choices[0].message.content or ""

    async def recognize_brand(
        text: str, description: Optional[str] = None
    ) -> BrandRecognitionResponse:
        """
        Распознаёт бренд, возвращая структуру BrandRecognitionResponse.
        Если найдена base64-строка изображения, распознаёт и заменяет её на извлечённый текст.
        В случае, если description_column указан, добавляет его текст к тексту для лучшего распознавания.
        """
        # Ищем в тексте возможную строку data:image/png;base64,....
        pattern = r"data:image/png;base64,[A-Za-z0-9+/=]+"
        match = re.search(pattern, text)
        if match:
            base64_str = match.group(0)
            recognized = await recognize_image(base64_str)
            # Заменяем base64 в тексте на извлечённый
            text = text.replace(base64_str, recognized)

        # Если описание присутствует, добавляем его
        if description:
            text = f"{description} {text}"

        logger.debug(f"Запрос: {text}")

        response = await call_openai(
            client,
            model=BRAND_MODEL,
            messages=[
                {"role": "system", "content": brand_system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
            max_tokens=256,
            response_format={"type": "json_object"},
        )

        raw_text: str = response.choices[0].message.content or ""
        logger.debug(f"Ответ: {raw_text}")

        # Парсим ответ в модель BrandRecognitionResponse
        return BrandRecognitionResponse.model_validate_json(raw_text)

    async def correct_row(row_data: Dict[str, Optional[str]]) -> RowCorrectionResponse:
        """
        Корректирует строку, возвращая структуру RowCorrectionResponse.
        """
        logger.debug(f"Запрос: {json.dumps(row_data, ensure_ascii=False)}")

        response = await call_openai(
            client,
            model=CORRECT_MODEL,
            messages=[
                {"role": "system", "content": row_system_prompt},
                {
                    "role": "user",
                    "content": json.dumps({"row": row_data}, ensure_ascii=False),
                },
            ],
            temperature=0.2,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )

        raw_text: str = response.choices[0].message.content or ""
        logger.debug(f"Ответ: {raw_text}")

        # Парсим ответ в модель RowCorrectionResponse
        corrected_resp = RowCorrectionResponse.model_validate_json(raw_text)

        # Убедимся, что все значения являются строками
        for k, v in corrected_resp.corrected_row.items():
            corrected_resp.corrected_row[k] = v

        return corrected_resp

    processed_rows: List[Dict[str, str]] = []

    # Перебираем строки DataFrame
    for idx, row in enumerate(df.iter_rows(named=True), start=1):
        row_dict: Dict[str, Optional[str]] = dict(row)
        logger.debug(f"=== Обрабатываем строку {idx}: {row_dict}")

        # 1) Проверяем, исключена ли ТМ, анализируя ВСЮ строку
        combined_text = " ".join(
            str(val) for val in row_dict.values() if val is not None
        )
        if is_excluded(combined_text):
            logger.info(
                f"Строка {idx}: обнаружено ключевое слово 'исключён'. Пропускаем распознавание бренда."
            )
            # Добавляем 'Исключено' со значением 'Да'
            row_dict["Исключено"] = "Да"
            # Удаляем другие возможные лишние поля, если необходимо
            # Преобразуем row_dict в плоский словарь
            flat_row = {k: (v or "") for k, v in row_dict.items()}
            processed_rows.append(flat_row)
            continue

        # 2) Если не исключена, обрабатываем только столбец brand_column и описание (если есть)
        brand_val = row_dict.get(brand_column)
        description_val = (
            row_dict.get(description_column) if description_column else None
        )

        if isinstance(brand_val, str) and brand_val.strip():
            brand_resp = await recognize_brand(brand_val, description_val)
            # Сформируем итоговую строку (original + english + russian)
            combined_brand_info = " ".join(
                [
                    brand_resp.original_text,
                    ", ".join(brand_resp.english_samples),
                    ", ".join(brand_resp.russian_samples),
                ]
            ).strip()
            row_dict[brand_column] = combined_brand_info

        # 3) Добавляем 'Исключено' со значением 'Нет'
        row_dict["Исключено"] = "Нет"

        # 4) Корректируем строку
        corrected_resp = await correct_row(row_dict)
        # Добавляем исправленную строку
        processed_rows.append(corrected_resp.corrected_row)

    return pl.DataFrame(processed_rows)
