from __future__ import annotations

import re
import base64
import json
import logging
import asyncio
import unicodedata
from io import BytesIO
from typing import Dict, List, Optional, Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import completion_create_params
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
    Проверяет, содержит ли строка ключевые слова "исключён" или "исключен".
    Игнорирует регистр и пробелы.
    """
    # Нормализуем текст: удаляем пробелы, заменяем "ё" на "е", приводим к нижнему регистру
    normalized_text = re.sub(r"\s+", "", row_text)  # Удаление всех пробелов
    normalized_text = normalized_text.casefold().replace("ё", "е")
    return "исключен" in normalized_text


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


def clean_messages(
    messages: List[ChatCompletionMessageParam],
) -> List[ChatCompletionMessageParam]:
    """
    Очищает содержимое сообщений от нежелательных Unicode-символов.

    Если значение ключа "content" — строка, удаляются все контрольные символы и нормализуются пробелы.
    Если значение — список (например, для GPT Vision), то для каждого элемента,
    если он представляет собой словарь с ключом "text", также очищается текст.
    """
    cleaned: List[ChatCompletionMessageParam] = []

    for msg in messages:
        # Создаём копию сообщения
        new_msg: Dict[str, Any] = dict(msg)

        content: Any = new_msg.get("content", None)

        if isinstance(content, str):
            # Удаляем контрольные символы (категория "C")
            cleaned_text = "".join(
                ch for ch in content if unicodedata.category(ch)[0] != "C"
            )
            # Нормализуем пробелы
            cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
            new_msg["content"] = cleaned_text

        elif isinstance(content, list):
            # Явно указываем, что content — это список неизвестных объектов
            content_list: List[Any] = cast(List[Any], content)
            new_content: List[Any] = []
            for item in content_list:
                if isinstance(item, dict) and "text" in item:
                    # Приводим item к dict[str, Any]
                    item_dict: Dict[str, Any] = cast(Dict[str, Any], item)
                    text_val: str = str(item_dict.get("text", ""))

                    # Удаляем контрольные символы и нормализуем пробелы
                    text_val = "".join(
                        ch for ch in text_val if unicodedata.category(ch)[0] != "C"
                    )
                    text_val = re.sub(r"\s+", " ", text_val).strip()

                    # Создаём копию, чтобы изменить только "text"
                    new_item: Dict[str, Any] = {}
                    for key_item, val_item in item_dict.items():
                        new_item[key_item] = val_item
                    new_item["text"] = text_val
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            new_msg["content"] = new_content

        cleaned.append(cast(ChatCompletionMessageParam, new_msg))

    return cleaned


async def call_openai(
    client: AsyncOpenAI,
    model: str,
    messages: List[ChatCompletionMessageParam],
    max_retries: int = 5,
    initial_delay: float = 2,
    temperature: float = 0.1,
    max_tokens: int = 64,
    response_format: completion_create_params.ResponseFormat = {"type": "text"},
) -> ChatCompletion:
    """
    Универсальная функция для вызова OpenAI ChatCompletion с повторными попытками при возникновении
    RateLimitError (429), AuthenticationError (401) и других API-ошибках.

    Дополнительно очищает содержимое сообщений от контрольных символов и лишних пробелов,
    чтобы в GPT никогда не попадали "грязные" Unicode-символы.

    Args:
        client (AsyncOpenAI): Клиент OpenAI.
        model (str): Название модели для запроса.
        messages (List[ChatCompletionMessageParam]): Список сообщений для модели.
        max_retries (int): Максимальное число попыток.
        initial_delay (float): Начальная задержка перед повторной попыткой.
        temperature (float): Параметр температуры.
        max_tokens (int): Максимальное число токенов в ответе.
        response_format (completion_create_params.ResponseFormat): Формат ответа.

    Returns:
        ChatCompletion: Ответ от OpenAI Chat API.
    """
    # Очищаем сообщения перед отправкой запроса
    messages = clean_messages(messages)

    delay: float = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            response: ChatCompletion = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )

            # Логируем токены
            usage = response.usage
            if usage:
                logger.info(
                    f"[Модель {model}] Использованные токены: "
                    f"prompt_tokens={usage.prompt_tokens}, "
                    f"completion_tokens={usage.completion_tokens}, "
                    f"total_tokens={usage.total_tokens}"
                )

            return response
        except (RateLimitError, AuthenticationError) as err:
            if attempt < max_retries:
                logger.warning(
                    f"Ошибка {err.__class__.__name__}. Попытка {attempt}/{max_retries}. Повтор через {delay} сек..."
                )
                await asyncio.sleep(delay)
                delay *= 2
            else:
                logger.error("Достигнуто максимальное число повторов!")
                raise
        except APIError as err:
            logger.error(f"API Ошибка: {err}. Попытка {attempt}/{max_retries}.")
            if attempt < max_retries:
                await asyncio.sleep(delay)
                delay *= 2
            else:
                logger.error("Достигнут максимум попыток API.")
                raise

    raise RuntimeError("Не удалось получить ответ.")


async def process_table(
    df: pl.DataFrame,
    brand_column: str,
    description_column: Optional[str] = None,
    correction: bool = False,
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
        "Analyze the input text to identify brand names, product names, or any potential trademark-like terms. "
        "Provide multiple variations of the identified names, including:\n"
        "- The most likely correct spelling\n"
        "- Russian transliteration\n"
        "- English transliteration\n"
        "- Common alternative spellings\n"
        "- Additional variations if uncertain\n"
        "\n"
        "**Guidelines:**\n"
        "- If the input contains a recognizable brand or product name, extract it.\n"
        "- If not 100% sure, still include possible brand-like terms.\n"
        "- Always return multiple variations (max 6 per language).\n"
        "- Ensure both Russian and English versions are included.\n"
        "- Avoid empty arrays—if unsure, provide the most plausible brand-like terms.\n"
        "- Normalize spacing and remove unnecessary formatting.\n"
        "- Strictly return JSON in the required schema with no extra text.\n"
        "\n"
        "**Example responses:**\n"
        "\n"
        "For input: 'Найки'\n"
        "{\n"
        '    "original_text": "Найки",\n'
        '    "english_samples": ["Nike", "Naiki", "NIKE", "Naykee"],\n'
        '    "russian_samples": ["Найки", "Найк"]\n'
        "}\n"
        "\n"
        "For input: 'Адидас спорт'\n"
        "{\n"
        '    "original_text": "Адидас спорт",\n'
        '    "english_samples": ["Adidas", "Adidas Sport"],\n'
        '    "russian_samples": ["Адидас", "Адидас Спорт"]\n'
        "}\n"
        "\n"
        "For input: 'Samsung Electronics Co., Ltd.'\n"
        "{\n"
        '    "original_text": "Samsung Electronics Co., Ltd.",\n'
        '    "english_samples": ["Samsung", "Samsung Electronics"],\n'
        '    "russian_samples": ["Самсунг", "Самсунг Электроникс"]\n'
        "}\n"
        "\n"
        "For input: 'ООО Рога и Копыта'\n"
        "{\n"
        '    "original_text": "ООО Рога и Копыта",\n'
        '    "english_samples": ["Roga i Kopyta LLC"],\n'
        '    "russian_samples": ["Рога и Копыта"]\n'
        "}\n"
        "\n"
        "Respond strictly in JSON format following the provided schema. "
        f"You MUST return ONLY valid JSON with the following schema:\n"
        f"{json.dumps(BrandRecognitionResponse.model_json_schema(), indent=2)}\n"
        "No markdown fences. No extra text. Strictly output valid JSON or return an empty JSON object."
    )

    row_system_prompt: str = (
        "Correct the table row. Respond strictly in JSON format with the key 'corrected_row' following the provided schema:\n"
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
        text: str, description: Optional[str] = None, row_idx: int = 0
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
            text = f"{text}. Description: {description}"

        # Убираем все цифры и нормализуем пробелы
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Ограничиваем длину текста для GPT
        max_length = 2000
        if len(text) > max_length:
            text = text[:max_length] + "..."

        logger.info(f"Запрос (Строка {row_idx}): {text}")

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
        logger.info(f"Ответ (Строка {row_idx}): {raw_text}")

        # Обрабатываем JSON-ответ
        try:
            data = json.loads(raw_text)

            if "english_samples" not in data:
                logger.warning(
                    f"[Строка {row_idx}] 'english_samples' отсутствует в ответе!"
                )
                data["english_samples"] = []
            if "russian_samples" not in data:
                logger.warning(
                    f"[Строка {row_idx}] 'russian_samples' отсутствует в ответе!"
                )
                data["russian_samples"] = []

            return BrandRecognitionResponse.model_validate(data)

        except Exception as e:
            logger.error(f"[Строка {row_idx}] Ошибка обработки JSON-ответа: {e}")
            logger.error(f"[Строка {row_idx}] Некорректный ответ: {raw_text}")
            return BrandRecognitionResponse(
                english_samples=[], russian_samples=[], original_text=text
            )

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
            max_tokens=384,
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
        logger.info(f"=== Обрабатываем строку {idx}: {row_dict}")

        # 1) Проверяем, исключена ли ТМ (анализируем всю строку)
        combined_text = " ".join(
            str(val) for val in row_dict.values() if val is not None
        )
        if is_excluded(combined_text):
            logger.info(f"[Строка {idx}] Исключена (содержит 'исключён'). Пропускаем.")
            row_dict["Исключено"] = "Да"
            processed_rows.append({k: (v or "") for k, v in row_dict.items()})
            continue  # 🚀 Теперь строка точно пропускается без вызова GPT!

        # 2) Обрабатываем только столбец brand_column (если не исключено)
        brand_val = row_dict.get(brand_column)
        description_val = (
            row_dict.get(description_column) if description_column else None
        )

        if isinstance(brand_val, str) and brand_val.strip():
            brand_resp = await recognize_brand(brand_val, description_val, row_idx=idx)

            # Формируем итоговое значение
            combined_brand_info = " ".join(
                [
                    brand_resp.original_text,
                    ", ".join(brand_resp.english_samples),
                    ", ".join(brand_resp.russian_samples),
                ]
            ).strip()
            row_dict[brand_column] = combined_brand_info

        # 3) Добавляем 'Исключено' = 'Нет'
        row_dict["Исключено"] = "Нет"

        # 4) Коррекция строки (если включена)
        if correction:
            try:
                corrected_resp = await correct_row(row_dict)
                processed_rows.append(corrected_resp.corrected_row)
            except Exception as e:
                logger.error(f"[Строка {idx}] Ошибка при корректировке: {e}")
                logger.error(f"[Строка {idx}] Данные перед корректировкой: {row_dict}")
                processed_rows.append({k: str(v or "") for k, v in row_dict.items()})
        else:
            processed_rows.append({k: str(v or "") for k, v in row_dict.items()})

    return pl.DataFrame(processed_rows)
