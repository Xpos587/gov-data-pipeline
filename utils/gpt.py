from __future__ import annotations

import re
import base64
import json
import logging
import asyncio
from io import BytesIO
from typing import Dict, List, Optional

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


class BrandRecognitionResponse(BaseModel):
    english_samples: List[str] = Field(
        ..., description="Samples of brand names in English (max 4)."
    )
    russian_samples: List[str] = Field(
        ..., description="Samples of brand names in Russian (max 4)."
    )


class RowCorrectionResponse(BaseModel):
    corrected_row: Dict[str, str] = Field(
        ..., description="Corrected row (column -> value dictionary)."
    )


def is_excluded(row_text: str) -> bool:
    """
    Проверяет, содержит ли строка слово "исключен" (или "исключён") как отдельное слово.
    Игнорирует регистр, символ "ё" заменяет на "е" и допускает наличие произвольных пробелов между буквами.
    Не воспринимает слова типа "исключение", "исключением" и т.д.
    """
    # Приводим текст к нижнему регистру и заменяем "ё" на "е"
    normalized_text = row_text.casefold().replace("ё", "е")
    # Строим паттерн для слова "исключен" с допускаемыми пробелами между буквами
    word = "исключен"
    # Паттерн: перед словом не должно быть буквы, после слова – тоже (чтобы исключить расширенные формы)
    pattern = r"(?<![а-я])" + r"\s*".join(list(word)) + r"(?![а-я])"
    return re.search(pattern, normalized_text, flags=re.IGNORECASE) is not None


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


async def recognize_image(client: AsyncOpenAI, base64_image: str) -> str:
    """
    Использует GPT Vision для распознавания текста из изображения.
    """
    response = await call_openai(
        client,
        model=settings.openai.image_model,
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


async def process_table(
    df: pl.DataFrame,
    brand_column: str,
    description_column: Optional[str] = None,
    image_column: str = "Изображение",
    correction: bool = False,
) -> pl.DataFrame:
    """
    Обрабатывает DataFrame, распознаёт бренды, корректирует строки и обрабатывает изображения.

    1. Проверяет, не содержит ли строка ключевые слова "исключён"/"исключен".
       Если содержит — помечает строку, добавляя поле "Исключено = 'Да'" и пропускает дальнейшую обработку.

    2. Если строка не исключена:
       - Если столбец с названием бренда пуст, но есть изображение, распознаёт текст из изображения
         и сохраняет его в столбце бренда с припиской "(RECOG)".
       - Если бренд указан, но есть изображение, оставляет данные без изменений.

    3. Вызывает `recognize_brand` для указанного столбца с брендом и, при наличии, столбца с описанием.
       Заменяет значение бренда на (original_text + english_samples + russian_samples).

    4. Вызывает `correct_row` для итоговых данных и сохраняет результат.

    Args:
        df (pl.DataFrame): Исходный DataFrame с данными.
        brand_column (str): Название столбца, содержащего торговую марку.
        description_column (Optional[str]): Название столбца с описанием фирмы для улучшения распознавания.
        image_column (str): Название столбца с изображением в формате base64.
        correction (bool): Флаг включения корректировки строки.

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
        '    "english_samples": ["Nike", "Naiki", "NIKE", "Naykee"],\n'
        '    "russian_samples": ["Найки", "Найк"]\n'
        "}\n"
        "\n"
        "For input: 'Адидас спорт'\n"
        "{\n"
        '    "english_samples": ["Adidas", "Adidas Sport"],\n'
        '    "russian_samples": ["Адидас", "Адидас Спорт"]\n'
        "}\n"
        "\n"
        "For input: 'DEERMA'\n"
        "{\n"
        '    "english_samples": ["Dirma", "Derma"],\n'
        '    "russian_samples": ["Дирма", "Дерма", "Деерма"]\n'
        "}\n"
        "\n"
        "For input: 'Samsung Electronics Co., Ltd.'\n"
        "{\n"
        '    "english_samples": ["Samsung", "Samsung Electronics"],\n'
        '    "russian_samples": ["Самсунг", "Самсунг Электроникс"]\n'
        "}\n"
        "\n"
        "For input: 'ООО Рога и Копыта'\n"
        "{\n"
        '    "english_samples": ["Roga i Kopyta LLC"],\n'
        '    "russian_samples": ["Рога и Копыта"]\n'
        "}\n"
        "\n"
        "Respond strictly in JSON format following the provided schema. "
        f"You MUST return ONLY valid JSON with the following schema:\n"
        f"{json.dumps(BrandRecognitionResponse.model_json_schema(), indent=2)}\n"
        "No markdown fences. No extra text. Strictly output valid JSON or return an empty JSON object."
    )

    async def gen_brand_samples(
        prompt: str, description: Optional[str] = None, row_idx: int = 0
    ) -> BrandRecognitionResponse:
        """
        Распознаёт бренд, возвращая структуру BrandRecognitionResponse.
        Если найдена base64-строка изображения, распознаёт и заменяет её на извлечённый текст.
        В случае, если description_column указан, добавляет его текст к тексту для лучшего распознавания.
        """
        # Если описание присутствует, добавляем его
        if description:
            prompt = f"{prompt}. Description: {description}"

        # Убираем все цифры
        prompt = re.sub(r"\d+", "", prompt)

        # Нормализуем пробелы
        prompt = re.sub(r"\s+", " ", prompt).strip()

        # Удаляем токены, состоящие только из спецсимволов
        prompt = " ".join(
            token
            for token in prompt.split()
            if not re.fullmatch(r"[!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~]+", token)
        )

        # Ограничиваем длину текста для GPT
        max_length = 2000
        if len(prompt) > max_length:
            prompt = prompt[:max_length] + "..."

        logger.info(f"Запрос (Строка {row_idx}): {prompt}")

        response = await call_openai(
            client,
            model=settings.openai.brand_model,
            messages=[
                {"role": "system", "content": brand_system_prompt},
                {"role": "user", "content": prompt},
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
            return BrandRecognitionResponse(english_samples=[], russian_samples=[])

    async def correct_row(row_data: Dict[str, Optional[str]]) -> RowCorrectionResponse:
        """
        Корректирует строку, возвращая структуру RowCorrectionResponse.
        """

        row_system_prompt: str = (
            "Correct the table row. Respond strictly in JSON format with the key 'corrected_row' following the provided schema:\n"
            f"{json.dumps(RowCorrectionResponse.model_json_schema(), indent=2)}"
        )

        logger.debug(f"Запрос: {json.dumps(row_data, ensure_ascii=False)}")

        response = await call_openai(
            client,
            model=settings.openai.correct_model,
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

        # Улучшаем вывод для логгера
        modified_row = {}
        for key, val in row_dict.items():
            if key == image_column and isinstance(val, str) and len(val) > 100:
                modified_row[key] = val[:50] + " ... " + val[-50:]
            else:
                modified_row[key] = val
        logger.info(f"=== Обрабатываем строку {idx}: {modified_row}")

        # 1) Проверяем, исключена ли ТМ (анализируем всю строку)
        combined_text = " ".join(
            str(val) for val in row_dict.values() if val is not None
        )
        if is_excluded(combined_text):
            logger.info(f"[Строка {idx}] Исключена. Пропускаем.")
            row_dict["Исключено"] = "Да"
            processed_rows.append({k: (v or "") for k, v in row_dict.items()})
            continue  # Строка пропускается без вызова GPT!

        # 2) Обработка столбца с изображением
        brand_val = row_dict.get(brand_column)
        image_val = row_dict.get(image_column)

        if not brand_val and image_val:
            try:
                # Распознаём текст из изображения
                recognized_text = await recognize_image(client, image_val)
                if recognized_text:
                    # Добавляем распознанный текст в столбец бренда с припиской "(RECOG)"
                    row_dict[brand_column] = f"{recognized_text} (RECOG)"
                    logger.info(
                        f"[Строка {idx}] Распознан бренд из изображения: {recognized_text}"
                    )
            except Exception as e:
                logger.error(f"[Строка {idx}] Ошибка распознавания изображения: {e}")

        # 3) Распознаём бренд
        brand_val = row_dict.get(
            brand_column
        )  # Обновляем значение бренда после обработки изображения
        if isinstance(brand_val, str) and brand_val.strip():
            # Удаляем приписку "(RECOG)", если она есть, чтобы не передавать её в модель
            plain_brand = brand_val.replace(" (RECOG)", "").strip()
            description_val = (
                row_dict.get(description_column) if description_column else None
            )
            brand_resp = await gen_brand_samples(
                plain_brand, description_val, row_idx=idx
            )

            # Формируем итоговые столбцы
            row_dict["Вариации бренда на англ. языке"] = ", ".join(
                brand_resp.english_samples
            )
            row_dict["Вариации бренда на рус. языке"] = ", ".join(
                brand_resp.russian_samples
            )

        # 4) Добавляем 'Исключено' = 'Нет'
        row_dict["Исключено"] = "Нет"

        # 5) Коррекция строки (если включена)
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
