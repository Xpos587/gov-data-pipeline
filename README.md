# Gov Data Pipeline

Автоматизированная система для регулярного сбора и обработки данных из официальных таможенных реестров Беларуси, Казахстана и Киргизии.
Основная цель — еженедельное получение данных, преобразование их в унифицированный Excel-формат, анализ изображений (OCR/GPT) и корректировка текста с помощью LLM (GPT).

---

## Описание проекта

1. **Беларусь**

   - **URL для скачивания:** [https://www.customs.gov.by/zashchita-prav-na-obekty-intellektualnoy-sobstvennosti](https://www.customs.gov.by/zashchita-prav-na-obekty-intellektualnoy-sobstvennosti)
   - Обработчик: `BelarusHandler`
   - Особенности:
     - Автоматическое обнаружение XLSX-файлов по шаблону
     - Интеграция обработки изображений через GPT Vision
     - Коррекция текстовых полей с помощью LLM

2. **Казахстан**

   - **URL для скачивания:** [https://kgd.gov.kz/ru/content/tamozhennyy-reestr-obektov-intellektualnoy-sobstvennosti-1](https://kgd.gov.kz/ru/content/tamozhennyy-reestr-obektov-intellektualnoy-sobstvennosti-1)
   - Обработчик: `KazakhstanHandler`
   - Особенности:
     - Автоматическое выравнивание структуры таблиц
     - Препроцессинг изображений с определением координат
     - Нормализация Unicode-текста

3. **Киргизия**
   - **URL для скачивания:** [https://www.customs.gov.kg/site/ru/master/customskg/intellektualdyk-menchik-ukuktaryn-korgoo](https://www.customs.gov.kg/site/ru/master/customskg/intellektualdyk-menchik-ukuktaryn-korgoo)
   - Обработчик: `KyrgyzstanHandler`
   - Особенности:
     - Конвертация PDF в DOCX через встроенный модуль
     - Автоматическое извлечение таблиц из документов Word
     - Обработка многострочных записей

---

## Коррекция текста и GPT

1. **Интеграция с OpenAI API**
   - Автоматическая коррекция орфографии и форматирования
   - Распознавание брендов из текстовых описаний
   - Генерация альтернативных написаний торговых марок

2. **Обработка изображений**
   - Конвертация вложенных изображений в Base64
   - Анализ графических элементов через GPT Vision
   - Совмещение текстовых и графических данных

3. **Унификация данных**
   - Приведение всех источников к единому формату
   - Автоматическое определение классов товаров
   - Нормализация кодов ТН ВЭД

---

## Установка

### Системные требования

- **Python**: 3.12 или выше.
- **Зависимости**: перечислены в `environment.yml` или `requirements.txt`.

   ```bash
   python -m venv venv
   source venv/bin/activate     # Linux/Mac
   venv\Scripts\activate        # Windows
   ```

2. Установите пакеты:

   ```bash
   pip install -r requirements.txt
   ```

_(При желании можно использовать `environment.yaml` вместе с Conda/Mamba.)_

---

## Настройка

1. Клонируйте репозиторий:

   ```bash
   git clone https://github.com/Xpos587/gov-data-pipeline.git
   cd gov-data-pipeline
   ```

2. Создайте файл `.env` на базе `.env.dist`:

   ```bash
   cp .env.dist .env
   ```

3. Укажите нужные параметры в `.env`:
   - **FTP**: параметры сервера, куда выгружаются итоговые Excel-файлы (`SFTP_HOST`, `SFTP_PORT`, `SFTP_USER` и т.д.).
   - **OpenAI API**: для GPT (`OPENAI_BASE_URL`, `OPENAI_API_KEY`).

---

## Использование

1. Убедитесь, что `.env` корректно заполнен (OpenAI и SFTP).
2. Запустите `main.py`:

   ```bash
   python main.py
   ```

3. Логи будут в консоли (на русском языке). При превышении лимитов (429) или ошибках аутентификации (401) скрипт повторит запросы автоматически.

---

## Архитектура проекта

- **handlers**
  Включает классы для каждой страны (`BelarusHandler`, `KazakhstanHandler`, `KyrgyzstanHandler`).
  Каждый хендлер умеет скачивать и преобразовывать данные.

- **utils**

  - **gpt.py**: функции для работы с GPT (распознавание брендов, корректировка строк, обработка изображений).
  - **settings.py**: конфигурации FTP и OpenAI.
  - **ftp.py**: загрузка результатов на FTP.
  - **loggers**: настройки логирования.

- **main.py**
  Точка входа. Инициализирует все хендлеры, обрабатывает результаты, при необходимости выгружает на SFTP.

---

## Лицензия

Проект распространяется под лицензией [MIT](LICENSE).
