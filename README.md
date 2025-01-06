# Gov Data Pipeline

## Описание проекта

Этот проект автоматизирует сбор и обработку данных из официальных источников
(Беларусь, Казахстан, Киргизия). Основная цель — еженедельное получение данных,
их преобразование в унифицированный формат таблиц Excel, обработка изображений
и корректировка текста с использованием LLM.

---

## Основные задачи

### Беларусь

- **Скачивание данных**:  
  [https://www.customs.gov.by/zashchita-prav-na-obekty-intellektualnoy-sobstvennosti](https://www.customs.gov.by/zashchita-prav-na-obekty-intellektualnoy-sobstvennosti).
- Обработка таблиц и корректировка текста.

**Примечание**: Обработка изображений для Беларуси не реализована из-за слишком
неточного расположения изображений в исходных файлах.

### Казахстан

- **Скачивание данных**:  
  [https://kgd.gov.kz/ru/content/tamozhennyy-reestr-obektov-intellektualnoy-sobstvennosti-1](https://kgd.gov.kz/ru/content/tamozhennyy-reestr-obektov-intellektualnoy-sobstvennosti-1).
- Сбор изображений из таблиц, их конвертация в PNG и сохранение в формате:  
  `data:image/png;base64,{base64_image}`
- Конвертация base64 строк в текст с помощью OCR. Если текст отсутствует, строка
  удаляется. Если текст есть, строка заменяется на распознанный текст.

### Киргизия

- **Скачивание данных**:  
  [https://www.customs.gov.kg/article/get?id=46&lang=ru](https://www.customs.gov.kg/article/get?id=46&lang=ru).
- Обработка данных в формате PDF не поддерживается.

---

## Коррекция текста с использованием LLM

Используется модель `gpt-4o-mini` для исправления текста, пострадавшего при
конвертации (например, разрывы строк, опечатки). На текущий момент функционал
реализован, но не подключен.

---

## Ограничения

1. **Python 3.12**: Код не тестировался на версиях ниже 3.12.
2. **Изображения**:
   - Проблемы с обработкой из-за различий в форматах (PNG, WFM, EFM).
   - Неструктурированное расположение в файлах.
3. **PDF**: Обработка для Киргизии отсутствует.

---

## Установка

### Системные требования

- **Python**: 3.12 или выше.
- **Tesseract OCR**: Установите для работы с распознаванием текста.
- **Зависимости**: Все указаны в `requirements.txt`.

### Установка Tesseract OCR

#### Arch Linux

```bash
sudo pacman -S tesseract tesseract-data-eng tesseract-data-rus
```

#### Ubuntu/Debian

```bash
sudo apt update
sudo apt install tesseract-ocr -y
sudo apt install libtesseract-dev -y
```

#### macOS

```bash
brew install tesseract
```

#### Windows

1. Скачайте [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract).
2. Убедитесь, что `tesseract.exe` находится в PATH.

---

### Установка зависимостей

1. Создайте виртуальное окружение:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Установите зависимости:

   ```bash
   pip install -r requirements.txt
   ```

---

## Настройка

1. Склонируйте репозиторий:

   ```bash
   git clone https://github.com/Xpos587/gov-data-pipeline.git
   cd gov-data-pipeline
   ```

2. Создайте файл `.env`:

   ```bash
   cp .env.dist .env
   ```

3. Укажите настройки в `.env`:
   - **SFTP**: Для выгрузки обработанных файлов.
   - **OpenAI API**: Получите API-ключ OpenAI здесь:  
     [https://platform.openai.com/settings/organization/api-keys](https://platform.openai.com/settings/organization/api-keys).

---

## Использование

1. Убедитесь, что `.env` настроен.
2. Запустите приложение:

   ```bash
   python main.py
   ```

---

## Архитектура проекта

- **handlers**: Обработчики для Беларуси, Казахстана и Киргизии.
- **utils**: Утилиты для OCR, логирования, загрузки через SFTP и работы с настройками.
- **settings**: Конфигурация приложения.
- **main.py**: Точка входа.

---

## Возможные доработки

1. Поддержка обработки PDF для Киргизии.
2. Улучшение работы с изображениями.
3. Интеграция корректировки текста с использованием OpenAI.
4. Оптимизация кода для повышения производительности.

---

## Лицензия

Проект распространяется под лицензией MIT.
