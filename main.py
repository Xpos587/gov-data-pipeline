import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, IO
from io import BytesIO
from polars import DataFrame

# from utils.sftp import SFTPHandler
from utils.settings import settings
from utils.loggers import setup_logger
from handlers import handlers_manager

# Настройка логгера
setup_logger()
logger = logging.getLogger(__name__)


async def main() -> None:
    """
    Главная функция для запуска обработчиков и загрузки файлов на SFTP.
    """
    logger.info("Начало обработки...")

    if not settings.sftp:
        logger.error("Настройки SFTP не найдены. Завершаем работу.")
        return

    # Получаем результаты обработки от всех обработчиков
    results: Dict[str, Optional[DataFrame]] = await handlers_manager.process_all()

    try:
        # with SFTPHandler() as sftp:
        for name, result in results.items():
            if result is None:
                logger.warning(f"Обработчик {name} не вернул данные.")
                continue

            try:
                logger.info(f"Преобразование данных обработчика {name} в Excel...")
                buffer: IO[bytes] = BytesIO()
                result.write_excel(workbook=buffer)  # type: ignore
                buffer.seek(0)
                # file_bytes: bytes = buffer.read()

                # Генерируем имя файла для загрузки
                file_name = f"{name}_{datetime.now().strftime('%Y-%m-%d')}.xlsx"

                result.write_excel(file_name)  # type: ignore
                # Загружаем файл на SFTP
                # sftp.upload_bytes(file_bytes, file_name)
            except Exception as e:
                logger.error(f"Ошибка при обработке данных обработчика {name}: {e}")
    except Exception as e:
        logger.error(f"Общая ошибка при загрузке на SFTP: {e}")

    logger.info("Обработка завершена.")


if __name__ == "__main__":
    asyncio.run(main())
