from logging import INFO, Logger, getLogger
from typing import Iterable, Optional


class MultilineLogger:
    """
    Логгер для вывода многострочных сообщений.

    Каждая строка переданного сообщения логируется отдельно, что позволяет избежать
    проблем с форматированием многострочных логов.
    """

    level: int
    logger: Logger

    __slots__ = ("level", "logger")

    def __init__(self, level: int = INFO, logger: Optional[Logger] = None) -> None:
        """
        Инициализация многострочного логгера.

        Args:
            level (int): Уровень логирования (по умолчанию INFO).
            logger (Optional[Logger]): Экземпляр логгера (если не указан, используется стандартный).
        """
        self.level = level
        self.logger = logger or getLogger()

    def __call__(self, message: Iterable[str]) -> None:
        """
        Логирует каждую строку переданного сообщения отдельно.

        Args:
            message (Iterable[str]): Многострочное сообщение (список строк или строка).
        """
        if isinstance(message, str):
            message = message.splitlines()
        for line in message:
            self.logger.log(level=self.level, msg=line)
