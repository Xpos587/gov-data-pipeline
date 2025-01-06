import logging
from .multiline import MultilineLogger

__all__ = ["setup_logger", "MultilineLogger"]


def setup_logger(level: int = logging.INFO) -> None:
    """
    Настройка основного логгера приложения.

    Устанавливает уровень логирования и формат сообщений для стандартных логгеров.
    """
    for name in ["aiohttp.access"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s | %(name)s: %(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
        level=level,
    )
