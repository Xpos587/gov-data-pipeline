import os
from io import BytesIO
import logging
import time
from typing import Optional, Type, cast
from ftplib import FTP, error_perm, error_temp
from utils.settings import settings

logger = logging.getLogger(__name__)


class FTPHandler:
    def __init__(self) -> None:
        if not settings.ftp:
            raise ValueError("Настройки FTP отсутствуют в конфигурации.")

        ftp_config = settings.ftp
        self.host: str = ftp_config.host
        self.port: int = ftp_config.port
        self.username: str = ftp_config.user
        self.password: str = ftp_config.passwd.get_secret_value()
        self.remote_dir: str = ftp_config.remote_dir
        self.max_retries: int = 3
        self.retry_delay: int = 5
        self.ftp: Optional[FTP] = None

    def connect(self) -> bool:
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Попытка подключения: {self.host} (попытка {attempt})")
                self.ftp = FTP()
                self.ftp.connect(self.host, self.port, timeout=30)
                self.ftp.login(self.username, self.password)
                logger.info("Успешное подключение")
                return True
            except (error_temp, error_perm) as e:
                logger.error(f"Ошибка подключения: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        logger.error("Не удалось подключиться")
        return False

    def ensure_connection(self) -> FTP:
        if not self.ftp:
            logger.warning("Соединение потеряно. Переподключение...")
            if not self.connect():
                raise ConnectionError("Не удалось восстановить соединение")
        return cast(FTP, self.ftp)

    def upload_bytes(self, file_bytes: bytes, file_name: str) -> None:
        ftp = self.ensure_connection()
        remote_path = os.path.join(self.remote_dir, file_name).replace("\\", "/")

        try:
            self._create_remote_dirs(ftp, self.remote_dir)
            with BytesIO(file_bytes) as buffer:
                ftp.storbinary(f"STOR {remote_path}", buffer)
            logger.info(f"Файл загружен: {remote_path}")
        except error_perm as e:
            logger.error(f"Ошибка загрузки: {e}")
            raise

    def _create_remote_dirs(self, ftp: FTP, path: str) -> None:
        try:
            current_dir = ""
            for part in path.strip("/").split("/"):
                current_dir = f"{current_dir}/{part}" if current_dir else part
                try:
                    ftp.cwd(current_dir)
                except error_perm:
                    ftp.mkd(current_dir)
                    ftp.cwd(current_dir)
        except error_perm as e:
            logger.error(f"Ошибка создания директории: {e}")
            raise

    def close(self) -> None:
        if self.ftp:
            self.ftp.quit()
        logger.info("Соединение закрыто")

    def __enter__(self) -> "FTPHandler":
        if not self.connect():
            raise ConnectionError("Ошибка подключения")
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Type[BaseException]],
    ) -> None:
        self.close()

