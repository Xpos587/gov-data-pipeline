import os
from io import BytesIO
import logging
import time
from typing import Optional, Type
import paramiko
from paramiko.sftp_client import SFTPClient
from utils.settings import settings

logger = logging.getLogger(__name__)


class SFTPHandler:
    def __init__(self) -> None:
        """
        Инициализация SFTP обработчика.
        Загружает настройки из `settings.py`.
        """
        if not settings.sftp:
            raise ValueError("Настройки SFTP отсутствуют в конфигурации.")

        sftp_config = settings.sftp
        self.host: str = sftp_config.host
        self.port: int = sftp_config.port
        self.username: str = sftp_config.user
        self.password: str = sftp_config.passwd.get_secret_value()
        self.remote_dir: str = sftp_config.remote_dir
        self.max_retries: int = 3
        self.retry_delay: int = 5
        self.sftp: Optional[SFTPClient] = None
        self.transport: Optional[paramiko.Transport] = None

    def connect(self) -> bool:
        """
        Устанавливает соединение с SFTP сервером.

        Returns:
            bool: True, если соединение успешно, иначе False.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Попытка подключения к SFTP серверу: {self.host} (попытка {attempt})"
                )
                self.transport = paramiko.Transport((self.host, self.port))
                self.transport.connect(username=self.username, password=self.password)
                self.sftp = paramiko.SFTPClient.from_transport(self.transport)
                logger.info("Успешное подключение к SFTP серверу")
                return True
            except paramiko.SSHException as e:
                logger.error(f"Ошибка подключения к SFTP серверу: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        logger.error("Не удалось подключиться к SFTP серверу после нескольких попыток")
        return False

    def ensure_connection(self) -> SFTPClient:
        """
        Проверяет и восстанавливает соединение с SFTP сервером.

        Returns:
            SFTPClient: Объект SFTP клиента с гарантией, что он не None.
        """
        if not self.sftp or not self.transport:
            logger.warning(
                "Соединение с SFTP сервером потеряно. Попытка переподключения..."
            )
            if not self.connect():
                raise ConnectionError(
                    "Не удалось установить соединение с SFTP сервером."
                )

        if self.sftp is None:
            raise RuntimeError("SFTP клиент не инициализирован после подключения.")

        return self.sftp  # Теперь гарантировано, что self.sftp не None

    def upload_bytes(self, file_bytes: bytes, file_name: str) -> None:
        """
        Загружает файл в формате байтов на SFTP сервер.

        Args:
            file_bytes (bytes): Содержимое файла.
            file_name (str): Имя файла для сохранения.
        """
        sftp = self.ensure_connection()  # Локальная переменная с уточнённым типом

        remote_path = os.path.join(self.remote_dir, file_name)
        remote_dir = os.path.dirname(remote_path)

        try:
            self._create_remote_dirs(sftp, remote_dir)
            with BytesIO(file_bytes) as buffer:
                sftp.putfo(buffer, remote_path)
            logger.info(f"Файл успешно загружен на SFTP: {remote_path}")
        except paramiko.SSHException as e:
            logger.error(f"Ошибка при загрузке файла на SFTP: {e}")
            raise

    def _create_remote_dirs(self, sftp: SFTPClient, remote_dir: str) -> None:
        """
        Создаёт директории на сервере, если их не существует.

        Args:
            sftp (SFTPClient): Объект SFTP клиента.
            remote_dir (str): Путь к директории.
        """
        try:
            dirs = remote_dir.strip("/").split("/")
            current_dir = ""
            for d in dirs:
                current_dir = f"{current_dir}/{d}" if current_dir else f"/{d}"
                try:
                    sftp.stat(current_dir)
                except FileNotFoundError:
                    sftp.mkdir(current_dir)
        except paramiko.SSHException as e:
            logger.error(f"Ошибка при создании директории: {e}")
            raise

    def close(self) -> None:
        """
        Закрывает соединение с SFTP сервером.
        """
        if self.sftp:
            self.sftp.close()
        if self.transport:
            self.transport.close()
        logger.info("Соединение с SFTP сервером закрыто")

    def __enter__(self) -> "SFTPHandler":
        """
        Поддержка контекстного менеджера.
        """
        if not self.connect():
            raise ConnectionError("Не удалось подключиться к SFTP серверу")
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Type[BaseException]],
    ) -> None:
        """
        Закрытие соединения при выходе из контекстного менеджера.
        """
        self.close()
