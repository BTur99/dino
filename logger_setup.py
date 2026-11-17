# logger_setup.py
import logging
import sys

def setup_logger():
    # Создаем логгер с именем "DinoRL"
    logger = logging.getLogger("DinoRL")
    logger.setLevel(logging.DEBUG)

    # Если логгер уже настроен, не добавляем дубликаты
    if logger.hasHandlers():
        return logger

    # Формат: [Время] [Уровень] Сообщение
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

    # 1. Запись в файл (все уровни, включая отладку)
    file_handler = logging.FileHandler('app.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 2. Запись в консоль (только INFO и выше, чтобы не засорять экран)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Создаем глобальный логгер, который будем импортировать
logger = setup_logger()