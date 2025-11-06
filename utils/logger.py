"""Here is defined a function get_logger(name) — it returns a configured logger for the specific module.
Each script (for example schema_loader.py, entity_extractor.py, relations_extractor.py) will be able to use it
"""
import logging
from pathlib import Path

"""Returns a logger instance that writes both to console
and to a file. Each module can get its own logger using: get_logger(__name__)"""


def get_logger(name: str):
    # Creates the directory for logs - in the root file of the project
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / "project.log"

    # Messages format
    formater = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler - writes in logs / project.log
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formater)
    file_handler.setLevel(logging.INFO)

    # Console handler - shows message in the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formater)
    console_handler.setLevel(logging.INFO)

    # Creating of the logger itself
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Adding the handlers if they are not added
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
