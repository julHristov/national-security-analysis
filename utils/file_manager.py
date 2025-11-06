import json
from pathlib import Path
from utils.logger import get_logger
from config import BASE_DIR

logger = get_logger(__name__)


def get_path(*parts):
    """Generates a full path compared to BASE_DIR."""
    return Path(BASE_DIR, *parts)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file deos not exist: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text_file(file_path: Path) -> str:
    """Прочита текстов файл с UTF-8 енкодинг."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"❌ Грешка при четене на {file_path}: {e}")
        return ""


def read_json_file(file_path: Path) -> dict:
    """Чете данни от JSON файл."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Грешка при четене на JSON от {file_path}: {e}")
        return {}


def write_json_file(data, file_path: Path):
    """Записва данни в JSON файл с четим формат."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ JSON записан: {file_path}")
    except Exception as e:
        logger.error(f"❌ Грешка при запис в {file_path}: {e}")
