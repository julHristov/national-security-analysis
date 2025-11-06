import json
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)


# Loads JSON file as a string name and returns a dictionary
# With logger logs/shows excess or error when file is not found or invalid
def load_schema(filename: str) -> dict:
    schema_path = Path(__file__).resolve().parent.parent / "schema" / filename
    logger.debug(f"Attempting to load schema from: {schema_path.resolve()}")

    if not schema_path.exists():
        logger.error(f"❌ Schema file not found: {filename}")
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
            logger.info(f"✅ Successfully loaded schema: {filename}")
            return schema
    except FileNotFoundError:
        logger.error(f"❌ Schema file not found: {filename}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in {schema_path.resolve()}: {e}")
        return {}
