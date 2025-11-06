import json
import re
from pathlib import Path
from config import CLEAN_DIR, RELATIONS_DIR
from schema.schema_loader import load_relation_schema


def get_logger(name):
    class SimpleLogger:
        def info(self, msg): print(f"INFO: {msg}")

        def warning(self, msg): print(f"WARNING: {msg}")

        def error(self, msg): print(f"ERROR: {msg}")

        def debug(self, msg): pass  # Добавен debug метод

    return SimpleLogger()


logger = get_logger(__name__)


def extract_relations(text: str, relation_schema: dict) -> list:
    """Extract relations from text using pattern matching"""
    if not text:
        logger.warning("Empty text provided for relations extraction.")
        return []

    relations = []
    logger.info("Starting relations extraction...")  # Променено от debug на info

    for rel_type, info in relation_schema.items():
        examples = info.get("examples", [])
        for ex in examples:
            pattern = re.compile(rf"\b({re.escape(ex)})\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                relations.append({
                    "relation_text": match.group(0),
                    "relation_type": rel_type,
                    "start": match.start(),
                    "end": match.end()
                })

    logger.info(f"Extracted {len(relations)} relations from text.")
    if not relations:
        logger.warning("No relations extracted from text.")

    return relations


def process_all_files():
    """Main function to process all files"""
    logger.info(">>> Relations extractor started...")

    # Use the existing schema loader
    relations_schema = load_relation_schema()

    if not relations_schema:
        logger.error("✗ Relations schema is empty. Aborting extraction.")
        return

    RELATIONS_DIR.mkdir(parents=True, exist_ok=True)

    if not any(CLEAN_DIR.glob("*.txt")):
        logger.warning(f"No text files in {CLEAN_DIR}.")
        return

    for file_path in CLEAN_DIR.glob("*.txt"):
        logger.info(f"Processing file: {file_path.name}")
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        relations = extract_relations(text, relations_schema)
        output_path = RELATIONS_DIR / f"{file_path.stem}_relations.json"

        with open(output_path, 'w', encoding='utf-8') as out:
            json.dump(relations, out, ensure_ascii=False, indent=2)

        logger.info(f"Retrieved {len(relations)} relations from {file_path.name}")


if __name__ == '__main__':
    process_all_files()