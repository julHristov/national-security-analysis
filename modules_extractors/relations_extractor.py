import json
import re
from pathlib import Path
from utils.logger import get_logger
from modules_extractors.schema_loader import load_schema
from config import CLEAN_DIR, RELATIONS_DIR

# Initialisation of the logger for this modul
logger = get_logger(__name__)

"""Dummy function: later we will do it more intelligent and NLP-based.
For now: searches for predefined relations in the text (from relations_type.json)
and returns list of dictionaries [{"type": "...", "subtype": "...", "text": "..."}]"""


def extract_relations(text: str, relation_schema: dict) -> list:
    if not text:
        logger.warning("⚠️ Empty text provided for relations extraction.")
        return []

    relations = []
    logger.debug("Starting relations extraction...")

    for rel_type, info in relation_schema.items():
        examples = info.get("examples", [])
        for ex in examples:
            pattern = re.compile(rf"\b{re.escape(ex)}\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                relations.append({
                    "relation_text": match.group(0),
                    "relation_type": rel_type,
                    "start": match.start(),
                    "end": match.end()
                })
    logger.info(f"✅ Extracted {len(relations)} relations from text.")
    if not relations:
        logger.warning("⚠️ No relations extracted from text.")
    return relations


# Processes all files from data/clean_texts and retrieves relations.
def process_all_files():
    logger.info(">>> Relations extractor started...")

    relations_schema = load_schema("relation_types.json")
    if not relations_schema:
        logger.error("❌ Relations schema is empty. Aborting extraction.")
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

        logger.info(f" Retrieved {len(relations)} relations from {file_path.name}")


if __name__ == '__main__':
    process_all_files()