import spacy
import json
from pathlib import Path
from collections import defaultdict, Counter
import time
from tqdm import tqdm

# Import config - will create this file next
try:
    from config import CLEAN_DIR, ANNOTATED_DIR, SPACY_MODEL

    ENTITIES_DIR = Path(ANNOTATED_DIR) / "entities"
except ImportError:
    # Fallback config
    CLEAN_DIR = "data/clean_texts"
    ANNOTATED_DIR = "data/annotated"
    ENTITIES_DIR = Path(ANNOTATED_DIR) / "entities"
    SPACY_MODEL = "en_core_web_md"


def get_logger(name):
    """Simple logger implementation"""

    class SimpleLogger:
        def info(self, msg): print(f"INFO: {msg}")

        def warning(self, msg): print(f"WARNING: {msg}")

        def error(self, msg): print(f"ERROR: {msg}")

        def debug(self, msg): pass

    return SimpleLogger()


logger = get_logger(__name__)


def load_schema(schema_name):
    """Load entity schema from JSON file"""
    schema_path = Path("schema") / schema_name
    if schema_path.exists():
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def extract_entities(text: str, entity_schema: dict) -> list:
    """Extract entities from text using spaCy NER and rule-based matching"""
    if not text:
        logger.warning("Empty text provided for entity extraction.")
        return []

    entities = []
    logger.debug("Starting entity extraction...")

    # Load spaCy model
    try:
        nlp = spacy.load(SPACY_MODEL, disable=["parser", "textcat"])
    except OSError:
        logger.error(f"SpaCy model {SPACY_MODEL} not found. Please install it.")
        return []

    # Process text in chunks for better performance
    for chunk in text.split("\n\n"):
        if not chunk.strip():
            continue

        doc = nlp(chunk)

        # SpaCy NER extraction
        for ent in doc.ents:
            for entity_type, info in entity_schema.items():
                if ent.label_ in info.get("spacy_labels", []):
                    entities.append({
                        "text": ent.text,
                        "type": entity_type,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })

        # Keyword fallback from entity schema
        for entity_type, info in entity_schema.items():
            keywords = info.get("keywords", [])
            for keyword in keywords:
                start = 0
                while True:
                    index = chunk.lower().find(keyword.lower(), start)
                    if index == -1:
                        break
                    end = index + len(keyword)
                    entities.append({
                        "text": chunk[index:end],
                        "type": entity_type,
                        "start": index,
                        "end": end
                    })
                    start = end

    logger.debug(f"Extracted {len(entities)} entities")
    return entities


def process_all_files():
    """Process all text files in CLEAN_DIR for entity extraction"""
    entity_schema = load_schema("entity_types.json")
    if not entity_schema:
        logger.error("Entity schema not found!")
        return

    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
    clean_dir = Path(CLEAN_DIR)

    text_files = list(clean_dir.glob("*.txt"))
    if not text_files:
        logger.warning(f"No text files found in {CLEAN_DIR}")
        return

    logger.info(f"Found {len(text_files)} text files for entity extraction")

    for file_path in tqdm(text_files, desc="Processing files"):
        logger.info(f"Processing file: {file_path.name}")
        start_time = time.time()

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            entities = extract_entities(text, entity_schema)

            output_path = ENTITIES_DIR / f"{file_path.stem}_entities.json"
            with open(output_path, "w", encoding="utf-8") as out:
                json.dump(entities, out, ensure_ascii=False, indent=2)

            elapsed = time.time() - start_time
            logger.info(f"Extracted {len(entities)} entities from {file_path.name} in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")


if __name__ == "__main__":
    print(">>> Entity extractor started...")
    process_all_files()