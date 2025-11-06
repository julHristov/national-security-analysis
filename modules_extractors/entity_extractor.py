import spacy
import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import time

from modules_extractors.schema_loader import load_schema
from config import CLEAN_DIR, ENTITIES_DIR, SPACY_MODE, SPACY_MODEL
from utils.logger import get_logger
from utils.file_manager import read_text_file, write_json_file

logger = get_logger(__name__)

# ===============================================================
#  🔹 Инициализация на spaCy модела (динамично според конфигурацията)
# ===============================================================
logger.info(f"Selected SPACY_MODE: {SPACY_MODE}")
logger.info(f"Loading spaCy model: {SPACY_MODEL}")
nlp = spacy.load(SPACY_MODEL, disable=["parser", "textcat"])


# ===============================================================
#  🔹 Custom labeling rules (STATE / ORG / INSTITUTION / OTHER)
# ===============================================================
def determine_custom_label(ent_text: str, spacy_label: str) -> str:
    text = ent_text.lower()

    if any(k in text for k in ["republic of", "bulgaria", "romania", "turkey", "serbia", "russia", "ukraine"]):
        return "STATE"
    elif any(k in text for k in ["ministry", "agency", "council", "organization", "union", "committee", "government"]):
        return "ORG"
    elif any(k in text for k in ["parliament", "president", "minister", "defense", "army", "forces"]):
        return "INSTITUTION"
    elif spacy_label in ["GPE", "LOC"]:
        return "STATE"
    elif spacy_label in ["ORG"]:
        return "ORG"
    else:
        return "OTHER"


# ===============================================================
#  🔹 Entity extraction logic
# ===============================================================
def extract_entities(text: str, entity_schema: dict) -> list:
    """
    Извлича ентитети чрез spaCy + rule-based слой + автоматична верификация.
    """
    if not text:
        logger.warning("⚠️ Empty text provided for entity extraction.")
        return []

    entities = []
    logger.debug("Starting entity extraction...")

    # Зареждаме нормализационния mapping
    from utils.normalizer import load_normalization_map, normalize_entity
    normalization_map = load_normalization_map()

    # Split text by paragraphs for efficiency
    for chunk in text.split("\n\n"):
        if not chunk.strip():
            continue

        doc = nlp(chunk)
        for ent in doc.ents:
            for entity_type, info in entity_schema.items():
                if ent.label_ in info.get("spacy_labels", []):
                    custom_label = determine_custom_label(ent.text, ent.label_)
                    verified = ent.text.lower() in text.lower()
                    normalized = normalize_entity(ent.text.strip(), normalization_map)

                    entities.append({
                        "entity": ent.text.strip(),
                        "normalized_entity": normalized,
                        "spacy_label": ent.label_,
                        "custom_label": custom_label,
                        "schema_type": entity_type,
                        "verified": verified,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })

    # === Keyword fallback от schema ===
    for entity_type, info in entity_schema.items():
        keywords = info.get("examples", []) + info.get("context_keywords", [])
        for keyword in keywords:
            start = 0
            while True:
                index = text.lower().find(keyword.lower(), start)
                if index == -1:
                    break
                end = index + len(keyword)
                verified = True
                normalized = normalize_entity(text[index:end], normalization_map)
                entities.append({
                    "entity": text[index:end],
                    "normalized_entity": normalized,
                    "spacy_label": "MANUAL",
                    "custom_label": determine_custom_label(keyword, "MANUAL"),
                    "schema_type": entity_type,
                    "verified": verified,
                    "start": index,
                    "end": end
                })
                start = end

    return entities


# ===============================================================
#  🔹 Main orchestration
# ===============================================================
def process_all_files():
    entity_schema = load_schema("entity_types.json")
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)

    text_files = list(CLEAN_DIR.glob("*.txt"))
    if not text_files:
        logger.warning(f"No text files found in {CLEAN_DIR}")
        return

    logger.info(f"Found {len(text_files)} text files for entity extraction")

    for file_path in tqdm(text_files, desc="Processing files"):
        logger.info(f"Processing file: {file_path.name}")
        start_time = time.time()

        text = read_text_file(file_path)
        entities = extract_entities(text, entity_schema)

        # === Подреждане и броене по честота ===
        counter = Counter([e["entity"].lower() for e in entities])
        for e in entities:
            e["count"] = counter[e["entity"].lower()]

        entities_sorted = sorted(entities, key=lambda x: x["count"], reverse=True)

        # === Запис като JSON ===
        output_path = ENTITIES_DIR / f"{file_path.stem}_entities.json"
        write_json_file(entities_sorted, output_path)

        elapsed = time.time() - start_time
        logger.info(f"✅ Extracted {len(entities_sorted)} entities from {file_path.name} ({elapsed:.2f}s)")


if __name__ == "__main__":
    print(">>> Entity extractor started...")
    process_all_files()
