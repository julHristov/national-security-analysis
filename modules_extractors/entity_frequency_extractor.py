import json
from collections import Counter, defaultdict
import spacy
from pathlib import Path
from utils.file_manager import read_text_file, write_json_file
from schema.schema_loader import load_entity_schema
from config import CLEAN_DIR, ANNOTATED_DIR, RESULTS_DIR, SPACY_MODEL
from modules_extractors.entity_extractor import determine_custom_label

# === Ново: автоматично извикване на графичния модул ===
from scripts.plot_entity_frequencies import plot_top_entities


def extract_entities_from_text(text, nlp, entity_schema):
    """
    Извлича ентитети чрез spaCy и schema правила.
    Връща речник с ентитети и техните честоти + етикети.
    """
    doc = nlp(text)
    entity_info = defaultdict(lambda: {"count": 0, "spacy_label": "", "custom_label": "", "schema_type": ""})

    for ent in doc.ents:
        ent_label = ent.label_
        ent_text = ent.text.lower().strip()

        for category, info in entity_schema.items():
            if ent_label in info["spacy_labels"] or ent_text in info["context_keywords"]:
                custom_label = determine_custom_label(ent_text, ent_label)
                entity_info[ent_text]["count"] += 1
                entity_info[ent_text]["spacy_label"] = ent_label
                entity_info[ent_text]["custom_label"] = custom_label
                entity_info[ent_text]["schema_type"] = category
                break

    return dict(entity_info)


def calculate_relative_frequency(entity_counts, total_words):
    """
    Връща речник с относителна честота (%) за всеки ентитет.
    Поддържа и прост формат {entity: int}, и разширен {entity: {"count": int}}.
    """
    result = {}
    for entity, data in entity_counts.items():
        if isinstance(data, dict):
            count = data.get("count", 0)
        else:
            count = data
        rel_freq = round((count / total_words) * 100, 5)
        result[entity] = {
            "count": count,
            "relative_frequency": rel_freq
        }
    return result


def process_all_texts():
    print("🔍 Loading spaCy model...")
    nlp = spacy.load(SPACY_MODEL)

    entity_schema = load_entity_schema()
    CLEAN_PATH = Path(CLEAN_DIR)
    ANNOTATED_PATH = Path(ANNOTATED_DIR)
    RESULTS_PATH = Path(RESULTS_DIR)

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    ANNOTATED_PATH.mkdir(parents=True, exist_ok=True)

    top_entities_global = {}

    for text_file in CLEAN_PATH.glob("*.txt"):
        print(f"📄 Processing {text_file.name}...")
        text = read_text_file(text_file)
        total_words = len(text.split())

        # --- Извличане и изчисления ---
        from utils.normalizer import load_normalization_map, normalize_entities_list
        # 1. Зареждаме речника с подобни имена (например "EU" и "European Union")
        mapping = load_normalization_map()

        # 2. Извличаме ентитетите (имената, които spaCy е разпознал)
        entity_info = extract_entities_from_text(text, nlp, entity_schema)

        # 3. Обединяваме вариантите на едно и също име (пример: "Bulgaria" + "Republic of Bulgaria")
        entity_info = normalize_entities_list(entity_info, mapping)
        # Ако имаме речник с допълнителни данни (count, label и т.н.), взимаме само броя за изчисленията
        if isinstance(list(entity_info.values())[0], dict):
            simple_counts = {k: v.get("count", 0) for k, v in entity_info.items()}
        else:
            simple_counts = entity_info

        # 4. Изчисляваме процентното им отношение спрямо броя думи в текста
        entity_info = calculate_relative_frequency(simple_counts, total_words)

        # --- Запис за отделния документ ---
        entities_output = {
            "document": text_file.name,
            "total_words": total_words,
            "entities": entity_info
        }

        output_path = ANNOTATED_PATH / f"{text_file.stem}_entities.json"
        write_json_file(entities_output, output_path)

        # --- Топ 10 за резултатния файл ---
        sorted_entities = sorted(
            entity_info.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        top_10 = {k: v["count"] for k, v in sorted_entities[:10]}
        top_entities_global[text_file.stem] = top_10

    # --- Запис на глобалния резултат ---
    write_json_file(top_entities_global, RESULTS_PATH / "top_entities.json")
    print("✅ Extraction completed successfully.")

    # === Автоматично генериране на графики ===
    print("📊 Generating visualizations...")
    plot_top_entities()
    print("🎨 All charts generated successfully!")


if __name__ == "__main__":
    process_all_texts()
