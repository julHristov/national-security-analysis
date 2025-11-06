import json
from pathlib import Path
from collections import Counter, defaultdict
from schema.schema_loader import load_entity_schema


# Import from config - handle missing values
try:
    from config import *

    # Fallback if TOP_ENTITIES_COUNT is not defined
    if 'TOP_ENTITIES_COUNT' not in globals():
        TOP_ENTITIES_COUNT = 10
except ImportError:
    # Fallback values if config is missing
    ENTITIES_DIR = Path("data/annotated/entities")
    ENTITY_FREQ_DIR = Path("data/annotated/entity_frequency")
    TOP_ENTITIES_FILE = ENTITY_FREQ_DIR / "top_entities.json"
    ENTITY_PERCENTAGES_FILE = ENTITY_FREQ_DIR / "entity_percentages.json"
    TOP_ENTITIES_COUNT = 10


def get_logger(name):
    class SimpleLogger:
        def info(self, msg): print(f"INFO: {msg}")

        def warning(self, msg): print(f"WARNING: {msg}")

        def error(self, msg): print(f"ERROR: {msg}")

    return SimpleLogger()


logger = get_logger(__name__)


class EntityFrequencyExtractor:
    def __init__(self):
        self.entity_counter = Counter()
        self.entity_types = defaultdict(Counter)

    def process_entity_files(self):
        """Process all entity JSON files and count frequencies"""
        entity_files = list(ENTITIES_DIR.glob("*_entities.json"))

        if not entity_files:
            logger.warning(f"No entity files found in {ENTITIES_DIR}")
            return

        logger.info(f"Processing {len(entity_files)} entity files")

        for entity_file in entity_files:
            try:
                with open(entity_file, 'r', encoding='utf-8') as f:
                    entities = json.load(f)

                for entity in entities:
                    entity_text = entity.get('text', '').strip()
                    entity_type = entity.get('type', 'UNKNOWN')

                    if entity_text:
                        self.entity_counter[entity_text] += 1
                        self.entity_types[entity_type][entity_text] += 1

            except Exception as e:
                logger.error(f"Error processing {entity_file}: {str(e)}")

    def get_top_entities(self, top_n=10):
        """Get top N entities across all documents"""
        return self.entity_counter.most_common(top_n)

    def get_entity_percentages(self):
        """Calculate percentage distribution of entity types"""
        total_entities = sum(self.entity_counter.values())

        if total_entities == 0:
            return {}

        percentages = {}
        for entity_type, type_counter in self.entity_types.items():
            type_total = sum(type_counter.values())
            percentages[entity_type] = (type_total / total_entities) * 100

        return percentages

    def save_results(self):
        """Save frequency analysis results"""
        ENTITY_FREQ_DIR.mkdir(parents=True, exist_ok=True)

        # Save top entities - use fallback if not defined
        try:
            top_n = TOP_ENTITIES_COUNT
        except NameError:
            top_n = 10

        top_entities = self.get_top_entities(top_n)
        with open(TOP_ENTITIES_FILE, 'w', encoding='utf-8') as f:
            json.dump(dict(top_entities), f, indent=2, ensure_ascii=False)

        # Save entity percentages
        percentages = self.get_entity_percentages()
        with open(ENTITY_PERCENTAGES_FILE, 'w', encoding='utf-8') as f:
            json.dump(percentages, f, indent=2, ensure_ascii=False)

        # Save detailed entity type breakdown
        type_breakdown = {
            entity_type: dict(counter.most_common(10))
            for entity_type, counter in self.entity_types.items()
        }

        breakdown_file = ENTITY_FREQ_DIR / "entity_type_breakdown.json"
        with open(breakdown_file, 'w', encoding='utf-8') as f:
            json.dump(type_breakdown, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved results to {ENTITY_FREQ_DIR}")

        # Print summary
        print(f"\n=== TOP {top_n} ENTITIES ===")
        for entity, count in top_entities:
            print(f"{entity}: {count}")

        print(f"\n=== ENTITY TYPE DISTRIBUTION ===")
        for entity_type, percentage in percentages.items():
            print(f"{entity_type}: {percentage:.1f}%")


def main():
    extractor = EntityFrequencyExtractor()
    extractor.process_entity_files()
    extractor.save_results()

    print(f"\nAnalysis complete. Results saved to {ENTITY_FREQ_DIR}")


if __name__ == "__main__":
    main()