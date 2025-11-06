# main.py
import scripts.clean_texts as clean_texts
import sys
from modules_extractors.scenario_extractor import ScenarioExtractor
from config import CLEAN_DIR, RESULTS_DIR


def main():
    print("== Start Cleaning ==")
    try:
        clean_texts.process_files()
        print("== Cleaning Complete ==")
    except Exception as e:
        print(f"⚠️ Грешка при стартиране на процеса: {e}")


def extract_scenarios():
    print("== Start Scenario Extraction ==")
    try:
        extractor = ScenarioExtractor()
        # Извличаме сценарии за всички документи в CLEAN_DIR
        from pathlib import Path
        clean_dir = Path(CLEAN_DIR)
        for doc_path in clean_dir.glob("*.txt"):
            extractor.process_document(doc_path)
        print("== Scenario Extraction Complete ==")
    except Exception as e:
        print(f"⚠️ Грешка при извличане на сценарии: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "extract_scenarios":
        extract_scenarios()
    else:
        main()
