# main.py
import scripts.clean_texts as clean_texts
import sys
import json
from pathlib import Path
from modules_extractors.entity_extractor import process_all_files as extract_entities


def main():
    """Пълен процес: почистване -> ентитети -> отношения -> сценарии -> статистика"""
    print("🎯 Starting Complete National Security Analysis Pipeline")
    print("=" * 50)

    try:
        # Стъпка 1: Почистване на текстове
        print("📝 Step 1: Cleaning texts...")
        clean_texts.process_files()

        # Стъпка 2: Извличане на ентитети
        print("🏷️ Step 2: Extracting entities...")
        from modules_extractors.entity_extractor import process_all_files as extract_entities
        extract_entities()

        # Стъпка 3: Статистика за ентитетите
        print("📊 Step 3: Analyzing entity frequencies...")
        from modules_extractors.entity_frequency_extractor import EntityFrequencyExtractor
        freq_extractor = EntityFrequencyExtractor()
        freq_extractor.process_entity_files()
        freq_extractor.save_results()

        # Стъпка 4: Извличане на отношения (FIXED)
        print("🔗 Step 4: Extracting relations...")
        try:
            # Пробвай с правилното име
            from modules_extractors.relations_extractor import process_all_files as extract_relations
            extract_relations()
        except ImportError:
            # Ако пак не работи, пусни директно
            print("  Using direct execution...")
            import subprocess
            subprocess.run([sys.executable, "-m", "modules_extractors.relations_extractor"])

        # Стъпка 5: Извличане на сценарии
        print("🎭 Step 5: Extracting scenarios...")
        extract_scenarios()

        print("✅ Analysis Complete! Check the 'data/annotated' folder for results.")

    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()


def extract_scenarios():
    """Извличане на сценарии за всички документи"""
    from modules_extractors.scenario_extractor import ScenarioExtractor
    from config import CLEAN_DIR

    print("Starting scenario extraction for all documents...")
    try:
        extractor = ScenarioExtractor()
        clean_dir = Path(CLEAN_DIR)

        for doc_path in clean_dir.glob("*.txt"):
            print(f"Processing: {doc_path.name}")

            with open(doc_path, 'r', encoding='utf-8') as f:
                text = f.read()

            scenarios = extractor.extract_scenarios(text)
            sentiment = extractor.analyze_context_sentiment(text)

            output = {
                "file": doc_path.name,
                "total_scenarios": len(scenarios),
                "context_sentiment": sentiment,
                "scenarios": scenarios
            }

            output_file = Path("data/annotated/scenarios") / f"{doc_path.stem}_scenarios.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            print(f"  Extracted {len(scenarios)} scenarios -> {output_file}")

    except Exception as e:
        print(f"Error in scenario extraction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "extract_scenarios":
        extract_scenarios()
    else:
        main()