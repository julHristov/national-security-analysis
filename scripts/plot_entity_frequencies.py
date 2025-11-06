import json
import matplotlib.pyplot as plt
from pathlib import Path
from config import RESULTS_DIR


def plot_top_entities():
    """
    Визуализира top_entities.json — топ ентитети по документи.
    Създава PNG графики в results/plots/.
    """
    results_path = Path(RESULTS_DIR)
    top_entities_path = results_path / "top_entities.json"
    plots_dir = results_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not top_entities_path.exists():
        print("❌ Не е намерен results/top_entities.json. Стартирай entity_frequency_extractor.py първо.")
        return

    with open(top_entities_path, "r", encoding="utf-8") as f:
        top_entities = json.load(f)

    for doc_name, entities in top_entities.items():
        if not entities:
            continue

        # Подреждаме по честота
        sorted_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)
        labels = [e[0] for e in sorted_entities]
        values = [e[1] for e in sorted_entities]

        plt.figure(figsize=(10, 6))
        plt.barh(labels, values, color="steelblue")
        plt.title(f"Top Entities in {doc_name}", fontsize=14, pad=10)
        plt.xlabel("Frequency (absolute count)", fontsize=12)
        plt.ylabel("Entity", fontsize=12)
        plt.gca().invert_yaxis()  # най-честите горе
        plt.tight_layout()

        output_file = plots_dir / f"{doc_name}_top_entities.png"
        plt.savefig(output_file, dpi=200)
        plt.close()

        print(f"✅ Графика записана: {output_file}")

    print("📊 Всички графики са успешно генерирани!")


if __name__ == "__main__":
    plot_top_entities()
