from pathlib import Path

# =============================
# Main structure of the project
# =============================

# Finds the path of root of the project (annotation_project)
BASE_DIR = Path(__file__).resolve().parent  # Project main folder

# Folders
DATA_DIR = BASE_DIR / "data"  # All data folder
CLEAN_DIR = DATA_DIR / "clean_texts"  # Clean texts folder
ANNOTATED_DIR = DATA_DIR / "annotated"
RESULTS_DIR = DATA_DIR / "results"
RAW_DIR = DATA_DIR / "raw_texts"  # Raw texts folder
LOOKUP_DIR = DATA_DIR / "lookup"

# Directories under annotated/
ENTITIES_DIR = ANNOTATED_DIR / "entities"
RELATIONS_DIR = ANNOTATED_DIR / "relations"
SCENARIOS_DIR = ANNOTATED_DIR / "scenarios"

# ==========================
# Schemas and configurations
# ==========================
SCHEMA_DIR = BASE_DIR / "schema"  # JSON files folder (entity_types.json, relation_types.json)
REPLACEMENTS_FILE = SCHEMA_DIR / "replacements.json"  # JSON file with replacement symbols
MODULES_DIR = BASE_DIR / "modules_extractors"
UTILS_DIR = BASE_DIR / "utils"
ENTITY_SCHEMA_DIR = SCHEMA_DIR / "entity_types.json"
RELATION_SCHEMA_PATH = SCHEMA_DIR / "relation_types.json"

# ============================
# Output statistics file paths
# ============================
ENTITY_FREQ_DIR = ANNOTATED_DIR / "entity_frequency"
TOP_ENTITIES_FILE = ENTITY_FREQ_DIR / "top_entities.json"
ENTITY_PERCENTAGES_FILE = ENTITY_FREQ_DIR / "entity_percentages.json"

# =================================
# Scripts and helper directories
# =================================
SCRIPTS_DIR = BASE_DIR / "scripts"  # Scripts folder
GUIDELINES_DIR = BASE_DIR / "guidelines"  # To be added later

# =================
# Normalization map
# =================
ENTITY_NORMALIZATION_FILE = LOOKUP_DIR / "entity_normalization.json"

# =================================
# Automatic creation of directories
# =================================
for directory in [DATA_DIR, RAW_DIR, CLEAN_DIR, ANNOTATED_DIR, ENTITIES_DIR, RELATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# === Mode Settings ===
# Choose between: "fast" (en_core_web_md) or "accurate" (en_core_web_trf)
SPACY_MODE = 'fast'
# Automatically pick model name based on mode

SPACY_MODELS = {
    "fast": "en_core_web_md",
    "accurate": "en_core_web_trf"
}

SPACY_MODEL = SPACY_MODELS.get(SPACY_MODE, "en_core_web_md")

# === Log settings ===
LOG_FILE = BASE_DIR / "logs" / "project.log"
LOG_LEVEL = "INFO"


if __name__ == "__main__":
    # Check
    print("BASE_DIR:", BASE_DIR)
    print("RAW_DIR:", RAW_DIR)
    print("CLEAN_DIR:", CLEAN_DIR)
