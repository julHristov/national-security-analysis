# schema/schema_loader.py

import json
from pathlib import Path

SCHEMA_DIR = Path(__file__).resolve().parent


def load_entity_schema():
    """Load entity_types.json schema file."""
    schema_path = SCHEMA_DIR / "entity_types.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_relation_schema():
    """Load relation_types.json schema file (future use)."""
    schema_path = SCHEMA_DIR / "relation_types.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)
