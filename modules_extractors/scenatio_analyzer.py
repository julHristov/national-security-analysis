# modules_extractors/scenario_analyzer.py

import json
from pathlib import Path
from config import ENTITIES_DIR, RELATIONS_DIR, SCENARIOS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

# =========================
# Scenario extraction logic
# =========================

def extract_scenarios(entities, relations):
    """
    Extract scenarios using entities and relations.
    For each relation, find closest ACTOR/ORG entity before (subject) and after (object).
    """
    scenarios = []

    # Sort entities by start offset
    entities_sorted = sorted(entities, key=lambda x: x["start"])

    for rel in relations:
        rel_start = rel.get("start")
        rel_end = rel.get("end")
        action = rel.get("relation_type", "UNKNOWN")

        # Find nearest entity before relation start (subject)
        subj_candidates = [e for e in entities_sorted if e["end"] <= rel_start and e["type"] in ["ACTOR", "ORG"]]
        subj = subj_candidates[-1]["text"] if subj_candidates else None

        # Find nearest entity after relation end (object)
        obj_candidates = [e for e in entities_sorted if e["start"] >= rel_end and e["type"] in ["ACTOR", "ORG", "CONCEPT", "VALUE"]]
        obj = obj_candidates[0]["text"] if obj_candidates else None

        if subj and obj:
            scenarios.append({
                "pattern": "ACTOR-ACT-ACTOR" if obj_candidates[0]["type"] in ["ACTOR", "ORG"] else "ACTOR-ACT",
                "subject": subj,
                "action": action,
                "object": obj
            })

    return scenarios


def process_scenario_files():
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    all_scenarios = {}

    text_files = list(ENTITIES_DIR.glob("*_entities.json"))
    if not text_files:
        logger.warning("⚠️ No entity files found. Run entity_extractor.py first.")
        return

    for ent_file in text_files:
        base_name = ent_file.stem.replace("_entities", "")
        rel_file = RELATIONS_DIR / f"{base_name}_relations.json"
        scenario_file = SCENARIOS_DIR / f"{base_name}_scenarios.json"

        if not rel_file.exists():
            logger.warning(f"⚠️ No relations file for {base_name}. Skipping...")
            continue

        with open(ent_file, "r", encoding="utf-8") as f:
            entities = json.load(f)
        with open(rel_file, "r", encoding="utf-8") as f:
            relations = json.load(f)

        scenarios = extract_scenarios(entities, relations)
        all_scenarios[base_name] = scenarios

        with open(scenario_file, "w", encoding="utf-8") as out:
            json.dump(scenarios, out, indent=2, ensure_ascii=False)

        logger.info(f"✅ Extracted {len(scenarios)} scenarios from {base_name}")

    # ==============================
    # Build scenario persistence map
    # ==============================
    doc_names = list(all_scenarios.keys())
    scenario_presence = {}

    for doc, scen_list in all_scenarios.items():
        for scen in scen_list:
            key = f"{scen['subject']}::{scen['action']}::{scen['object']}"
            if key not in scenario_presence:
                scenario_presence[key] = []
            scenario_presence[key].append(doc)

    summary["persistent"] = [k for k, docs in scenario_presence.items() if len(docs) == len(doc_names)]
    summary["emerging"] = [k for k, docs in scenario_presence.items() if len(docs) < len(doc_names) and doc_names[-1] in docs]
    summary["disappearing"] = [k for k, docs in scenario_presence.items() if doc_names[0] in docs and doc_names[-1] not in docs]

    summary_file = SCENARIOS_DIR / "scenario_summary_all.json"
    with open(summary_file, "w", encoding="utf-8") as s:
        json.dump(summary, s, indent=2, ensure_ascii=False)

    logger.info(f"📊 Summary saved to {summary_file}")
    logger.info(f"Persistent: {len(summary['persistent'])}, Emerging: {len(summary['emerging'])}, Disappearing: {len(summary['disappearing'])}")


if __name__ == "__main__":
    logger.info(">>> Scenario Analyzer started...")
    process_scenario_files()
