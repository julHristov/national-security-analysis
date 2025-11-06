# utils/normalizer.py
import json
from pathlib import Path
from config import ENTITY_NORMALIZATION_FILE
from typing import Dict, Optional

try:
    # rapidfuzz е по-бърз и поддържан от много проекти (ако не е инсталиран, fallback към simple match)
    from rapidfuzz import process, fuzz

    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False


def load_normalization_map(path: Path = ENTITY_NORMALIZATION_FILE) -> Dict[str, str]:
    """
    Зарежда нормализационния mapping от зададения път (по подразбиране от config.py)
    """
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k.strip().lower(): v.strip().lower() for k, v in data.items()}


def save_normalization_map(mapping: Dict[str, str], path: Path = ENTITY_NORMALIZATION_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    # save with canonical form as provided
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def normalize_entity(name: str, mapping: Dict[str, str], fuzzy_threshold: float = 90.0) -> str:
    """
    Нормализира единичен ентитет:
    1) exact lookup по lowercase ключовете в mapping
    2) ако rapidfuzz е наличен -> fuzzy match с threshold (по подразбиране 90)
    3) ако няма попадение -> връща оригинала, нормализиран (lowercase stripped)
    """
    if not name:
        return name

    key = name.strip().lower()
    if key in mapping:
        return mapping[key]

    # try simple prefix/suffix strip of common articles (опционално)
    # но за сега връщаме оригинала, при нужда можеш да добавиш правила тук.

    # fuzzy fallback
    if _HAS_RAPIDFUZZ and mapping:
        # търсим най-доброто съвпадение сред ключовете
        choices = list(mapping.keys())
        match, score, _ = process.extractOne(key, choices, scorer=fuzz.WRatio)  # по-robust scorer
        if score >= fuzzy_threshold:
            return mapping[match]

    # default: return lowercased input (може да се further-normalize)
    return key


def normalize_entities_list(entities, mapping: Dict[str, str], fuzzy_threshold: float = 90.0):
    """
    Поддържа два входни формата:
      1) entities: Dict[str, int]  (variant -> count)
      2) entities: Dict[str, Dict] (variant -> {"count":int, "spacy_label":..., ...})

    Връща агрегирани резултати в същия „богат“ формат (variant -> {count, spacy_label, ...}).
    За случая на консолидиране на метаданни (label-и) ще запазим тези от записа с най-голям count.
    """
    result = {}

    for ent, value in entities.items():
        # determine canonical name
        canon = normalize_entity(ent, mapping, fuzzy_threshold=fuzzy_threshold)

        # unpack incoming value
        if isinstance(value, dict):
            cnt = int(value.get("count", 0))
            spacy_label = value.get("spacy_label", "")
            custom_label = value.get("custom_label", "")
            schema_type = value.get("schema_type", "")
            # keep other metadata if present
            extra = {k: v for k, v in value.items() if k not in {"count", "spacy_label", "custom_label", "schema_type"}}
        else:
            # assume integer count
            try:
                cnt = int(value)
            except Exception:
                cnt = 0
            spacy_label = ""
            custom_label = ""
            schema_type = ""
            extra = {}

        if canon not in result:
            # initialize
            entry = {
                "count": cnt,
                "spacy_label": spacy_label,
                "custom_label": custom_label,
                "schema_type": schema_type
            }
            entry.update(extra)
            result[canon] = entry
        else:
            # aggregate counts
            existing = result[canon]
            existing_count = int(existing.get("count", 0))
            new_count = existing_count + cnt
            # choose labels from the side with larger contribution
            # if incoming cnt > existing_count, prefer incoming labels
            if cnt > existing_count:
                sp = spacy_label or existing.get("spacy_label", "")
                cu = custom_label or existing.get("custom_label", "")
                sc = schema_type or existing.get("schema_type", "")
            else:
                sp = existing.get("spacy_label", "")
                cu = existing.get("custom_label", "")
                sc = existing.get("schema_type", "")

            existing.update({
                "count": new_count,
                "spacy_label": sp,
                "custom_label": cu,
                "schema_type": sc
            })
            # merge extra keys naively (keep existing unless missing)
            for k, v in extra.items():
                if k not in existing or not existing[k]:
                    existing[k] = v

    return result


def build_mapping_from_pairs(pairs: Dict[str, str], path: Optional[Path] = None):
    """
    Удобна функция за запис на ръчен map. pairs трябва да е {variant: canonical}
    """
    m = {k.strip().lower(): v.strip().lower() for k, v in pairs.items()}
    save_normalization_map(m, path or ENTITY_NORMALIZATION_FILE)
    return m


def suggest_mappings_from_corpus(unique_entities: list, top_n: int = 100):
    """
    Помощна: дава candidate pairs за човешка проверка, базирани на частично съвпадение.
    Връща list of (entity, suggested_canonical, score) — само ако rapidfuzz е инсталиран.
    """
    if not _HAS_RAPIDFUZZ:
        return []

    candidates = []
    # сравнението е quadratic — в малки корпуси е ок
    from rapidfuzz import process, fuzz
    for ent in unique_entities:
        match, score, _ = process.extractOne(ent.lower(), unique_entities, scorer=fuzz.WRatio)
        if match and match != ent and score > 80:
            candidates.append((ent, match, score))
    # сортираме по score desc
    candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
    return candidates
