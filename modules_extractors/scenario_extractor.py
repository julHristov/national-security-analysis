import spacy
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from config import CLEAN_DIR, RESULTS_DIR, SPACY_MODEL
from utils.file_manager import read_text_file, write_json_file
from utils.normalizer import load_normalization_map, normalize_entity
from modules_extractors.context_analyzer import ContextAnalyzer
from modules_extractors.scenario_weight_calculator import ScenarioWeightCalculator





class TermAnalyzer:
    """Клас за анализ на термини и техните тегла."""
    
    def __init__(self, schema_dir: str = None):
        """Инициализира анализатора с речници."""
        if not schema_dir:
            schema_dir = Path(__file__).parent.parent / "schema"
            
        # Зареждане на речниците с термини
        with open(schema_dir / "positive_terms.json", "r", encoding="utf-8") as f:
            self.positive_terms = set(json.load(f))
            
        with open(schema_dir / "negative_terms.json", "r", encoding="utf-8") as f:
            self.negative_terms = set(json.load(f))
            
        with open(schema_dir / "neutral_terms.json", "r", encoding="utf-8") as f:
            self.neutral_terms = set(json.load(f))
            
        with open(schema_dir / "context_dictionaries.json", "r", encoding="utf-8") as f:
            self.context_dictionaries = json.load(f)
            
        # Създаваме речник за бързо търсене на контексти
        self._security_contexts = {
            term.lower(): context
            for context, terms in self.context_dictionaries["security_contexts"].items()
            for term in terms
        }
            
        # Речници за съхранение на честотите и теглата
        self.term_frequencies = defaultdict(lambda: defaultdict(int))
        self.term_weights = defaultdict(dict)

        

    def analyze_term_frequencies(self, text: str):
        """Анализира честотата на термините в текста."""
        text = text.lower()
        
        # Проверяваме срещания за всички термини
        for term_type, terms in [
            ("positive", self.positive_terms),
            ("negative", self.negative_terms),
            ("neutral", self.neutral_terms)
        ]:
            for term in terms:
                count = text.count(term)
                if count > 0:
                    self.term_frequencies[term_type][term] += count
    
    def calculate_weights(self):
        """Изчислява тегла за термините според честотата им."""
        for term_type in ["positive", "negative", "neutral"]:
            frequencies = self.term_frequencies[term_type]
            if not frequencies:
                continue
                
            max_freq = max(frequencies.values())
            self.term_weights[term_type] = {
                term: 0.1 + 0.9 * (freq / max_freq)
                for term, freq in frequencies.items()
            }

    def analyze_text_sentiment(self, text: str) -> Dict[str, dict]:
        """Анализира емоционалната окраска на текст."""
        text = text.lower()
        sentiment_scores = {
            "positive": {"terms": [], "total_weight": 0.0},
            "negative": {"terms": [], "total_weight": 0.0},
            "neutral": {"terms": [], "total_weight": 0.0}
        }
        
        # Проверяваме за всички възможни термини
        for term_type, terms in [
            ("positive", self.positive_terms),
            ("negative", self.negative_terms),
            ("neutral", self.neutral_terms)
        ]:
            for term in terms:
                if term in text:
                    weight = self.term_weights[term_type].get(term, 0.1)
                    sentiment_scores[term_type]["terms"].append({
                        "term": term,
                        "weight": weight
                    })
                    sentiment_scores[term_type]["total_weight"] += weight
                    
        return sentiment_scores
    
    def analyze_text_context(self, text: str) -> Dict[str, int]:
        """Анализира тематичния контекст на текст."""
        text = text.lower()
        contexts = defaultdict(int)
        
        for term, context in self._security_contexts.items():
            if term in text:
                contexts[context] += 1
                
        return dict(contexts)


class ScenarioExtractor:
    """Основен клас за извличане и анализ на сценарии."""
    
    def __init__(self):
        """Инициализира екстрактора."""
        self.term_analyzer = TermAnalyzer()
        self.weight_calculator = ScenarioWeightCalculator()
        self.nlp = spacy.load(SPACY_MODEL)
        
    def calculate_term_weights(self):
        """Изчислява теглата на термините от всички документи."""
        print("📊 Извличане на честоти на термините...")
        clean_dir = Path(CLEAN_DIR)
        
        for file in clean_dir.glob("*.txt"):
            print(f"📄 Обработка на {file.name}...")
            text = read_text_file(file)
            self.term_analyzer.analyze_term_frequencies(text)
            
        print("⚖️ Изчисляване на тегла...")
        self.term_analyzer.calculate_weights()
        
        # Записваме теглата за референция
        weights_data = {
            "frequencies": {k: dict(v) for k, v in self.term_analyzer.term_frequencies.items()},
            "weights": dict(self.term_analyzer.term_weights)
        }
        weights_path = Path(RESULTS_DIR) / "term_weights.json"
        write_json_file(weights_data, weights_path)
    
    def analyze_scenario_context(self, scenario: dict) -> dict:
        """Анализира контекста на сценарий."""
        # Анализ на цялото изречение
        sent_context = self.term_analyzer.analyze_text_context(scenario["sentence"])
        sent_sentiment = self.term_analyzer.analyze_text_sentiment(scenario["sentence"])
        
        # Анализ само на фразата с действието
        action_context = self.term_analyzer.analyze_text_context(scenario["action_phrase"])
        action_sentiment = self.term_analyzer.analyze_text_sentiment(scenario["action_phrase"])
        
        # Определяме доминиращия контекст и сантимент
        dominant_context = max(sent_context.items(), key=lambda x: x[1])[0] if sent_context else "undefined"
        dominant_sentiment = max(
            ["positive", "negative", "neutral"],
            key=lambda x: sent_sentiment[x]["total_weight"]
        )
        
        return {
            **scenario,
            "context_analysis": {
                "sentence": {
                    "context": sent_context,
                    "sentiment": sent_sentiment
                },
                "action": {
                    "context": action_context,
                    "sentiment": action_sentiment
                },
                "dominant_context": dominant_context,
                "dominant_sentiment": {
                    "type": dominant_sentiment,
                    "weight": sent_sentiment[dominant_sentiment]["total_weight"]
                }
            }
        }
    
    def extract_scenarios_for_actor(self, doc, actor: str) -> List[dict]:
        """Извлича сценарии за даден актьор от документ."""
        scenarios = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if actor.lower() not in sent_text.lower():
                continue

            for token in sent:
                if token.dep_ in ("nsubj", "nsubjpass") and actor.lower() in token.text.lower():
                    verb = token.head.lemma_
                    action_phrase = " ".join([t.text for t in token.head.subtree])
                    targets = [child.text for child in token.head.children if child.dep_ in ("dobj", "pobj", "attr")]
                    if not targets:
                        targets = [t.text for t in token.head.subtree if t.dep_ in ("dobj", "pobj")]

                    # Създаваме базовия сценарий
                    scenario = {
                        "actor_1": actor,
                        "action": verb,
                        "action_phrase": action_phrase,
                        "targets": targets,
                        "sentence": sent_text
                    }
                    
                    # Добавяме анализ на контекста
                    scenario = self.analyze_scenario_context(scenario)
                    scenarios.append(scenario)

        return scenarios
    
    def process_document(self, doc_path: Path, actor_name: str = None, top_n: int = 5):
        """Обработва документ и извлича сценарии."""
        print(f"📄 Обработка на {doc_path.name} ...")
        text = read_text_file(doc_path)
        doc = self.nlp(text)
        
        # Зареждаме нужните ресурси
        mapping = load_normalization_map()
        entities = load_entities_for_doc(doc_path.stem)
        actors = choose_actors(entities, mapping, actor_name, top_n)
        
        # Извличаме сценарии за всеки актьор
        all_scenarios = defaultdict(list)
        for actor in tqdm(actors, desc=f"🔍 Извличане на сценарии ({doc_path.stem})"):
            scenarios = self.extract_scenarios_for_actor(doc, actor)
            all_scenarios[actor].extend(scenarios)
            
        # Записваме резултатите
        out_dir = Path(RESULTS_DIR) / "scenarios"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{doc_path.stem}_scenarios.json"
        # Обогатяваме сценариите с тежести
        enriched_scenarios = self.weight_calculator.enrich_scenarios(all_scenarios)
        write_json_file(dict(enriched_scenarios), out_path)
        print(f"✅ Сценариите са записани в {out_path}")


def load_entities_for_doc(doc_stem: str):
    """Зарежда вече извлечените и нормализирани ентитети за документа."""
    entities_path = Path(RESULTS_DIR) / "top_entities.json"
    if not entities_path.exists():
        raise FileNotFoundError(f"⚠️ Не е открит {entities_path}")
    with open(entities_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(doc_stem, {})


def choose_actors(doc_entities: dict, mapping: dict, actor_name: str = None, top_n: int = 5):
    """Връща списък с актьори – ако е подаден actor_name, връща само него; иначе топ N от документа."""
    if actor_name:
        return [normalize_entity(actor_name.lower(), mapping)]
    else:
        sorted_entities = sorted(doc_entities.items(), key=lambda x: x[1], reverse=True)
        top_actors = [normalize_entity(a.lower(), mapping) for a, _ in sorted_entities[:top_n]]
        return top_actors


def main():
    parser = argparse.ArgumentParser(description="Екстрактор на сценарии с контекстуален анализ")
    parser.add_argument("--actor", help="Избери актьор (ако не е зададен, се ползват топ 5)")
    parser.add_argument("--top-n", type=int, default=5, help="Брой топ актьори по подразбиране")
    args = parser.parse_args()
    
    extractor = ScenarioExtractor()
    
    # Първо изчисляваме теглата
    extractor.calculate_term_weights()
    
    # После обработваме документите
    clean_dir = Path(CLEAN_DIR)
    for file in clean_dir.glob("*.txt"):
        extractor.process_document(file, args.actor, args.top_n)


if __name__ == "__main__":
    main()