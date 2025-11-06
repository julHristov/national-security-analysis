"""
Модул за изчисляване на тежест на сценарии базирано на честота,
контекст и емоционална окраска.
"""
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import spacy
from pathlib import Path


class ScenarioWeightCalculator:
    """Клас за изчисляване тежестта на сценариите."""
    
    def __init__(self):
        """Инициализира калкулатора."""
        self.scenario_frequencies = defaultdict(int)
        self.context_weights = defaultdict(float)
        self.sentiment_weights = defaultdict(float)
        
    def calculate_action_similarity(self, action1: str, action2: str) -> float:
        """
        Изчислява подобност между две действия.
        Засега простo съвпадение, но може да се разшири със синоними/WordNet.
        """
        return 1.0 if action1.lower() == action2.lower() else 0.0
        
    def calculate_context_weight(self, context_analysis: dict) -> float:
        """Изчислява тежест базирана на контекстуалния анализ."""
        # Взимаме предвид и двата контекста - на изречението и на действието
        sent_contexts = context_analysis["sentence"]["context"]
        action_contexts = context_analysis["action"]["context"]
        
        # Сумираме срещанията на контексти, давайки по-голяма тежест на
        # контекстите, които се срещат и в двете места
        total_weight = 0.0
        all_contexts = set(sent_contexts.keys()) | set(action_contexts.keys())
        
        for context in all_contexts:
            sent_count = sent_contexts.get(context, 0)
            action_count = action_contexts.get(context, 0)
            
            # Ако контекстът е и в двете места, увеличаваме тежестта
            if sent_count > 0 and action_count > 0:
                weight = (sent_count + action_count) * 1.5
            else:
                weight = sent_count + action_count
                
            total_weight += weight
            
        return total_weight
        
    def calculate_sentiment_weight(self, context_analysis: dict) -> float:
        """Изчислява тежест базирана на емоционалната окраска."""
        sent_sentiment = context_analysis["sentence"]["sentiment"]
        action_sentiment = context_analysis["action"]["sentiment"]
        
        # Взимаме средно-претеглена стойност от двата анализа
        sent_weight = sum(s["total_weight"] for s in sent_sentiment.values())
        action_weight = sum(s["total_weight"] for s in action_sentiment.values())
        
        # Даваме по-голяма тежест на емоционалната окраска в действието
        return (sent_weight + 2 * action_weight) / 3
        
    def calculate_scenario_weight(self, scenario: dict, doc_scenarios: List[dict]) -> float:
        """
        Изчислява обща тежест на сценарий, взимайки предвид:
        1. Честота на подобни сценарии в документа
        2. Тежест на контекста
        3. Тежест на емоционалната окраска
        """
        # Намираме подобни сценарии
        similar_count = sum(
            1 for s in doc_scenarios
            if self.calculate_action_similarity(s["action"], scenario["action"]) > 0.8
        )
        
        # Изчисляваме различните компоненти
        freq_weight = 0.1 + 0.9 * (similar_count / len(doc_scenarios))
        context_weight = self.calculate_context_weight(scenario["context_analysis"])
        sentiment_weight = self.calculate_sentiment_weight(scenario["context_analysis"])
        
        # Комбинираме теглата (може да се настроят коефициентите)
        total_weight = (
            0.4 * freq_weight +
            0.4 * context_weight +
            0.2 * sentiment_weight
        )
        
        return total_weight
        
    def enrich_scenarios(self, scenarios: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
        """
        Обогатява сценариите с изчислените тежести.
        """
        enriched_scenarios = {}
        
        for actor, actor_scenarios in scenarios.items():
            weighted_scenarios = []
            for scenario in actor_scenarios:
                weight = self.calculate_scenario_weight(
                    scenario,
                    actor_scenarios
                )
                # Допълнителни критерии
                # 1. Уникални термини
                unique_terms = set()
                for sent_type in ["positive", "negative", "neutral"]:
                    unique_terms.update([t["term"] for t in scenario["context_analysis"]["sentence"]["sentiment"][sent_type]["terms"]])
                n_unique_terms = len(unique_terms)

                # 2. Брой актьори (ако има поле actors)
                n_actors = len(scenario.get("actors", []))

                # 3. Брой връзки (ако има cross_document_links)
                n_links = len(scenario.get("cross_document_links", []))

                # 4. Промяна на емоционалната тежест (ако има cross_document_links)
                change_emotion = 0.0
                base_sentiment = scenario["context_analysis"]["sentence"]["sentiment"]
                for link in scenario.get("cross_document_links", []):
                    link_sentiment = link["context_analysis"]["sentence"]["sentiment"]
                    for sent_type in ["positive", "negative", "neutral"]:
                        base_weight = base_sentiment[sent_type]["total_weight"]
                        link_weight = link_sentiment[sent_type]["total_weight"]
                        change_emotion += abs(link_weight - base_weight)

                # Комбиниран индекс
                # Коефициентите могат да се настроят според нуждите
                combined_score = (
                    0.4 * weight +
                    0.15 * n_unique_terms +
                    0.1 * n_actors +
                    0.1 * n_links +
                    0.25 * change_emotion
                )
                # Праг за значимост
                highlighted = combined_score >= 0.7

                enriched_scenario = {
                    **scenario,
                    "weights": {
                        "total": weight,
                        "frequency": self.scenario_frequencies.get(scenario["action"], 0),
                        "context": self.calculate_context_weight(scenario["context_analysis"]),
                        "sentiment": self.calculate_sentiment_weight(scenario["context_analysis"]),
                        "unique_terms": n_unique_terms,
                        "actors": n_actors,
                        "links": n_links,
                        "change_emotion": change_emotion,
                        "combined_score": combined_score
                    },
                    "highlighted": highlighted
                }
                weighted_scenarios.append(enriched_scenario)
            weighted_scenarios.sort(key=lambda s: s["weights"]["combined_score"], reverse=True)
            enriched_scenarios[actor] = weighted_scenarios
        return enriched_scenarios