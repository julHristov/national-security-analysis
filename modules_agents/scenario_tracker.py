"""
Модул за проследяване развитието на сценарии във времето, анализиращ промените
в контекста и емоционалната окраска между различни документи.
"""
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json
from collections import defaultdict

from config import RESULTS_DIR
from utils.file_manager import read_json_file, write_json_file


class ScenarioTracker:
    """Клас за анализ на развитието на сценарии във времето."""
    
    def __init__(self):
        """Инициализира тракера."""
        self.doc_order = {}  # Хронологичен ред на документите
        self.scenario_chains = defaultdict(list)  # Вериги от свързани сценарии
        
    def _extract_year(self, doc_name: str) -> int:
        """Извлича годината от името на документа (напр. '2_NS_Strategy_2011_ENG' -> 2011)."""
        try:
            return int([part for part in doc_name.split('_') if part.isdigit() and len(part) == 4][0])
        except (IndexError, ValueError):
            return 0
            
    def _sort_documents(self, scenarios: dict) -> List[str]:
        """Сортира документите хронологично по година."""
        doc_years = {doc: self._extract_year(doc) for doc in scenarios.keys()}
        sorted_docs = sorted(scenarios.keys(), key=lambda x: doc_years[x])
        
        # Запазваме реда за по-късна употреба
        self.doc_order = {doc: i for i, doc in enumerate(sorted_docs)}
        return sorted_docs
        
    def _calculate_context_changes(
        self,
        prev_scenario: dict,
        curr_scenario: dict
    ) -> Dict[str, float]:
        """
        Изчислява промените в контекста между два сценария.
        Връща речник с промените за всеки контекст.
        """
        prev_contexts = prev_scenario["context_analysis"]["sentence"]["context"]
        curr_contexts = curr_scenario["context_analysis"]["sentence"]["context"]
        
        all_contexts = set(prev_contexts.keys()) | set(curr_contexts.keys())
        changes = {}
        
        for context in all_contexts:
            prev_value = prev_contexts.get(context, 0)
            curr_value = curr_contexts.get(context, 0)
            changes[context] = curr_value - prev_value
            
        return changes
        
    def _calculate_sentiment_changes(
        self,
        prev_scenario: dict,
        curr_scenario: dict
    ) -> Dict[str, float]:
        """
        Изчислява промените в емоционалната окраска между два сценария.
        Връща речник с промените за всеки тип сантимент.
        """
        prev_sent = prev_scenario["context_analysis"]["sentence"]["sentiment"]
        curr_sent = curr_scenario["context_analysis"]["sentence"]["sentiment"]
        
        changes = {}
        for sent_type in ["positive", "negative", "neutral"]:
            prev_weight = prev_sent[sent_type]["total_weight"]
            curr_weight = curr_sent[sent_type]["total_weight"]
            changes[sent_type] = curr_weight - prev_weight
            
        return changes
        
    def _analyze_scenario_evolution(
        self,
        base_scenario: dict,
        linked_scenarios: List[dict]
    ) -> dict:
        """
        Анализира развитието на един сценарий през времето, проследявайки
        промените в контекста и емоционалната окраска.
        """
        evolution = {
            "base_scenario": base_scenario,
            "timeline": [],
            "context_evolution": defaultdict(list),
            "sentiment_evolution": defaultdict(list),
            "summary": {
                "context_trends": {},
                "sentiment_trends": {},
                "overall_direction": "stable"
            }
        }
        
        # Сортираме свързаните сценарии хронологично
        sorted_scenarios = sorted(
            linked_scenarios,
            key=lambda x: self.doc_order[x["document"]]
        )
        
        prev_scenario = base_scenario
        for link in sorted_scenarios:
            curr_doc = link["document"]
            curr_scenario = link  # Опростено, в реалността трябва да заредим сценария
            
            # Изчисляваме промените
            context_changes = self._calculate_context_changes(prev_scenario, curr_scenario)
            sentiment_changes = self._calculate_sentiment_changes(prev_scenario, curr_scenario)
            
            # Добавяме към еволюцията
            evolution["timeline"].append({
                "document": curr_doc,
                "scenario": curr_scenario,
                "changes": {
                    "context": context_changes,
                    "sentiment": sentiment_changes
                }
            })
            
            # Обновяваме трендовете
            for context, change in context_changes.items():
                evolution["context_evolution"][context].append(change)
                
            for sent_type, change in sentiment_changes.items():
                evolution["sentiment_evolution"][sent_type].append(change)
                
            prev_scenario = curr_scenario
            
        # Анализираме общите трендове
        evolution["summary"]["context_trends"] = self._analyze_trends(
            evolution["context_evolution"]
        )
        evolution["summary"]["sentiment_trends"] = self._analyze_trends(
            evolution["sentiment_evolution"]
        )
        
        # Определяме общата посока на развитие
        evolution["summary"]["overall_direction"] = self._determine_overall_direction(
            evolution["summary"]["context_trends"],
            evolution["summary"]["sentiment_trends"]
        )
        
        return evolution
        
    def _analyze_trends(self, changes: Dict[str, List[float]]) -> Dict[str, str]:
        """
        Анализира трендовете в промените за всяка категория.
        Връща речник с посоката на развитие за всяка категория.
        """
        trends = {}
        for category, values in changes.items():
            if not values:
                trends[category] = "stable"
                continue
                
            # Изчисляваме общата промяна
            total_change = sum(values)
            if abs(total_change) < 0.1:
                trends[category] = "stable"
            elif total_change > 0:
                trends[category] = "increasing"
            else:
                trends[category] = "decreasing"
                
        return trends
        
    def _determine_overall_direction(
        self,
        context_trends: Dict[str, str],
        sentiment_trends: Dict[str, str]
    ) -> str:
        """
        Определя общата посока на развитие на сценария,
        базирано на промените в контекста и емоционалната окраска.
        """
        # Брой на различните посоки в контекста
        context_directions = {
            "increasing": 0,
            "decreasing": 0,
            "stable": 0
        }
        for trend in context_trends.values():
            context_directions[trend] += 1
            
        # Проверяваме емоционалната промяна
        sentiment_direction = "stable"
        if sentiment_trends.get("positive", "") == "increasing":
            sentiment_direction = "improving"
        elif sentiment_trends.get("negative", "") == "increasing":
            sentiment_direction = "worsening"
            
        # Комбинираме двете оценки
        if sentiment_direction != "stable":
            return sentiment_direction
        elif context_directions["increasing"] > context_directions["decreasing"]:
            return "expanding"
        elif context_directions["decreasing"] > context_directions["increasing"]:
            return "contracting"
        else:
            return "stable"
            
    def track_scenarios(self, cross_doc_scenarios: dict) -> Dict[str, List[dict]]:
        """
        Основен метод за проследяване на сценарии.
        Анализира развитието на всеки сценарий през различните документи.
        """
        # Първо сортираме документите хронологично
        sorted_docs = self._sort_documents(cross_doc_scenarios)
        tracked_scenarios = defaultdict(list)
        
        # За всеки документ
        for doc in sorted_docs:
            # За всеки актьор
            for actor, scenarios in cross_doc_scenarios[doc].items():
                # За всеки сценарий
                for scenario in scenarios:
                    # Анализираме развитието му
                    if "cross_document_links" in scenario:
                        evolution = self._analyze_scenario_evolution(
                            scenario,
                            scenario["cross_document_links"]
                        )
                        tracked_scenarios[actor].append(evolution)
                        
        # Записваме резултатите
        out_path = Path(RESULTS_DIR) / "scenarios" / "scenario_evolution.json"
        write_json_file(dict(tracked_scenarios), out_path)
        print(f"✅ Анализът на развитието на сценариите е записан в {out_path}")
        
        return dict(tracked_scenarios)