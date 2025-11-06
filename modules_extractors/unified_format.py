"""
Модул за обединяване и структуриране на всички анализи на сценарии
в единен JSON формат.
"""
from pathlib import Path
from typing import Dict, List
import json
from collections import defaultdict

from config import RESULTS_DIR
from utils.file_manager import read_json_file, write_json_file


class UnifiedScenarioFormat:
    """
    Клас за обединяване на всички анализи в единен JSON формат:
    - Базови сценарии с тежести
    - Cross-document връзки
    - Анализ на развитието
    """
    
    def __init__(self):
        """Инициализира форматера."""
        self.scenarios_dir = Path(RESULTS_DIR) / "scenarios"
        
    def load_all_analyses(self) -> tuple:
        """Зарежда всички налични анализи."""
        # Зареждаме базовите сценарии с cross-document връзки
        cross_doc_path = self.scenarios_dir / "cross_document_scenarios.json"
        cross_doc_scenarios = read_json_file(cross_doc_path)
        
        # Зареждаме анализа на развитието
        evolution_path = self.scenarios_dir / "scenario_evolution.json"
        scenario_evolution = read_json_file(evolution_path)
        
        # Зареждаме теглата на термините (опционално)
        weights_path = self.scenarios_dir / "term_weights.json"
        try:
            term_weights = read_json_file(weights_path)
        except:
            term_weights = {}
        
        return cross_doc_scenarios, scenario_evolution, term_weights
        
    def create_unified_format(self) -> dict:
        """
        Създава обединен формат, който включва всички аспекти на анализа:
        1. Мета информация (времеви период, брой документи и т.н.)
        2. Речници с термини и тегла
        3. Сценарии по документи с:
           - Базова информация
           - Контекстуален анализ
           - Тежести
           - Cross-document връзки
        4. Обобщен анализ на развитието
        """
        # Зареждаме всички анализи
        cross_doc, evolution, weights = self.load_all_analyses()
        
        # Извличаме времевия период от имената на документите
        years = []
        for doc in cross_doc.keys():
            try:
                year = int([p for p in doc.split('_') if p.isdigit() and len(p) == 4][0])
                years.append(year)
            except (IndexError, ValueError):
                continue
                
        # Създаваме базовата структура
        unified_data = {
            "meta": {
                "time_period": {
                    "start_year": min(years) if years else None,
                    "end_year": max(years) if years else None
                },
                "document_count": len(cross_doc),
                "actor_count": len(set(actor for doc in cross_doc.values() for actor in doc))
            },
            "term_weights": weights,
            "scenarios_by_document": {},
            "scenario_evolution": {}
        }
        
        # Обработваме сценариите по документи
        for doc_name, doc_scenarios in cross_doc.items():
            doc_data = {}
            
            for actor, scenarios in doc_scenarios.items():
                # Търсим еволюцията, където базовият сценарий е от текущия документ
                actor_evolution = None
                for e in evolution.get(actor, []):
                    if doc_name in str(e["base_scenario"]):
                        actor_evolution = e
                        break
                
                # Обединяваме информацията за всеки сценарий
                enriched_scenarios = []
                for scenario in scenarios:
                    enriched_scenario = {
                        "basic_info": {
                            "actor": scenario["actor_1"],
                            "action": scenario["action"],
                            "targets": scenario["targets"],
                            "sentence": scenario["sentence"]
                        },
                        "context_analysis": scenario["context_analysis"],
                        "weights": scenario.get("weights", {}),
                        "cross_document_links": scenario.get("cross_document_links", [])
                    }
                    
                    # Добавяме информация за развитието ако е налична
                    if actor_evolution:
                        enriched_scenario["evolution"] = {
                            "timeline": actor_evolution["timeline"],
                            "trends": {
                                "context": actor_evolution["summary"]["context_trends"],
                                "sentiment": actor_evolution["summary"]["sentiment_trends"]
                            },
                            "overall_direction": actor_evolution["summary"]["overall_direction"]
                        }
                        
                    enriched_scenarios.append(enriched_scenario)
                    
                doc_data[actor] = enriched_scenarios
                
            unified_data["scenarios_by_document"][doc_name] = doc_data
            
        # Добавяме обобщен анализ на развитието
        unified_data["scenario_evolution"] = {
            actor: [{
                "base_document": next(
                    doc for doc in cross_doc.keys()
                    if doc in str(e["base_scenario"])
                ),
                "timeline": e.get("timeline", []),
                "context_evolution": e.get("context_evolution", {}),
                "sentiment_evolution": e.get("sentiment_evolution", {}),
                "summary": e.get("summary", {})
            } for e in evol]
            for actor, evol in evolution.items()
        }
        
        # Записваме резултата
        out_path = self.scenarios_dir / "unified_scenarios.json"
        write_json_file(unified_data, out_path)
        print(f"✅ Обединеният анализ е записан в {out_path}")
        
        return unified_data