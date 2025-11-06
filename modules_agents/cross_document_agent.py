# modules_agents/cross_document_agent.py
"""
CrossDocumentAgent:
- Чете per-document scenario JSONs (data/results/scenarios/)
- Открива и свързва подобни сценарии между документи
- Анализира развитието на сценариите във времето
- Изчислява семантична близост между сценарии и документи
"""

from pathlib import Path
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer, util
    _HAS_ST = True
except Exception:
    _HAS_ST = False
    # spaCy fallback will be used

from config import RESULTS_DIR, CLEAN_DIR
from utils.file_manager import read_text_file, read_json_file, write_json_file


class CrossDocumentAgent:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Инициализира агента за cross-document анализ."""
        self.model_name = model_name
        self.model = None
        if _HAS_ST:
            self.model = SentenceTransformer(model_name)
        self.scenario_links = defaultdict(list)
        self.document_order = {}
        
    def _get_doc_text(self, doc_path: Path) -> str:
        """Зарежда текста на документ."""
        return read_text_file(doc_path)
        
    def load_document_scenarios(self, doc_path: Path) -> Dict[str, List[dict]]:
        """Зарежда сценариите от документ."""
        return read_json_file(doc_path)
        
    def get_document_embedding(self, text: str) -> np.ndarray:
        """Генерира embedding за текст."""
        if not text:
            return None
        if self.model:
            return self.model.encode(text, convert_to_tensor=False)
        # fallback: average token vectors via spaCy if available
        import spacy
        nlp = spacy.load("en_core_web_md")
        doc = nlp(text)
        return doc.vector
        
    def load_all_documents(self, clean_dir: Path = Path(CLEAN_DIR)):
        """Зарежда всички документи от директория."""
        docs = []
        for p in sorted(clean_dir.glob("*.txt")):
            docs.append({
                "path": p,
                "stem": p.stem,
                "text": self._get_doc_text(p)
            })
        return docs
        
    def calculate_scenario_similarity(
        self,
        scenario1: dict,
        scenario2: dict,
        weights: Dict[str, float] = None
    ) -> float:
        """
        Изчислява подобност между два сценария базирано на различни критерии:
        - Подобност на действията
        - Подобност на контекстите
        - Подобност в емоционалната окраска
        - Подобност на целите
        """
        if weights is None:
            weights = {
                "action": 0.3,  # Reduced from 0.4
                "context": 0.3,
                "sentiment": 0.2,
                "targets": 0.2   # Increased from 0.1
            }
            
        similarities = {
            "action": self._calculate_action_similarity(
                scenario1["action"],
                scenario2["action"]
            ),
            "context": self._calculate_context_similarity(
                scenario1["context_analysis"],
                scenario2["context_analysis"]
            ),
            "sentiment": self._calculate_sentiment_similarity(
                scenario1["context_analysis"],
                scenario2["context_analysis"]
            ),
            "targets": self._calculate_targets_similarity(
                scenario1.get("targets", []),
                scenario2.get("targets", [])
            )
        }
        
        # Добавяме и семантична близост ако имаме sentence-transformers
        if self.model:
            text1 = f"{scenario1['action']} {' '.join(scenario1.get('targets', []))}"
            text2 = f"{scenario2['action']} {' '.join(scenario2.get('targets', []))}"
            semantic_sim = float(
                util.pytorch_cos_sim(
                    self.model.encode(text1),
                    self.model.encode(text2)
                )[0][0]
            )
            similarities["semantic"] = semantic_sim
            weights["semantic"] = 0.2
            # Преизчисляваме останалите тегла
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
        
        # Изчисляваме претеглена сума
        total_similarity = sum(
            weight * similarities[key]
            for key, weight in weights.items()
        )
        
        return total_similarity

    def _calculate_action_similarity(self, action1: str, action2: str) -> float:
        """Изчислява подобност между действия."""
        # Базово сравнение - може да се разшири със синоними
        if not action1 or not action2:
            return 0.0
        return 1.0 if action1.lower() == action2.lower() else 0.0
        
    def _calculate_context_similarity(
        self,
        context1: dict,
        context2: dict
    ) -> float:
        """Изчислява подобност между контексти."""
        # Вземаме контекстите от изреченията
        contexts1 = set(context1["sentence"]["context"].keys())
        contexts2 = set(context2["sentence"]["context"].keys())
        
        if not contexts1 or not contexts2:
            return 0.0
            
        # Изчисляваме Jaccard similarity
        intersection = len(contexts1 & contexts2)
        union = len(contexts1 | contexts2)
        
        return intersection / union
        
    def _calculate_sentiment_similarity(
        self,
        context1: dict,
        context2: dict
    ) -> float:
        """Изчислява подобност в емоционалната окраска."""
        sent1 = context1["dominant_sentiment"]
        sent2 = context2["dominant_sentiment"]
        
        # Проверяваме дали са от един и същ тип
        type_match = 1.0 if sent1["type"] == sent2["type"] else 0.0
        
        # Сравняваме и теглата
        weight_diff = abs(sent1["weight"] - sent2["weight"])
        weight_sim = 1.0 - min(weight_diff, 1.0)
        
        return 0.7 * type_match + 0.3 * weight_sim
        
    def _calculate_targets_similarity(
        self,
        targets1: List[str],
        targets2: List[str]
    ) -> float:
        """Изчислява подобност между целите."""
        if not targets1 or not targets2:
            return 0.0
            
        # Превръщаме в множества и нормализираме
        targets1_set = {t.lower() for t in targets1}
        targets2_set = {t.lower() for t in targets2}
        
        # Jaccard similarity
        intersection = len(targets1_set & targets2_set)
        union = len(targets1_set | targets2_set)
        
        return intersection / union
        
    def find_similar_scenarios(
        self,
        base_scenario: dict,
        other_scenarios: List[dict],
        threshold: float = 0.4  # Намален праг за по-либерално свързване на сценарии
    ) -> List[Tuple[dict, float]]:
        """
        Намира подобни сценарии над определен праг на подобност.
        Връща списък от двойки (сценарий, степен_на_подобност).
        """
        similar = []
        
        for scenario in other_scenarios:
            similarity = self.calculate_scenario_similarity(
                base_scenario,
                scenario
            )
            
            if similarity >= threshold:
                similar.append((scenario, similarity))
                
        # Сортираме по подобност
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar
        
    def analyze_cross_document_patterns(self):
        """Анализира общи шаблони между документите."""
        docs = self.load_all_documents()
        embeddings = []
        for d in docs:
            emb = self.get_document_embedding(d["text"])
            embeddings.append(np.array(emb))
            
        # compute pairwise cosine similarities
        sims = np.inner(embeddings, embeddings)
        norms = np.linalg.norm(embeddings, axis=1)
        denom = np.outer(norms, norms) + 1e-12
        cosine = sims / denom
        
        # simple semantic drift: 1 - mean similarity to previous document
        drift_scores = []
        for i in range(len(docs)):
            if i == 0:
                drift_scores.append(0.0)
            else:
                drift = 1.0 - float(np.mean(cosine[i, :i]))
                drift_scores.append(drift)
                
        return {
            "docs": [d["stem"] for d in docs],
            "cosine_matrix": cosine.tolist(),
            "drift_scores": drift_scores
        }
        
    def link_scenarios_across_documents(
        self,
        scenarios_by_doc: Dict[str, Dict[str, List[dict]]],
        threshold: float = 0.7
    ) -> Dict[str, Dict[str, List[dict]]]:
        """
        Свързва подобни сценарии между документите и обогатява ги с cross-document
        препратки.
        """
        # Първо определяме хронологичния ред на документите по имената им
        doc_names = list(scenarios_by_doc.keys())
        doc_names.sort()  # Очакваме имената да са от типа 1_doc, 2_doc и т.н.
        
        for i, doc_name in enumerate(doc_names):
            self.document_order[doc_name] = i
            
        # За всеки документ
        enriched_scenarios = {}
        for curr_doc in tqdm(doc_names, desc="Анализ между документи"):
            enriched_scenarios[curr_doc] = {}
            
            # За всеки актьор в документа
            for actor, scenarios in scenarios_by_doc[curr_doc].items():
                enriched_actor_scenarios = []
                
                # За всеки сценарий на актьора
                for scenario in scenarios:
                    # Търсим подобни сценарии в другите документи
                    cross_doc_links = []
                    
                    for other_doc in doc_names:
                        if other_doc == curr_doc:
                            continue
                            
                        # Вземаме сценариите на същия актьор от другия документ
                        other_scenarios = scenarios_by_doc[other_doc].get(actor, [])
                        
                        # Намираме подобните сценарии
                        similar = self.find_similar_scenarios(
                            scenario,
                            other_scenarios,
                            threshold
                        )
                        
                        # Добавяме връзки към подобните сценарии
                        for similar_scenario, similarity in similar:
                            cross_doc_links.append({
                                "document": other_doc,
                                "similarity": similarity,
                                "scenario_id": similar_scenario.get("id", "unknown"),
                                "chronological_order": self.document_order[other_doc]
                            })
                    
                    # Обогатяваме сценария с cross-document връзки
                    enriched_scenario = {
                        **scenario,
                        "cross_document_links": sorted(
                            cross_doc_links,
                            key=lambda x: (x["chronological_order"], -x["similarity"])
                        )
                    }
                    
                    enriched_actor_scenarios.append(enriched_scenario)
                    
                enriched_scenarios[curr_doc][actor] = enriched_actor_scenarios
                
        return enriched_scenarios
        
    def analyze_scenarios_across_documents(
        self,
        base_dir: Path = None,
        threshold: float = 0.7
    ) -> Dict[str, Dict[str, List[dict]]]:
        """
        Основен метод за анализ на сценарии между документи.
        Зарежда всички сценарии, открива връзки и ги обогатява с cross-document
        информация.
        """
        if base_dir is None:
            base_dir = Path(RESULTS_DIR) / "scenarios"
            
        # Зареждаме всички сценарии
        print("📚 Зареждане на сценарии от всички документи...")
        scenarios_by_doc = {}
        for file in base_dir.glob("*_scenarios.json"):
            doc_name = file.stem.replace("_scenarios", "")
            scenarios_by_doc[doc_name] = self.load_document_scenarios(file)
            
        # Свързваме сценариите между документите
        print("🔍 Търсене на подобни сценарии между документите...")
        enriched_scenarios = self.link_scenarios_across_documents(
            scenarios_by_doc,
            threshold
        )
        
        # Записваме резултата
        print("💾 Записване на резултатите...")
        out_path = base_dir / "cross_document_scenarios.json"
        write_json_file(enriched_scenarios, out_path)
        print(f"✅ Cross-document анализът е записан в {out_path}")
        
        return enriched_scenarios
