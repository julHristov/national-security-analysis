import json
from pathlib import Path
from collections import defaultdict
import spacy
from typing import Dict, List, Set, Tuple

from config import CLEAN_DIR, RESULTS_DIR
from utils.file_manager import read_text_file, write_json_file


class TermWeightExtractor:
    """Клас за извличане на термини и изчисляване на техните тегла според честотата."""
    
    def __init__(self, schema_dir: str = None):
        """Инициализира екстрактора с необходимите речници."""
        if not schema_dir:
            schema_dir = Path(__file__).parent.parent / "schema"
            
        # Зареждане на речниците с термини
        with open(schema_dir / "positive_terms.json", "r", encoding="utf-8") as f:
            self.positive_terms = set(json.load(f))
            
        with open(schema_dir / "negative_terms.json", "r", encoding="utf-8") as f:
            self.negative_terms = set(json.load(f))
            
        with open(schema_dir / "neutral_terms.json", "r", encoding="utf-8") as f:
            self.neutral_terms = set(json.load(f))
            
        # Речници за съхранение на честотите
        self.term_frequencies = {
            "positive": defaultdict(int),
            "negative": defaultdict(int),
            "neutral": defaultdict(int)
        }
        
        # Речник за съхранение на теглата
        self.term_weights = {
            "positive": {},
            "negative": {},
            "neutral": {}
        }
        
    def _get_term_type(self, term: str) -> str:
        """Определя типа на термина (positive, negative, neutral)."""
        if term in self.positive_terms:
            return "positive"
        elif term in self.negative_terms:
            return "negative"
        elif term in self.neutral_terms:
            return "neutral"
        return None
        
    def extract_terms(self, text: str) -> Dict[str, List[Tuple[str, int]]]:
        """Извлича термини от текста и брои честотата им."""
        text = text.lower()
        
        # Извличаме всички термини и честотите им
        for term_type, terms in [
            ("positive", self.positive_terms),
            ("negative", self.negative_terms),
            ("neutral", self.neutral_terms)
        ]:
            for term in terms:
                count = text.count(term)
                if count > 0:
                    self.term_frequencies[term_type][term] += count
                    
        return {k: dict(v) for k, v in self.term_frequencies.items()}
    
    def calculate_weights(self):
        """Изчислява тегла за термините според честотата им."""
        for term_type in ["positive", "negative", "neutral"]:
            frequencies = self.term_frequencies[term_type]
            if not frequencies:
                continue
                
            # Намираме максималната честота за нормализация
            max_freq = max(frequencies.values())
            
            # Изчисляваме теглата (0.1 до 1.0) според относителната честота
            self.term_weights[term_type] = {
                term: 0.1 + 0.9 * (freq / max_freq)
                for term, freq in frequencies.items()
            }
            
    def get_term_weight(self, term: str) -> Tuple[str, float]:
        """Връща типа и теглото на даден термин."""
        term = term.lower()
        term_type = self._get_term_type(term)
        if term_type:
            return term_type, self.term_weights[term_type].get(term, 0.1)
        return None, 0.0
    
    def process_documents(self):
        """Обработва всички документи и извлича/изчислява тегла на термините."""
        print("📊 Извличане на честоти на термините...")
        
        # Обработваме всички документи
        clean_dir = Path(CLEAN_DIR)
        for file in clean_dir.glob("*.txt"):
            text = read_text_file(file)
            print(f"📄 Обработка на {file.name}...")
            self.extract_terms(text)
            
        # Изчисляваме теглата
        print("⚖️ Изчисляване на тегла...")
        self.calculate_weights()
        
        # Записваме резултатите
        results = {
            "frequencies": self.term_frequencies,
            "weights": self.term_weights
        }
        
        out_dir = Path(RESULTS_DIR)
        out_path = out_dir / "term_weights.json"
        write_json_file(results, out_path)
        print(f"✅ Теглата са записани в {out_path}")
        
        return results


def main():
    extractor = TermWeightExtractor()
    extractor.process_documents()
    

if __name__ == "__main__":
    main()