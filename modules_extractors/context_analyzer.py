from collections import Counter
from pathlib import Path
import json
from collections import defaultdict
import spacy
from typing import Dict, List, Tuple

class ContextAnalyzer:
    def document_term_statistics(self, text: str) -> dict:
        """
        Извлича статистика за термините в документа:
        - Топ 10 позитивни, негативни, неутрални термини
        - Обща оценка на средата
        """
        text = text.lower()
        pos_counter = Counter()
        neg_counter = Counter()
        neu_counter = Counter()

        for entry in self.positive_terms:
            if entry["term"] in text:
                pos_counter[entry["term"]] += text.count(entry["term"])
        for entry in self.negative_terms:
            if entry["term"] in text:
                neg_counter[entry["term"]] += text.count(entry["term"])
        for entry in self.neutral_terms:
            if entry["term"] in text:
                neu_counter[entry["term"]] += text.count(entry["term"])

        top_pos = pos_counter.most_common(10)
        top_neg = neg_counter.most_common(10)
        top_neu = neu_counter.most_common(10)

        # Оценка на средата
        total_pos = sum(pos_counter.values())
        total_neg = sum(neg_counter.values())
        total_neu = sum(neu_counter.values())
        if total_pos > total_neg and total_pos > total_neu:
            overall = "positive"
        elif total_neg > total_pos and total_neg > total_neu:
            overall = "negative"
        else:
            overall = "neutral"

        return {
            "top_positive_terms": top_pos,
            "top_negative_terms": top_neg,
            "top_neutral_terms": top_neu,
            "overall_environment": overall,
            "total_positive": total_pos,
            "total_negative": total_neg,
            "total_neutral": total_neu
        }
def aggregate_top_terms_across_documents(doc_texts: list, analyzer: ContextAnalyzer, top_n: int = 10) -> dict:
    """
    Групира топ термините от всички документи и извежда най-влиятелните сред всички.
    """
    global_counter = Counter()
    for text in doc_texts:
        stats = analyzer.document_term_statistics(text)
        for term, count in stats["top_positive_terms"]:
            global_counter[term] += count
        for term, count in stats["top_negative_terms"]:
            global_counter[term] += count
        for term, count in stats["top_neutral_terms"]:
            global_counter[term] += count
    top_global = global_counter.most_common(top_n)
    return {"top_influential_terms": top_global}
    """Клас за анализ на контекстуална терминология в текст."""
    
    def __init__(self, schema_dir: str = None):
        """Инициализира анализатора с необходимите речници."""
        if not schema_dir:
            schema_dir = Path(__file__).parent.parent / "schema"
            
        # Зареждане на тематични речници
        with open(schema_dir / "context_dictionaries.json", "r", encoding="utf-8") as f:
            self.dictionaries = json.load(f)
        
        # Зареждане на речници за конотация
        with open(schema_dir / "positive_terms.json", "r", encoding="utf-8") as f:
            self.positive_terms = json.load(f)
        with open(schema_dir / "negative_terms.json", "r", encoding="utf-8") as f:
            self.negative_terms = json.load(f)
        with open(schema_dir / "neutral_terms.json", "r", encoding="utf-8") as f:
            self.neutral_terms = json.load(f)
            
        # Създаваме речници за бързо търсене на тематични термини
        self._security_contexts = {
            term.lower(): context
            for context, terms in self.dictionaries["security_contexts"].items()
            for term in terms
        }
        
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Анализира емоционалната окраска на текста чрез речниците с тегла."""
        text = text.lower()
        sentiment_weights = {
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 0.0
        }
        # Търсим съвпадения с термини от всеки речник и сумираме теглата
        for entry in self.positive_terms:
            if entry["term"] in text:
                sentiment_weights["positive"] += entry["weight"]
        for entry in self.negative_terms:
            if entry["term"] in text:
                sentiment_weights["negative"] += entry["weight"]
        for entry in self.neutral_terms:
            if entry["term"] in text:
                sentiment_weights["neutral"] += entry["weight"]
        return sentiment_weights
        
    def analyze_sentence_context(self, sentence: str) -> Dict[str, dict]:
        """Анализира контекста на едно изречение, включително тежестите, топ термините и общата среда."""
        sentence = sentence.lower()
        contexts = defaultdict(int)
        
        # Анализ на тематичен контекст
        for term, context in self._security_contexts.items():
            if term in sentence:
                contexts[context] += 1
        
        # Анализ на емоционална окраска
        sentiment_weights = self.analyze_sentiment(sentence)
        
        pos_counter = Counter()
        neg_counter = Counter()
        neu_counter = Counter()
        
        # Търсене на топ термини
        for entry in self.positive_terms:
            if entry["term"] in sentence:
                pos_counter[entry["term"]] += sentence.count(entry["term"])
        for entry in self.negative_terms:
            if entry["term"] in sentence:
                neg_counter[entry["term"]] += sentence.count(entry["term"])
        for entry in self.neutral_terms:
            if entry["term"] in sentence:
                neu_counter[entry["term"]] += sentence.count(entry["term"])

        top_pos = pos_counter.most_common(5)
        top_neg = neg_counter.most_common(5)
        top_neu = neu_counter.most_common(5)

        # Оценка на средата
        total_pos = sentiment_weights["positive"]
        total_neg = sentiment_weights["negative"]
        total_neu = sentiment_weights["neutral"]
        if total_pos > total_neg and total_pos > total_neu:
            overall = "positive"
        elif total_neg > total_pos and total_neg > total_neu:
            overall = "negative"
        else:
            overall = "neutral"

        return {
            "security_contexts": dict(contexts),
            "sentiment_weights": sentiment_weights,
            "top_positive_terms": top_pos,
            "top_negative_terms": top_neg,
            "top_neutral_terms": top_neu,
            "overall_environment": overall
        }
    
    def get_dominant_context(self, context_analysis: Dict[str, dict]) -> Tuple[str, str]:
        """Определя доминиращия контекст и неговия тон от анализа."""
        contexts = context_analysis["security_contexts"]
        sentiment = context_analysis["sentiment"]
        
        # Намираме най-честия контекст
        dominant_context = max(contexts.items(), key=lambda x: x[1])[0] if contexts else "undefined"
        
        # Определяме преобладаващия тон
        sentiment_type = max(sentiment.items(), key=lambda x: x[1])[0]
        
        return dominant_context, sentiment_type
    
    def analyze_scenario_context(self, scenario_data: dict) -> dict:
        """Анализира контекста на цял сценарий."""
        # Анализираме контекста на цялото изречение
        sentence_context = self.analyze_sentence_context(scenario_data["sentence"])
        
        # Анализираме контекста на фразата с действието
        action_context = self.analyze_sentence_context(scenario_data["action_phrase"])
        
        # Определяме доминиращия контекст и тон
        dominant_context, context_tone = self.get_dominant_context(sentence_context)
        
        # Добавяме анализа към сценария
        return {
            **scenario_data,
            "context_analysis": {
                "sentence_context": sentence_context,
                "action_context": action_context,
                "dominant_context": dominant_context,
                "context_tone": context_tone
            }
        }