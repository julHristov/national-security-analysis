# modules_agents/dynamic_lexicon.py
"""
Dynamic Lexicon Expansion:
- Взема базов речник (data/lookup/*.json)
- За даден domain corpus (list of docs) предлага семантично близки термини
- Използва sentence-transformers или spaCy fallback
"""

from pathlib import Path
import json
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer, util

    _HAS_ST = True
except Exception:
    _HAS_ST = False

from config import DATA_DIR

LOOKUP_DIR = Path(DATA_DIR) / "lookup"


def load_base_lexicon(file_path: Path):
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


class DynamicLexiconExpander:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        if _HAS_ST:
            self.model = SentenceTransformer(model_name)

    def expand(self, base_terms: list, domain_texts: list, top_k: int = 10, score_threshold: float = 0.7):
        """
        base_terms: list[str]
        domain_texts: list[str] (corpus)
        returns: dict{term: [(candidate, score), ...]}
        """
        if not self.model:
            # fallback: no-op
            return {}
        # encode all domain n-grams (naive: use sentences)
        domain_embs = self.model.encode(domain_texts, convert_to_tensor=True)
        expansions = {}
        for term in base_terms:
            term_emb = self.model.encode(term, convert_to_tensor=True)
            sims = util.cos_sim(term_emb, domain_embs)[0].cpu().numpy()
            # pick top sentences with high similarity -> extract candidate tokens
            idxs = sims.argsort()[::-1][:top_k]
            candidates = []
            for idx in idxs:
                score = float(sims[idx])
                if score < score_threshold:
                    continue
                sent = domain_texts[int(idx)]
                # naive extraction: split and pick nouns (could be improved)
                for token in sent.split():
                    if token.lower() not in term.lower():
                        candidates.append((token.strip(",.()\"'"), score))
            # aggregate scores by candidate
            agg = defaultdict(float)
            for cand, sc in candidates:
                agg[cand.lower()] += sc
            # return sorted list
            sorted_cands = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:top_k]
            expansions[term] = sorted_cands
        return expansions
