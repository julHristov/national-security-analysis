import spacy
import json
from pathlib import Path
from collections import defaultdict
import argparse
from config import *


def get_logger(name):
    class SimpleLogger:
        def info(self, msg): print(f"INFO: {msg}")

        def warning(self, msg): print(f"WARNING: {msg}")

        def error(self, msg): print(f"ERROR: {msg}")

    return SimpleLogger()


logger = get_logger(__name__)


class ScenarioExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load(SPACY_MODEL)
            logger.info(f"Loaded spaCy model: {SPACY_MODEL}")
        except OSError:
            logger.error(f"Please install spaCy model: python -m spacy download {SPACY_MODEL}")
            raise

        # Load dictionaries
        self.positive_terms = self.load_terms(POSITIVE_TERMS_PATH)
        self.negative_terms = self.load_terms(NEGATIVE_TERMS_PATH)
        self.neutral_terms = self.load_terms(NEUTRAL_TERMS_PATH)

    def load_terms(self, file_path):
        """Load terms from JSON file"""
        path = Path(file_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('positive_terms', data.get('negative_terms', data.get('neutral_terms', [])))
        return []

    def extract_scenarios(self, text, target_actor=None):
        """Extract scenarios from text using dependency parsing"""
        doc = self.nlp(text)
        scenarios = []

        for sent in doc.sents:
            scenarios.extend(self.analyze_sentence(sent, target_actor))

        return scenarios

    def analyze_sentence(self, sentence, target_actor=None):
        """Analyze a single sentence for scenarios"""
        scenarios = []

        # Find verbs and their dependencies
        for token in sentence:
            if token.pos_ == "VERB":
                scenario = self.extract_actor_action_pattern(token, target_actor)
                if scenario:
                    scenarios.append(scenario)

        return scenarios

    def extract_actor_action_pattern(self, verb_token, target_actor=None):
        """Extract Actor-Action-Target patterns from dependency tree"""
        # Find subject (actor)
        actor = None
        for child in verb_token.children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                actor = child.text
                break

        # If target actor specified, check if it matches
        if target_actor and actor and target_actor.lower() not in actor.lower():
            return None

        # Find object (target)
        target = None
        concept = None

        for child in verb_token.children:
            if child.dep_ in ["dobj", "pobj"]:
                target = child.text
            elif child.dep_ == "prep":
                # Handle prepositional phrases like "for stability"
                for grandchild in child.children:
                    if grandchild.dep_ in ["pobj"]:
                        concept = grandchild.text

        # Only return if we have at least actor and action
        if actor and verb_token.lemma_:
            return {
                "actor_1": actor,
                "action": verb_token.lemma_,
                "actor_2": target,
                "concept": concept,
                "sentence": sentence.text,
                "confidence": self.calculate_confidence(verb_token, actor, target)
            }

        return None

    def calculate_confidence(self, verb, actor, target):
        """Calculate confidence score for the extracted scenario"""
        confidence = 0.5  # Base confidence

        # Boost confidence if we have clear dependencies
        if actor and verb:
            confidence += 0.3

        if target:
            confidence += 0.2

        return min(confidence, 1.0)

    def analyze_context_sentiment(self, text):
        """Analyze sentiment of text using dictionaries"""
        words = text.lower().split()

        positive_count = sum(1 for word in words if word in self.positive_terms)
        negative_count = sum(1 for word in words if word in self.negative_terms)
        neutral_count = sum(1 for word in words if word in self.neutral_terms)

        total_terms = len(words)

        return {
            "positive_score": positive_count / total_terms if total_terms > 0 else 0,
            "negative_score": negative_count / total_terms if total_terms > 0 else 0,
            "neutral_score": neutral_count / total_terms if total_terms > 0 else 0
        }


def main():
    parser = argparse.ArgumentParser(description='Extract scenarios from documents')
    parser.add_argument('--actor', type=str, help='Target actor to analyze')
    parser.add_argument('--file', type=str, required=True, help='Input text file')

    args = parser.parse_args()

    extractor = ScenarioExtractor()

    # Read input file
    with open(args.file, 'r', encoding='utf-8') as f:
        text = f.read()

    scenarios = extractor.extract_scenarios(text, args.actor)

    # Analyze context sentiment
    sentiment = extractor.analyze_context_sentiment(text)

    # Prepare output
    output = {
        "file": args.file,
        "target_actor": args.actor,
        "scenarios": scenarios,
        "context_sentiment": sentiment,
        "total_scenarios": len(scenarios)
    }

    # Save results
    output_file = SCENARIOS_DIR / f"{Path(args.file).stem}_scenarios.json"
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(scenarios)} scenarios. Saved to {output_file}")


# Добави това в края на relations_extractor.py ПРЕД if __name__ == '__main__':

def process_all_files():
    """Main function to process all files - for integration with main.py"""
    logger.info(">>> Relations extractor started...")
    relations_schema = load_schema("relation_types.json")
    if not relations_schema:
        logger.error("✗ Relations schema is empty. Aborting extraction.")
        return

    RELATIONS_DIR.mkdir(parents=True, exist_ok=True)

    if not any(CLEAN_DIR.glob("*.txt")):
        logger.warning(f"No text files in {CLEAN_DIR}.")
        return

    for file_path in CLEAN_DIR.glob("*.txt"):
        logger.info(f"Processing file: {file_path.name}")
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        relations = extract_relations(text, relations_schema)
        output_path = RELATIONS_DIR / f"{file_path.stem}_relations.json"
        with open(output_path, 'w', encoding='utf-8') as out:
            json.dump(relations, out, ensure_ascii=False, indent=2)
        logger.info(f" Retrieved {len(relations)} relations from {file_path.name}")


if __name__ == "__main__":
    main()