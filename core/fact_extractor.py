# core/fact_extractor.py (Updated: Merged is_probable_fact from fact_memory.py; kept naive extraction but gated by probability check; added try-except for robustness)
import json
from pathlib import Path

FACTS_PATH = Path("data/facts.json")

def load_facts() -> list[str]:
    if not FACTS_PATH.exists():
        return []
    try:
        with open(FACTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Fact Load Error]: {e}")
        return []

def save_facts(new_facts: list[str]):
    existing = load_facts()
    combined = list(set(existing + new_facts))  # Deduplicate
    try:
        FACTS_PATH.parent.mkdir(exist_ok=True)
        with open(FACTS_PATH, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
    except Exception as e:
        print(f"[Fact Save Error]: {e}")

def is_probable_fact(text: str) -> bool:
    keywords = ["remember", "my name is", "call me", "i live", "i am from", "i work at", "i like", "i love"]
    return any(kw in text.lower() for kw in keywords)

def store_fact(text: str):
    """
    Extract and store facts if probable. Combines naive check with keyword detection.
    """
    fact_candidate = text.strip()
    if (
        len(fact_candidate) > 10
        and not fact_candidate.endswith("?")
        and not fact_candidate.lower().startswith("what")
        and is_probable_fact(fact_candidate)
    ):
        save_facts([fact_candidate])
