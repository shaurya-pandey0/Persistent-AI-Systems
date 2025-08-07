# memory/fact_memory.py
# Placeholder file for fact memory functionality
# This can be expanded later for more sophisticated fact extraction and storage

def extract_facts_from_conversation(conversation_text: str):
    """
    Extract facts from conversation text.
    Placeholder implementation.
    """
    # This would contain more sophisticated fact extraction logic
    facts = []
    return facts

def store_facts(facts: list):
    """
    Store extracted facts.
    Placeholder implementation.
    """
    pass


__all__ = ["store_fact", "extract_facts_from_conversation"]

"""#memory/long_term_memory.py
import json, datetime
from pathlib import Path
from typing import List

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

FACTS_FILE = DATA_DIR / "facts.jsonl"

def save_fact(text: str) -> None:
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "fact": text.strip(),
    }
    with FACTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def load_facts() -> List[str]:
    if not FACTS_FILE.exists():
        return []
    try:
        return [
            json.loads(line)["fact"]
            for line in FACTS_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception as e:
        print(f"[Fact Load Error]: {e}")
        return []

def is_probable_fact(text: str) -> bool:
    keywords = ["remember", "my name is", "call me", "i live", "i am from", "i work at", "i like", "i love"]
    return any(kw in text.lower() for kw in keywords)

def store_fact(user_input: str):
    if is_probable_fact(user_input):
        save_fact(user_input.strip())
"""