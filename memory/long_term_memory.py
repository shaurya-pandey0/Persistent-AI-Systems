# memory/long_term_memory.py
import json
from pathlib import Path
from typing import Dict, List, Any

DATA_DIR = Path("data")  
DATA_DIR.mkdir(exist_ok=True)
LONG_TERM_FILE = DATA_DIR / "long_term_memory.jsonl"

def append_long_term_memory(entry: Dict[str, Any]) -> None:
    with LONG_TERM_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def load_long_term_memory(user_id: str = None) -> List[Dict[str, Any]]:
    if not LONG_TERM_FILE.exists():
        return []
    results = []
    try:
        with LONG_TERM_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if user_id is None or data.get("user_id") == user_id:
                        results.append(data)
    except Exception as e:
        print(f"[Long-term memory load error]: {e}")
    return results
