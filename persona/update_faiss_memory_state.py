# update_faiss_memory_state.py - CORRECTED VERSION

"""Improvements
===============
1. Robust session discovery even if folder missing.
2. Better topic extraction via simple keyword mapping.
3. Added safeguard to truncate excessive FAISS memory lines (>10k).
4. Type annotations and clearer error handling.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

PERSONA_DIR = Path("persona")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
FAISS_MEMORY_JSON = PERSONA_DIR / "faiss_memory_state.json"

# -------------------- Utility Helpers -------------------- #

KEYWORD_TOPICS = {
    "adorable": ["compliments", "emotional bonding"],
    "joke": ["humor"],
    "sad": ["empathy", "support"],
    "project": ["goal discussion"],
}

def determine_convo_phase(text: str) -> str:
    lower = text.lower()
    if any(lower.startswith(g) for g in ["hi", "hello", "hey"]):
        return "onboarding"
    if any(q in lower for q in ["why", "how", "what", "tell me"]):
        return "engagement"
    return "closure"


def extract_topics(text: str) -> List[str]:
    topics = set()
    for kw, mapped in KEYWORD_TOPICS.items():
        if kw in text.lower():
            topics.update(mapped)
    return list(topics) or ["general conversation"]


def latest_session_file() -> Optional[Path]:
    sessions = sorted(DATA_DIR.glob("session_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return sessions[0] if sessions else None


def last_user_input(session_file: Path) -> Optional[str]:
    try:
        data = json.loads(session_file.read_text(encoding="utf-8"))
        return data[-1]["user"] if data else None
    except Exception as exc:
        print(f"❌ Failed to read {session_file}: {exc}")
        return None

# -------------------- Main Logic ------------------------ #

def main():
    session = latest_session_file()
    if not session:
        print("⚠️ No session file located.")
        return

    user_input = last_user_input(session)
    if not user_input:
        print("⚠️ Session contains no user turns.")
        return

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "turn": {"user": user_input, "assistant": "_placeholder_"},
        "memory_context": {
            "conversation_phase": determine_convo_phase(user_input),
            "topic_stack": extract_topics(user_input),
            "intent_trend": "emotional exploration",
            "rapport_level": 0.88,
        },
    }

    # Truncate file if too big (>10k lines) to prevent huge FAISS JSONL
    if FAISS_MEMORY_JSON.exists():
        lines = FAISS_MEMORY_JSON.read_text(encoding="utf-8").splitlines()
        if len(lines) > 10_000:
            FAISS_MEMORY_JSON.write_text("\n".join(lines[-8_000:]), encoding="utf-8")

    with FAISS_MEMORY_JSON.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"✅ FAISS memory updated ({session.name}) → {FAISS_MEMORY_JSON}")

if __name__ == "__main__":
    main()