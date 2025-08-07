# persona/faiss_memory_writer.py

import json
from datetime import datetime
from pathlib import Path

from utils.session_id import get_or_create_session_file
from config.constants import FAISS_MEMORY_JSON  # e.g. "persona/faiss_memory_state.json"

def determine_convo_phase(text: str) -> str:
    if any(w in text.lower() for w in ["start", "hello", "hi"]):
        return "onboarding"
    elif any(w in text.lower() for w in ["why", "how", "what"]):
        return "engagement"
    return "closure"

def extract_topics(text: str) -> list:
    if "adorable" in text.lower():
        return ["compliments", "emotional bonding"]
    return ["general conversation"]

def update_faiss_memory_state_from_session(session_id: str):
    session_file = Path(f"data/session_{session_id}.json")
    if not session_file.exists():
        print(f"[FAISS Memory]: Session file not found: {session_file}")
        return

    with open(session_file, encoding="utf-8") as f:
        session_data = json.load(f)

    if not session_data:
        print("[FAISS Memory]: Session file is empty.")
        return

    last_turn = session_data[-1]
    user_input = last_turn.get("user", "")

    entry = {
        "timestamp": datetime.now().isoformat(),
        "turn": last_turn,
        "memory_context": {
            "conversation_phase": determine_convo_phase(user_input),
            "topic_stack": extract_topics(user_input),
            "preferences": ["empathy", "quick wit"],
            "intent_trend": "emotional exploration",
            "rapport_level": 0.88,
            "attachment_style": "secure"
        }
    }

    # Dump to faiss memory
    Path("persona").mkdir(exist_ok=True)
    with open(FAISS_MEMORY_JSON, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"✅ Appended to {FAISS_MEMORY_JSON}")
