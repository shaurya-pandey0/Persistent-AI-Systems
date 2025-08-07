# memory/session_summarizer.py
import json
from pathlib import Path
from typing import List, Dict, Any
from core.api_client import get_completion
from memory.long_term_memory import append_long_term_memory

def summarize_session(user_id: str, session_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """Summarize a session and append to long-term memory."""
    raw_text = "\n".join(
        f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        for turn in session_history
    )

    prompt = (
        "You are a memory summarizer.\n"
        "Summarize this conversation into key user traits, preferences, facts, "
        "and relationship notes as JSON with keys: summary, topics, trust_level.\n"
        "Conversation:\n"
        f"{raw_text}\n"
        "Output ONLY the JSON."
    )

    # Call LLM - commented out for now as requested
    # response = get_completion([{"role": "system", "content": prompt}])
    # For now, create a basic summary without LLM
    try:
        # response = get_completion([{"role": "system", "content": prompt}])
        # summary = json.loads(response)

        # Placeholder summary for now (replace when LLM is enabled)
        summary = {
            "summary": f"Session with {len(session_history)} turns covering various topics",
            "topics": ["general conversation"],
            "trust_level": "developing"
        }
    except Exception as e:
        print(f"[Session summarize error]: {e}")
        summary = {
            "summary": "No summary available due to error.",
            "topics": [],
            "trust_level": "unknown"
        }

    from datetime import datetime
    entry = {
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "summary": summary.get("summary", ""),
        "topics": summary.get("topics", []),
        "trust_level": summary.get("trust_level", "unknown"),
    }
    append_long_term_memory(entry)
    return entry

# Optionally:
def summarize_session_file(user_id: str, session_file_path: str):
    """Helper: Summarize directly from a saved session file."""
    session_file = Path(session_file_path)
    if not session_file.exists():
        raise FileNotFoundError(f"Session file {session_file_path} not found")
    session_data = json.loads(session_file.read_text(encoding="utf-8"))
    return summarize_session(user_id, session_data)

__all__ = ["summarize_session"]