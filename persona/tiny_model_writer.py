# persona/tiny_model_writer.py
# 📊 Analyzes the latest user message for emotion, tone, toxicity, and NSFW content — then appends the result to persona/tiny_model_state.json
import json
from datetime import datetime
from pathlib import Path

TINY_MODEL_JSON = Path("persona/tiny_model_state.json")

def mock_emotion_detection(user_input: str) -> str:
    if "love" in user_input.lower() or "adorable" in user_input.lower():
        return "affectionate"
    elif "hate" in user_input.lower():
        return "angry"
    return "neutral"

def mock_toxicity_score(user_input: str) -> float:
    if "kill" in user_input.lower():
        return 0.9
    return 0.01

def mock_nsfw_flag(user_input: str) -> bool:
    nsfw_keywords = ["nude", "sex", "horny"]
    return any(word in user_input.lower() for word in nsfw_keywords)

def update_tiny_model_state_from_session(session_id: str):
    from utils.session_id import get_or_create_session_file
    session_file = get_or_create_session_file({"session_id": session_id})

    if not Path(session_file).exists():
        print(f"[Tiny Model]: No session file found for ID {session_id}")
        return

    with open(session_file, encoding="utf-8") as f:
        session_data = json.load(f)

    if not session_data:
        print(f"[Tiny Model]: No turns to process")
        return

    last_turn = session_data[-1]
    user_input = last_turn.get("user", "")

    result = {
        "timestamp": datetime.now().isoformat(),
        "turn": {
            "user": user_input,
            "assistant": "Prompt generated"
        },
        "analysis": {
            "emotion": mock_emotion_detection(user_input),
            "toxicity_score": mock_toxicity_score(user_input),
            "nsfw_flag": mock_nsfw_flag(user_input),
            "user_tone": "curious",
            "mood": "playful"
        }
    }

    try:
        TINY_MODEL_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(TINY_MODEL_JSON, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        print(f"[Tiny Model]: ✅ Appended analysis to {TINY_MODEL_JSON}")
    except Exception as e:
        print(f"[Tiny Model Error]: {e}")
