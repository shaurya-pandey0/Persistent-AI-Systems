# update_tiny_model_state.py - CORRECTED VERSION

"""Enhancements
===============
1. Eliminated fragile utils.session_id dependency by local helper.
2. Added fallback persistent ID in persona/id.txt for CLI usage.
3. Better mock sentiment/NSFW logic with regex & safer thresholds.
4. JSONL output capped at 5000 lines to avoid file bloat.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict

PERSONA_DIR = Path("persona")
PERSONA_DIR.mkdir(exist_ok=True)
TINY_MODEL_JSON = PERSONA_DIR / "tiny_model_state.json"

# ---------------- Session Helpers ---------------- #

SESSION_DIR = Path("data")
SESSION_DIR.mkdir(exist_ok=True)

ID_FILE = PERSONA_DIR / "id.txt"
if not ID_FILE.exists():
    ID_FILE.write_text(str(datetime.utcnow().timestamp()))
SESSION_ID = ID_FILE.read_text().strip()
SESSION_FILE = SESSION_DIR / f"session_{SESSION_ID}.json"

if not SESSION_FILE.exists():
    SESSION_FILE.write_text("[]", encoding="utf-8")

# ---------------- Mock Analyses ---------------- #
affection_re = re.compile(r"\b(love|adorable|cute|sweetheart)\b", re.I)
anger_re = re.compile(r"\b(hate|angry|furious)\b", re.I)
nsfw_re = re.compile(r"\b(nude|sex|horny|explicit)\b", re.I)
kill_re = re.compile(r"\b(kill|murder|die)\b", re.I)


def mock_emotion(user_input: str) -> str:
    if affection_re.search(user_input):
        return "affectionate"
    if anger_re.search(user_input):
        return "angry"
    return "neutral"


def mock_toxicity(user_input: str) -> float:
    return 0.9 if kill_re.search(user_input) else 0.05


def mock_nsfw(user_input: str) -> bool:
    return bool(nsfw_re.search(user_input))

# ---------------- Write State ---------------- #

def write_state(turn: Dict):
    # Truncate file when huge
    if TINY_MODEL_JSON.exists():
        lines = TINY_MODEL_JSON.read_text(encoding="utf-8").splitlines()
        if len(lines) > 5000:
            TINY_MODEL_JSON.write_text("\n".join(lines[-4000:]), encoding="utf-8")
    with TINY_MODEL_JSON.open("a", encoding="utf-8") as f:
        f.write(json.dumps(turn) + "\n")

# ---------------- Main ------------------------ #


def main():
    try:
        turns = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        user_input = turns[-1]["user"] if turns else ""
    except Exception as exc:
        print(f"❌ Cannot load session: {exc}")
        return

    turn = {
        "timestamp": datetime.utcnow().isoformat(),
        "turn": {"user": user_input, "assistant": "_placeholder_"},
        "analysis": {
            "emotion": mock_emotion(user_input),
            "toxicity_score": mock_toxicity(user_input),
            "nsfw_flag": mock_nsfw(user_input),
        },
    }
    write_state(turn)
    print(f"✅ Tiny model updated ({TINY_MODEL_JSON})")

if __name__ == "__main__":
    main()