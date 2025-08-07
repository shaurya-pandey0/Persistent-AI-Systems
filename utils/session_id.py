# utils/session_id.py gives unique session id to each session
import uuid
import json
from pathlib import Path
import threading

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# File locks to prevent race conditions
_file_locks = {}
_locks_lock = threading.Lock()

def _get_file_lock(file_path):
    """Get or create a lock for a specific file."""
    with _locks_lock:
        if file_path not in _file_locks:
            _file_locks[file_path] = threading.Lock()
        return _file_locks[file_path]

def get_or_create_session_file(st_session_state):
    if "session_id" not in st_session_state:
        st_session_state["session_id"] = str(uuid.uuid4())
    session_file = DATA_DIR / f"session_{st_session_state['session_id']}.json"
    if not session_file.exists():
        session_file.write_text("[]", encoding="utf-8")   # Start with empty list
    return session_file

def save_turn_to_session(turn: dict, st_session_state=None):
    """Append one turn (dict with 'user','assistant') to the current session file."""
    session_file = get_or_create_session_file(st_session_state)

    # Use file-specific lock to prevent race conditions
    file_lock = _get_file_lock(str(session_file))

    with file_lock:
        session_data = []
        try:
            if session_file.exists():
                session_data = json.loads(session_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[Session file read error]: {e}")
            # Create backup of corrupted file
            backup_file = session_file.with_suffix('.json.backup')
            try:
                backup_file.write_text(session_file.read_text(encoding="utf-8"))
                print(f"Backup created: {backup_file}")
            except:
                pass
            session_data = []

        session_data.append(turn)

        try:
            session_file.write_text(json.dumps(session_data, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[Session file write error]: {e}")
__all__ = [
    "get_or_create_session_file",
    "save_turn_to_session"
]
