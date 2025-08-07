# persona/relationship_status.py
# Track relationship status with users

import json
from pathlib import Path
from typing import Dict, Any

RELATIONSHIP_FILE = Path("data/relationships.json")

def load_relationships() -> Dict[str, Any]:
    """Load relationship data for all users."""
    if not RELATIONSHIP_FILE.exists():
        return {}
    try:
        with open(RELATIONSHIP_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Relationship load error]: {e}")
        return {}

def save_relationships(relationships: Dict[str, Any]):
    """Save relationship data."""
    try:
        RELATIONSHIP_FILE.parent.mkdir(exist_ok=True)
        with open(RELATIONSHIP_FILE, "w", encoding="utf-8") as f:
            json.dump(relationships, f, indent=2)
    except Exception as e:
        print(f"[Relationship save error]: {e}")

def get_user_relationship(user_id: str) -> Dict[str, Any]:
    """Get relationship status for a specific user."""
    relationships = load_relationships()
    return relationships.get(user_id, {
        "trust_level": "new",
        "interaction_count": 0,
        "relationship_type": "acquaintance"
    })

def update_user_relationship(user_id: str, updates: Dict[str, Any]):
    """Update relationship status for a user."""
    relationships = load_relationships()
    if user_id not in relationships:
        relationships[user_id] = get_user_relationship(user_id)

    relationships[user_id].update(updates)
    save_relationships(relationships)

def get_relationship_summary(user_id):
    """
    Placeholder for relationship intelligence.
    Should return a string describing the relationship status for UI prompt assembly.
    """
    # In a future version, you might load user_id-specific summaries from disk or DB
    return (
        "- This user enjoys flirty banter and quirky science facts.\n"
        "- Trust level: Medium\n"
        "- Past interactions suggest light teasing is well-received."
    )
