# persona/hormone_api.py
"""
Neutral hormone API module to break circular imports.
Contains core hormone data access functions shared across modules.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Any, Union

# File paths
import os
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PERSONA_DIR = SCRIPT_DIR  # hormone_api.py is already in persona/
HORMONES_FILE = PERSONA_DIR / "hormones.json" 
MOOD_WEIGHTS_FILE = PERSONA_DIR / "mood_weights.json"

# Ensure directory exists
PERSONA_DIR.mkdir(exist_ok=True)

# Default values
_DEFAULT_HORMONES: Dict[str, float] = {
    "dopamine": 0.5,
    "serotonin": 0.5,
    "cortisol": 0.5,
    "oxytocin": 0.5,
}

_DEFAULT_MOOD_WEIGHTS: Dict[str, Dict[str, float]] = {
    "cheerful": {"dopamine": 0.7, "serotonin": 0.3},
    "anxious": {"cortisol": 0.8, "dopamine": -0.3},
    "affectionate": {"oxytocin": 0.9},
    "depressed": {"serotonin": -0.5, "cortisol": 0.6},
    "excited": {"dopamine": 0.8, "cortisol": 0.2},
    "melancholic": {"serotonin": -0.4, "dopamine": -0.2},
    "energetic": {"dopamine": 0.6, "serotonin": 0.4},
    "contemplative": {"serotonin": 0.3, "cortisol": -0.2},
    "restless": {"cortisol": 0.5, "dopamine": 0.3},
    "serene": {"serotonin": 0.8, "cortisol": -0.4},
    "neutral": {"dopamine": 0.5, "serotonin": 0.5, "cortisol": 0.5, "oxytocin": 0.5}
}

def _ensure_file(path: Path, default: dict):
    """Ensure file exists with default content if not present."""
    if not path.exists():
        path.write_text(json.dumps(default, indent=2), encoding="utf-8")

def load_hormone_levels() -> Dict[str, float]:
    """Load current hormone levels from file."""
    _ensure_file(HORMONES_FILE, _DEFAULT_HORMONES)
    try:
        return json.loads(HORMONES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return _DEFAULT_HORMONES.copy()

def save_hormone_levels(levels: Dict[str, float]):
    """Save hormone levels to file."""
    try:
        HORMONES_FILE.write_text(json.dumps(levels, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[Hormone API Error]: Failed to save hormone levels: {e}")

def load_mood_weights() -> Dict:
    """Load mood weight mappings from file."""
    _ensure_file(MOOD_WEIGHTS_FILE, _DEFAULT_MOOD_WEIGHTS)
    try:
        return json.loads(MOOD_WEIGHTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return _DEFAULT_MOOD_WEIGHTS.copy()
        

        
def infer_mood_from_hormones(hormone_levels: Dict[str, float], mood_weights: Dict) -> tuple:
    """
    Infer mood from current hormone levels using mood weights.
    Returns (mood_name, intensity)
    """
    scores = {}
    for mood, weights in mood_weights.items():
        score = 0.0
        for hormone, weight in weights.items():
            if hormone in hormone_levels:
                score += hormone_levels[hormone] * weight
            else:
                score += 0.5 * weight  # Default neutral level
        scores[mood] = score
    
    # Find highest scoring mood
    if not scores:
        return "neutral", 0.5
        
    sorted_moods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_mood, top_score = sorted_moods[0]
    
    # Calculate intensity based on score magnitude
    intensity = min(1.0, max(0.0, abs(top_score)))
    
    return top_mood, intensity

def get_mood_context(mood: str, intensity: float) -> Dict:
    """
    Generate mood context information including hybrid/emergent state detection.
    """
    hormone_levels = load_hormone_levels()
    mood_weights = load_mood_weights()
    
    # Calculate all mood scores
    all_scores = {}
    for mood_name, weights in mood_weights.items():
        score = sum(hormone_levels.get(h, 0.5) * w for h, w in weights.items())
        all_scores[mood_name] = score
    
    # Sort by score
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Detect hybrid states (top 2 scores are close)
    is_hybrid = False
    is_emergent = False
    
    if len(sorted_scores) >= 2:
        top_score = sorted_scores[0][1]
        second_score = sorted_scores[1][1]
        score_diff = abs(top_score - second_score)
        
        # Hybrid if top 2 scores are very close
        if score_diff < 0.3:
            is_hybrid = True
    
    # Detect emergent states (unusual hormone combinations)
    hormone_variance = sum((v - 0.5) ** 2 for v in hormone_levels.values()) / len(hormone_levels)
    if hormone_variance > 0.2:  # High variance indicates unusual combination
        is_emergent = True
    
    # Determine stability
    stability = "high"
    if intensity > 0.8:
        stability = "low"  # High intensity = less stable
    elif intensity > 0.6:
        stability = "medium"
    
    return {
        "is_hybrid": is_hybrid,
        "is_emergent": is_emergent,
        "stability": stability,
        "hormone_variance": hormone_variance,
        "all_mood_scores": all_scores
    }
