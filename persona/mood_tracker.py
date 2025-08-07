# persona/mood_tracker.py

"""
Enhanced mood tracking system with Phase 4 compound mood modeling.

Phase 1: Basic mood calculation (WORKING) ✅
Phase 2: Enhanced hormone integration (WORKING) ✅  
Phase 3: Cross-session persistence, personality integration (WORKING) ✅
Phase 4: Compound mood vectors, transition modeling, prediction (NEW) 🚀
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import statistics
import time
import math
from math import isnan

from persona.hormone_api import (
    load_hormone_levels, save_hormone_levels,
    infer_mood_from_hormones, get_mood_context
)

# Import Phase 4 configuration
try:
    from config.constants import MOOD_PERSISTENCE_CONFIG, COMPOUND_MOOD_CONFIG
except ImportError:
    # Fallback configuration
    MOOD_PERSISTENCE_CONFIG = {
        "enable_cross_session_mood": True,
        "mood_decay_hours": 24,
        "mood_decay_rate": 0.1,
        "session_break_threshold": 4,
        "enable_personality_integration": True,
        "personality_influence_strength": 0.3,
        "max_mood_history": 1000,
        "mood_pattern_analysis": True,
        "mood_trend_window": 24,
        "enable_mood_interpolation": True,
        "mood_stability_factor": 0.8,
        "complex_emotion_detection": True,
    }
    
    COMPOUND_MOOD_CONFIG = {
        "enable_vector_moods": True,
        "mood_vector_dimensions": ["valence", "arousal", "dominance", "sociability", "creativity", "focus"],
        "enable_mood_transitions": True,
        "transition_smoothing_factor": 0.7,
        "max_transition_speed": 0.3,
        "transition_history_length": 100,
        "enable_mixed_moods": True,
        "mood_conflict_threshold": 0.4,
        "dominant_mood_threshold": 0.6,
        "mood_stability_tracking": True,
        "enable_mood_prediction": True,
        "prediction_horizon_hours": 6,
        "mood_pattern_learning": True,
        "pattern_learning_rate": 0.05,
    }

# File paths
import os
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PERSONA_DIR = SCRIPT_DIR  # mood_tracker.py is already in persona/
MOOD_HISTORY_FILE = PERSONA_DIR / "mood_history.json"
PERSONALITY_FILE = PERSONA_DIR / "personality.json"
MOOD_SESSION_FILE = PERSONA_DIR / "mood_session_state.json"
MOOD_VECTOR_FILE = PERSONA_DIR / "mood_vectors.json"
MOOD_TRANSITIONS_FILE = PERSONA_DIR / "mood_transitions.jsonl"

# ─────────────────── 🔄 Patched formulas & helpers ─────────────
def _smooth_intensity(mood_vector: Dict[str, float],
                      user_text: str = "",
                      smoothing: float = 0.7) -> float:
    """
    Compute average deviation from neutral (0.5) and dampen micro-fluctuations.
    • Round to 2 decimals max.
    • Cap reactivity for very short inputs (< 6 chars) to avoid overfitting.
    """
    raw = sum(abs(v - 0.5) for v in mood_vector.values()) / len(mood_vector)
    if raw < 0.10:
        smoothed = raw * 0.5
    elif raw < 0.20:
        smoothed = raw * smoothing
    else:
        smoothed = raw

    # Context-based ceiling
    length = len(user_text.strip())
    if length <= 5:
        smoothed = min(smoothed, 0.30)
    elif length <= 20:
        smoothed = min(smoothed, 0.50)

    # Guard against NaN or out-of-range
    if isnan(smoothed):
        smoothed = 0.0
    return round(max(0.0, min(1.0, smoothed)), 2)

# ─────────────────── 🎭 Phase 4: Compound Mood System ─────────────

class CompoundMoodCalculator:
    """
    Phase 4: Advanced compound mood calculation with multi-dimensional
    mood vectors, transition modeling, and predictive capabilities.
    """
    
    def __init__(self):
        self.config = COMPOUND_MOOD_CONFIG
        self.dimensions = self.config["mood_vector_dimensions"]
        self.transition_history = self._load_transition_history()
        self.mood_patterns = {}
    
    def _load_transition_history(self) -> List[Dict]:
        """Load mood transition history."""
        if not MOOD_TRANSITIONS_FILE.exists():
            return []
        
        try:
            history = []
            with MOOD_TRANSITIONS_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        history.append(json.loads(line))
            return history[-self.config["transition_history_length"]:]
        except Exception as e:
            print(f"[Transition History Error]: {e}")
            return []
    
    def _save_transition(self, transition: Dict) -> None:
        """Save mood transition to history."""
        try:
            MOOD_TRANSITIONS_FILE.parent.mkdir(exist_ok=True)
            with MOOD_TRANSITIONS_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(transition) + "\n")
        except Exception as e:
            print(f"[Transition Save Error]: {e}")
    
    def calculate_mood_vector(self, hormone_levels: Dict[str, float]) -> Dict[str, float]:
        """
        Phase 4: Calculate multi-dimensional mood vector from hormone levels.
        Returns mood vector with values for each dimension (valence, arousal, etc.)
        """
        mood_vector = {}
        
        # Extract hormone values
        dopamine = hormone_levels.get("dopamine", 0.5)
        serotonin = hormone_levels.get("serotonin", 0.5)
        cortisol = hormone_levels.get("cortisol", 0.5)
        oxytocin = hormone_levels.get("oxytocin", 0.5)
        
        # ─────────────────── 🧮 REPLACED mood-vector formulas - START ───────────────────
        for dimension in self.dimensions:
            if dimension == "valence":             # happiness – sadness
                mood_vector[dimension] = (dopamine * 0.40 + serotonin * 0.60) - (cortisol * 0.30)
            elif dimension == "arousal":           # energy – calm
                # Cortisol drives alertness more strongly; serotonin mildly dampens.
                mood_vector[dimension] = (dopamine * 0.40 + cortisol * 0.50) - (serotonin * 0.10)
            elif dimension == "dominance":         # control – submission
                # Trim baseline offset to avoid chronic inflation.
                mood_vector[dimension] = dopamine * 0.50 - cortisol * 0.30 + 0.40
            elif dimension == "sociability":       # social – solitary
                mood_vector[dimension] = (oxytocin * 0.70 + serotonin * 0.30) - (cortisol * 0.20)
            elif dimension == "creativity":        # creative – analytical
                # Directly track dopamine; penalise stress.
                mood_vector[dimension] = dopamine * 0.80 - cortisol * 0.20 + 0.10
            elif dimension == "focus":             # focused – scattered
                # Peak focus at moderate arousal (~0.6); high cortisol penalises.
                moderate_bonus = 1.0 - abs(((dopamine + cortisol * 0.5) / 1.5) - 0.60) * 2
                stress_penalty = cortisol * 0.40
                mood_vector[dimension] = moderate_bonus * 0.60 + dopamine * 0.30 - stress_penalty + 0.30

            mood_vector[dimension] = max(0.0, min(1.0, mood_vector[dimension]))
        # ─────────────────── 🧮 REPLACED mood-vector formulas - END ─────────────────────
        
        return mood_vector

    def detect_mixed_moods(self, mood_vector: Dict[str, float]) -> Dict[str, Any]:
        """Phase 4: Detect mixed or conflicting mood states."""
        if not self.config["enable_mixed_moods"]:
            return {"mixed_mood": False}
        
        # Calculate mood conflicts
        conflicts = []
        conflict_threshold = self.config["mood_conflict_threshold"]
        
        # Check for specific conflicts
        valence = mood_vector.get("valence", 0.5)
        arousal = mood_vector.get("arousal", 0.5)
        dominance = mood_vector.get("dominance", 0.5)
        
        # High arousal + negative valence = anxiety/anger
        if arousal > 0.7 and valence < 0.3:
            conflicts.append({"type": "anxious", "intensity": (arousal - valence) / 2})
        
        # Low arousal + positive valence = contentment
        if arousal < 0.3 and valence > 0.7:
            conflicts.append({"type": "content", "intensity": (valence - arousal) / 2})
        
        # High dominance + high arousal = excitement/aggression
        if dominance > 0.7 and arousal > 0.7:
            conflicts.append({"type": "intense", "intensity": (dominance + arousal) / 2})
        
        # Calculate overall conflict level
        overall_conflict = 0.0
        if conflicts:
            overall_conflict = max(c["intensity"] for c in conflicts)
        
        return {
            "mixed_mood": overall_conflict > conflict_threshold,
            "conflicts": conflicts,
            "conflict_level": overall_conflict,
            "dominant_emotions": [c["type"] for c in conflicts if c["intensity"] > self.config["dominant_mood_threshold"]]
        }

    def calculate_mood_transitions(self,
                                 current_vector: Dict[str, float],
                                 previous_vector: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Phase 4: Calculate mood transitions and apply smoothing."""
        if not self.config["enable_mood_transitions"] or not previous_vector:
            return {
                "transition_occurred": False,
                "current_vector": current_vector,
                "transition_speed": 0.0
            }
        
        # Calculate transition vector (change in each dimension)
        transition_vector = {}
        total_change = 0.0
        
        for dimension in self.dimensions:
            change = current_vector[dimension] - previous_vector.get(dimension, 0.5)
            transition_vector[dimension] = change
            total_change += abs(change)
        
        # Calculate transition speed
        transition_speed = total_change / len(self.dimensions)
        max_speed = self.config["max_transition_speed"]
        
        # Apply transition smoothing if change is too rapid
        smoothed_vector = current_vector.copy()
        if transition_speed > max_speed:
            smoothing_factor = self.config["transition_smoothing_factor"]
            for dimension in self.dimensions:
                # Blend current and previous values
                smoothed_vector[dimension] = (
                    previous_vector.get(dimension, 0.5) * smoothing_factor +
                    current_vector[dimension] * (1 - smoothing_factor)
                )
        
        # Record transition
        transition_data = {
            "timestamp": datetime.now().isoformat(),
            "from_vector": previous_vector,
            "to_vector": current_vector,
            "smoothed_vector": smoothed_vector,
            "transition_speed": transition_speed,
            "was_smoothed": transition_speed > max_speed
        }
        
        self.transition_history.append(transition_data)
        self._save_transition(transition_data)
        
        # Keep history manageable
        if len(self.transition_history) > self.config["transition_history_length"]:
            self.transition_history = self.transition_history[-self.config["transition_history_length"]:]
        
        return {
            "transition_occurred": True,
            "current_vector": smoothed_vector,
            "transition_speed": transition_speed,
            "transition_vector": transition_vector,
            "was_smoothed": transition_speed > max_speed
        }

    def predict_mood_evolution(self, current_vector: Dict[str, float]) -> Dict[str, Any]:
        """Phase 4: Predict future mood states based on patterns."""
        if not self.config["enable_mood_prediction"]:
            return {"prediction_available": False}
        
        if len(self.transition_history) < 5:
            return {"prediction_available": False, "reason": "insufficient_history"}
        
        # Analyze recent transitions to predict trends
        recent_transitions = self.transition_history[-10:]
        
        # Calculate average transition vectors
        avg_transitions = {}
        for dimension in self.dimensions:
            dimension_changes = [
                t["to_vector"][dimension] - t["from_vector"].get(dimension, 0.5)
                for t in recent_transitions
                if dimension in t["to_vector"] and dimension in t.get("from_vector", {})
            ]
            
            if dimension_changes:
                avg_transitions[dimension] = statistics.mean(dimension_changes)
            else:
                avg_transitions[dimension] = 0.0
        
        # Predict future state
        prediction_hours = self.config["prediction_horizon_hours"]
        predicted_vector = {}
        
        for dimension in self.dimensions:
            # Apply trend with decay
            trend = avg_transitions[dimension]
            decay_factor = 0.8 ** (prediction_hours / 6)  # Decay over time
            predicted_value = current_vector[dimension] + (trend * prediction_hours * decay_factor)
            predicted_vector[dimension] = max(0.0, min(1.0, predicted_value))
        
        # Calculate confidence based on trend consistency
        trend_consistency = self._calculate_trend_consistency(recent_transitions)
        
        return {
            "prediction_available": True,
            "predicted_vector": predicted_vector,
            "prediction_horizon_hours": prediction_hours,
            "confidence": trend_consistency,
            "trend_analysis": avg_transitions
        }

    def _calculate_trend_consistency(self, transitions: List[Dict]) -> float:
        """Calculate how consistent recent mood transitions have been."""
        if len(transitions) < 3:
            return 0.0
        
        # Calculate variance in transition directions
        dimension_variances = []
        for dimension in self.dimensions:
            changes = []
            for t in transitions:
                if dimension in t["to_vector"] and dimension in t.get("from_vector", {}):
                    change = t["to_vector"][dimension] - t["from_vector"].get(dimension, 0.5)
                    changes.append(change)
            
            if len(changes) > 1:
                variance = statistics.variance(changes)
                dimension_variances.append(variance)
        
        if not dimension_variances:
            return 0.0
        
        # Lower variance = higher consistency
        avg_variance = statistics.mean(dimension_variances)
        consistency = 1.0 / (1.0 + avg_variance * 10)  # Scale and invert
        return max(0.0, min(1.0, consistency))

# Create global compound mood calculator
_compound_mood_calc = CompoundMoodCalculator()

# ─────────────────── 🏗️ Phase 3+4: Enhanced Mood Tracking ─────────────

class EnhancedMoodTracker:
    """
    Phase 3+4: Advanced mood tracking with cross-session persistence,
    personality integration, and compound mood modeling.
    """
    
    def __init__(self):
        self.config = MOOD_PERSISTENCE_CONFIG
        self.compound_config = COMPOUND_MOOD_CONFIG
        self.personality = self._load_personality()
        self.session_state = self._load_session_state()
        self.mood_patterns = {}
    
    def _load_personality(self) -> Dict[str, Any]:
        """Load personality configuration with Phase 3+4 enhancements."""
        try:
            if PERSONALITY_FILE.exists():
                with open(PERSONALITY_FILE, "r", encoding="utf-8") as f:
                    personality = json.load(f)
                
                # Ensure Phase 3 personality structure
                if "personality_traits" not in personality:
                    personality["personality_traits"] = {
                        "tone": personality.get("tone", "friendly"),
                        "style": personality.get("style", "helpful"),
                        "temperament": personality.get("temperament", "balanced"),
                        "formality": personality.get("formality", "moderate")
                    }
                
                # Set default mood tendencies if not present
                if "mood_tendencies" not in personality:
                    personality["mood_tendencies"] = {
                        "baseline_dopamine": 0.5,
                        "baseline_serotonin": 0.5,
                        "baseline_cortisol": 0.5,
                        "baseline_oxytocin": 0.5,
                        "emotional_volatility": 0.5,
                        "mood_recovery_rate": 0.7
                    }
                
                return personality
            else:
                return self._create_default_personality()
        except Exception as e:
            print(f"[Personality Load Error]: {e}")
            return self._create_default_personality()
    
    def _create_default_personality(self) -> Dict[str, Any]:
        """Create default personality structure."""
        return {
            "name": "Isabella",
            "personality_traits": {
                "tone": "witty",
                "style": "conversational",
                "temperament": "playful",
                "formality": "casual"
            },
            "mood_tendencies": {
                "baseline_dopamine": 0.5,
                "baseline_serotonin": 0.5,
                "baseline_cortisol": 0.5,
                "baseline_oxytocin": 0.5,
                "emotional_volatility": 0.5,
                "mood_recovery_rate": 0.7
            }
        }
    
    def _load_session_state(self) -> Dict[str, Any]:
        """Phase 3: Load cross-session mood state."""
        if not self.config["enable_cross_session_mood"]:
            return self._create_default_session_state()
        
        try:
            if MOOD_SESSION_FILE.exists():
                with open(MOOD_SESSION_FILE, "r", encoding="utf-8") as f:
                    state = json.load(f)
                
                # Check if session is still valid
                if self._is_session_valid(state):
                    return state
                else:
                    print("[Session State]: Previous session expired, starting fresh")
                    return self._create_default_session_state()
            else:
                return self._create_default_session_state()
        except Exception as e:
            print(f"[Session State Load Error]: {e}")
            return self._create_default_session_state()
    
    def _create_default_session_state(self) -> Dict[str, Any]:
        """Create default session state."""
        return {
            "session_id": int(time.time()),
            "session_start": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "current_mood": "neutral",
            "mood_intensity": 0.5,
            "session_mood_changes": 0,
            "mood_stability_score": 1.0,
            "personality_adaptations": {},
            # Phase 4: Compound mood additions
            "current_mood_vector": {dim: 0.5 for dim in _compound_mood_calc.dimensions},
            "vector_mood_history": []
        }
    
    def _is_session_valid(self, state: Dict[str, Any]) -> bool:
        """Check if session state is still valid."""
        try:
            last_activity = datetime.fromisoformat(state["last_activity"])
            hours_elapsed = (datetime.now() - last_activity).total_seconds() / 3600
            return hours_elapsed < self.config["session_break_threshold"]
        except:
            return False
    
    def _save_session_state(self) -> None:
        """Save current session state."""
        if not self.config["enable_cross_session_mood"]:
            return
        
        try:
            self.session_state["last_activity"] = datetime.now().isoformat()
            MOOD_SESSION_FILE.parent.mkdir(exist_ok=True)
            with open(MOOD_SESSION_FILE, "w", encoding="utf-8") as f:
                json.dump(self.session_state, f, indent=2)
        except Exception as e:
            print(f"[Session State Save Error]: {e}")

# Create global instance
_enhanced_tracker = EnhancedMoodTracker()

# ─────────────────── 🧠 Enhanced mood calculation (Phase 1+2+3+4) ─────

def _calculate_compound_mood(hormone_levels: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Phase 4: Calculate compound mood vector and analysis."""
    # Calculate mood vector
    mood_vector = _compound_mood_calc.calculate_mood_vector(hormone_levels)
    
    # Get previous mood vector for transition analysis
    previous_vector = _enhanced_tracker.session_state.get("current_mood_vector")
    
    # Calculate transitions
    transition_info = _compound_mood_calc.calculate_mood_transitions(mood_vector, previous_vector)
    
    # Use smoothed vector if transition was smoothed
    final_vector = transition_info["current_vector"]
    
    # Detect mixed moods
    mixed_mood_info = _compound_mood_calc.detect_mixed_moods(final_vector)
    
    # Predict future mood evolution
    prediction_info = _compound_mood_calc.predict_mood_evolution(final_vector)
    
    # Create comprehensive mood analysis
    mood_analysis = {
        "mood_vector": final_vector,
        "hormone_levels": hormone_levels,
        "mixed_moods": mixed_mood_info,
        "transition": transition_info,
        "prediction": prediction_info,
        "timestamp": datetime.now().isoformat(),
        "session_id": _enhanced_tracker.session_state["session_id"]
    }
    
    # Update session state
    _enhanced_tracker.session_state["current_mood_vector"] = final_vector
    _enhanced_tracker.session_state["vector_mood_history"].append({
        "timestamp": datetime.now().isoformat(),
        "vector": final_vector,
        "analysis": mood_analysis
    })
    
    # Keep history manageable
    if len(_enhanced_tracker.session_state["vector_mood_history"]) > 50:
        _enhanced_tracker.session_state["vector_mood_history"] = _enhanced_tracker.session_state["vector_mood_history"][-25:]
    
    print(f"[Mood Calculator]: Compound mood calculated - vector dimensions: {len(final_vector)}")
    return final_vector, mood_analysis

def _calculate_scalar_mood(hormone_levels: Dict[str, float]) -> Tuple[str, float]:
    """Legacy scalar mood calculation for backward compatibility."""
    # Calculate deviations from personality-adjusted baseline
    baseline = _get_personality_baselines()
    deviations = {
        hormone: level - baseline[hormone]
        for hormone, level in hormone_levels.items()
    }
    
    print(f"[Mood Calculator]: Deviations - " +
          ", ".join(f"{h}:{d:+.3f}" for h, d in deviations.items()))
    
    # Enhanced mood determination with Phase 3 features
    mood_scores = _calculate_enhanced_mood_scores(deviations, hormone_levels)
    print(f"[Mood Calculator]: Mood scores calculated - {mood_scores}")
    
    # Apply mood interpolation and stability factors
    if _enhanced_tracker.config["enable_mood_interpolation"]:
        mood_scores = _apply_mood_interpolation(mood_scores)
    
    # Select best mood
    if not mood_scores:
        print("[Mood Calculator]: No strong patterns detected, defaulting to neutral")
        return "neutral", 0.5
    
    top_mood = max(mood_scores.items(), key=lambda x: x[1])
    mood_name = top_mood[0]
    raw_intensity = top_mood[1]
    
    # Apply personality-based intensity adjustments
    intensity = _apply_personality_intensity_adjustments(mood_name, raw_intensity)
    
    print(f"[Mood Calculator]: Selected mood '{mood_name}' with intensity {intensity:.3f}")
    return mood_name, intensity

def _apply_personality_baseline_adjustments(hormone_levels: Dict[str, float]) -> Dict[str, float]:
    """Phase 3: Apply personality-based hormone baseline adjustments."""
    if not _enhanced_tracker.config["enable_personality_integration"]:
        return hormone_levels
    
    personality = _enhanced_tracker.personality
    tendencies = personality.get("mood_tendencies", {})
    adjusted_levels = hormone_levels.copy()
    influence_strength = _enhanced_tracker.config["personality_influence_strength"]
    
    # Apply personality baseline influences
    for hormone in adjusted_levels:
        baseline_key = f"baseline_{hormone}"
        if baseline_key in tendencies:
            personality_baseline = tendencies[baseline_key]
            current_level = adjusted_levels[hormone]
            
            # Blend current level with personality baseline
            adjusted_levels[hormone] = (
                current_level * (1 - influence_strength) +
                personality_baseline * influence_strength
            )
    
    return adjusted_levels

def _get_personality_baselines() -> Dict[str, float]:
    """Get personality-adjusted hormone baselines."""
    if not _enhanced_tracker.config["enable_personality_integration"]:
        return {"dopamine": 0.5, "serotonin": 0.5, "cortisol": 0.5, "oxytocin": 0.5}
    
    tendencies = _enhanced_tracker.personality.get("mood_tendencies", {})
    return {
        "dopamine": tendencies.get("baseline_dopamine", 0.5),
        "serotonin": tendencies.get("baseline_serotonin", 0.5),
        "cortisol": tendencies.get("baseline_cortisol", 0.5),
        "oxytocin": tendencies.get("baseline_oxytocin", 0.5),
    }

def calculate_mood_from_hormones(hormone_levels: Dict[str, float]) -> Union[Tuple[str, float], Tuple[Dict[str, float], Dict[str, Any]]]:
    """
    Phase 4: Enhanced mood calculation with compound mood vectors
    and personality integration.
    
    Returns either (mood_name, intensity) for legacy compatibility
    or (mood_vector, mood_analysis) for Phase 4 features.
    """
    print(f"[Mood Calculator]: Processing hormones {hormone_levels}")
    
    # Apply personality-based baseline adjustments
    if _enhanced_tracker.config["enable_personality_integration"]:
        hormone_levels = _apply_personality_baseline_adjustments(hormone_levels)
    
    # Phase 4: Calculate compound mood vector if enabled
    if _enhanced_tracker.compound_config["enable_vector_moods"]:
        return _calculate_compound_mood(hormone_levels)
    else:
        # Legacy scalar mood calculation
        return _calculate_scalar_mood(hormone_levels)

def _calculate_enhanced_mood_scores(deviations: Dict[str, float], hormone_levels: Dict[str, float]) -> Dict[str, float]:
    """Phase 3: Enhanced mood scoring with complex emotion detection."""
    mood_scores = {}
    
    dopa_dev = deviations["dopamine"]
    sero_dev = deviations["serotonin"]
    cort_dev = deviations["cortisol"]
    oxy_dev = deviations["oxytocin"]
    
    # Enhanced mood patterns with personality traits
    traits = _enhanced_tracker.personality.get("personality_traits", {})
    playful_modifier = 1.2 if traits.get("temperament") == "playful" else 1.0
    witty_modifier = 1.1 if traits.get("tone") == "witty" else 1.0
    
    # High cortisol patterns (stress/anxiety) - reduced if personality is playful
    if cort_dev > 0.1:
        base_stress_factor = 0.8 if traits.get("temperament") == "playful" else 1.0
        if sero_dev < -0.05:
            mood_scores["anxious"] = (abs(cort_dev) + abs(sero_dev) * 0.7) * base_stress_factor
        if dopa_dev < -0.05:
            mood_scores["restless"] = (abs(cort_dev) + abs(dopa_dev) * 0.6) * base_stress_factor
        if cort_dev > 0.15:
            mood_scores["stressed"] = abs(cort_dev) * 1.2 * base_stress_factor
    
    # Low serotonin patterns (sadness/depression)
    if sero_dev < -0.08:
        if dopa_dev < -0.05:
            mood_scores["depressed"] = abs(sero_dev) + abs(dopa_dev) * 0.8
        elif cort_dev > 0.05:
            mood_scores["melancholic"] = abs(sero_dev) + abs(cort_dev) * 0.6
        else:
            mood_scores["sad"] = abs(sero_dev) * 1.1
    
    # High dopamine patterns (joy/excitement) - enhanced if personality is playful
    if dopa_dev > 0.08:
        if oxy_dev > 0.05:
            mood_scores["euphoric"] = (dopa_dev + oxy_dev * 0.8) * playful_modifier
        elif sero_dev > 0.05:
            mood_scores["cheerful"] = (dopa_dev + sero_dev * 0.7) * playful_modifier * witty_modifier
        else:
            mood_scores["energetic"] = dopa_dev * 1.2 * playful_modifier
    
    # High oxytocin patterns (love/affection)
    if oxy_dev > 0.1:
        if dopa_dev > 0.05:
            mood_scores["loving"] = oxy_dev + dopa_dev * 0.6
        elif sero_dev > 0.03:
            mood_scores["affectionate"] = oxy_dev + sero_dev * 0.8
        else:
            mood_scores["caring"] = oxy_dev * 1.1
    
    # Balanced positive states - enhanced by conversational style
    if dopa_dev > 0.03 and sero_dev > 0.03 and cort_dev < 0.1:
        conversational_modifier = 1.1 if traits.get("style") == "conversational" else 1.0
        mood_scores["content"] = (dopa_dev + sero_dev) * 0.8 * conversational_modifier
    
    # Phase 3: Complex emotion detection
    if _enhanced_tracker.config["complex_emotion_detection"]:
        # Mixed emotional states
        if abs(dopa_dev) > 0.05 and abs(sero_dev) > 0.05 and abs(cort_dev) > 0.05:
            mood_scores["conflicted"] = (abs(dopa_dev) + abs(sero_dev) + abs(cort_dev)) * 0.4
        
        # Anticipatory excitement
        if dopa_dev > 0.06 and cort_dev > 0.03 and cort_dev < 0.1:
            mood_scores["anticipatory"] = (dopa_dev + cort_dev * 0.5) * playful_modifier
        
        # Contemplative state
        if abs(dopa_dev) < 0.03 and abs(sero_dev) < 0.03 and abs(cort_dev) < 0.05 and oxy_dev > 0.02:
            mood_scores["contemplative"] = oxy_dev * 0.8
    
    return mood_scores

def _apply_mood_interpolation(mood_scores: Dict[str, float]) -> Dict[str, float]:
    """Phase 3: Apply mood interpolation for smooth transitions."""
    if not _enhanced_tracker.config["enable_mood_interpolation"]:
        return mood_scores
    
    current_mood = _enhanced_tracker.session_state.get("current_mood", "neutral")
    stability_factor = _enhanced_tracker.config["mood_stability_factor"]
    
    # If current mood is in the scores, boost it slightly for stability
    if current_mood in mood_scores:
        mood_scores[current_mood] *= (1 + stability_factor * 0.1)
    
    return mood_scores

def _apply_personality_intensity_adjustments(mood_name: str, raw_intensity: float) -> float:
    """Phase 3: Apply personality-based intensity adjustments."""
    personality = _enhanced_tracker.personality
    tendencies = personality.get("mood_tendencies", {})
    
    # Get emotional volatility from personality
    volatility = tendencies.get("emotional_volatility", 0.5)
    
    # Adjust intensity based on volatility
    if volatility < 0.3:  # Low volatility - dampen extreme emotions
        intensity = 0.3 + (raw_intensity - 0.3) * 0.7
    elif volatility > 0.7:  # High volatility - amplify emotions
        intensity = min(1.0, raw_intensity * 1.2)
    else:
        intensity = raw_intensity
    
    # Ensure reasonable bounds
    intensity = min(1.0, max(0.1, intensity))
    
    return intensity

# ─────────────────── 🎭 Enhanced mood management (Phase 1+2+3+4) ─────────

def load_mood_history() -> list:
    """Enhanced mood history loading with Phase 3+4 features."""
    if not MOOD_HISTORY_FILE.exists():
        try:
            MOOD_HISTORY_FILE.parent.mkdir(exist_ok=True)
            with open(MOOD_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
            print("[Mood History]: Created default mood history file") 
        except Exception as e:
            print(f"[Mood history init error]: {e}")
            return []
    
    try:
        with open(MOOD_HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        
        # Phase 3: Limit history size
        max_history = _enhanced_tracker.config["max_mood_history"]
        if len(history) > max_history:
            history = history[-max_history:]
            save_mood_history(history)
        
        return history
    except Exception as e:
        print(f"[Mood history load error]: {e}")
        return []

def save_mood_history(history: list):
    """Enhanced mood history saving."""
    try:
        MOOD_HISTORY_FILE.parent.mkdir(exist_ok=True)
        with open(MOOD_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[Mood history save error]: {e}")

def update_mood(new_mood: Union[str, Dict[str, float]], intensity: float = None, reason: str = "", hormone_context: Dict = None):
    """
    Phase 4: Enhanced mood update with compound mood support and
    cross-session persistence.
    """
    # Handle both scalar and vector moods
    if isinstance(new_mood, dict):
        # Phase 4: Vector mood update
        return _update_vector_mood(new_mood, reason, hormone_context)
    else:
        # Legacy scalar mood update
        return _update_scalar_mood(new_mood, intensity, reason, hormone_context)

def _update_vector_mood(mood_vector: Dict[str, float], reason: str = "", hormone_context: Dict = None):
    """Phase 4: Update compound mood vector."""
    # Get mood context if not provided
    if hormone_context is None:
        hormone_context = get_mood_context("vector_mood", 0.7)  # Default intensity for vectors
    
    # ─────────────────── 🔧 Hook smoothed intensity into _update_vector_mood ────────
    # Calculate overall mood intensity from vector using smoothed intensity
    intensity = _smooth_intensity(mood_vector, reason)
    
    # Determine dominant mood characteristics
    dominant_characteristics = []
    for dim, value in mood_vector.items():
        if value > 0.7:
            dominant_characteristics.append(f"high_{dim}")
        elif value < 0.3:
            dominant_characteristics.append(f"low_{dim}")
    
    primary_mood = "_".join(dominant_characteristics[:3]) if dominant_characteristics else "balanced"
    
    # Phase 4: Apply cross-session mood persistence
    if _enhanced_tracker.config["enable_cross_session_mood"]:
        mood_vector, intensity = _apply_cross_session_vector_decay(mood_vector, intensity)
    
    # Update mood_adjustments.json with enhanced vector data
    mood_data = {
        "current_mood": primary_mood,
        "mood_vector": mood_vector,
        "intensity": intensity,
        "context": hormone_context,
        "last_updated": datetime.utcnow().isoformat(),
        "session_id": _enhanced_tracker.session_state["session_id"],
        "personality_version": _enhanced_tracker.personality.get("version", "4.0"),
        "mood_type": "vector"
    }
    
    try:
        with open("persona/mood_adjustments.json", "w", encoding="utf-8") as f:
            json.dump(mood_data, f, indent=2)
    except Exception as e:
        print(f"[Mood update error]: {e}")
        return
    
    # Phase 4: Update session state with vector information
    _enhanced_tracker.session_state["current_mood"] = primary_mood
    _enhanced_tracker.session_state["current_mood_vector"] = mood_vector
    _enhanced_tracker.session_state["mood_intensity"] = intensity
    _enhanced_tracker.session_state["session_mood_changes"] += 1
    _enhanced_tracker._save_session_state()
    
    # Log to history with enhanced vector information
    history = load_mood_history()
    history_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "mood": primary_mood,
        "mood_vector": mood_vector,
        "intensity": intensity,
        "reason": reason,
        "session_id": _enhanced_tracker.session_state["session_id"],
        "is_hybrid": hormone_context.get("is_hybrid", False),
        "is_emergent": hormone_context.get("is_emergent", False),
        "stability": hormone_context.get("stability", "medium"),
        "personality_influence": _enhanced_tracker.config["enable_personality_integration"],
        "mood_type": "vector"
    }
    
    # Add hormone levels snapshot for debugging contextual changes
    if reason and ("contextual" in reason or "hormone_event" in reason):
        history_entry["hormone_snapshot"] = load_hormone_levels()
    
    history.append(history_entry)
    
    # Keep only recent history
    max_history = _enhanced_tracker.config["max_mood_history"]
    if len(history) > max_history:
        history = history[-max_history:]
    
    save_mood_history(history)
    
    # Print mood change for debugging
    dominant_dims = [f"{k}:{v:.2f}" for k, v in mood_vector.items() if abs(v - 0.5) > 0.1]
    print(f"[Mood Update]: {primary_mood} [VECTOR] ({', '.join(dominant_dims)}) - {reason}")

def _update_scalar_mood(new_mood: str, intensity: float, reason: str = "", hormone_context: Dict = None):
    """Legacy scalar mood update with Phase 3+4 enhancements."""
    # Get mood context if not provided
    if hormone_context is None:
        hormone_context = get_mood_context(new_mood, intensity)
    
    # Phase 3: Apply cross-session mood decay if needed
    if _enhanced_tracker.config["enable_cross_session_mood"]:
        new_mood, intensity = _apply_cross_session_mood_decay(new_mood, intensity)
    
    # Update mood_adjustments.json with enhanced data
    mood_data = {
        "current_mood": new_mood,
        "intensity": intensity,
        "context": hormone_context,
        "last_updated": datetime.utcnow().isoformat(),
        "session_id": _enhanced_tracker.session_state["session_id"],
        "personality_version": _enhanced_tracker.personality.get("version", "4.0"),
        "mood_type": "scalar"
    }
    
    try:
        with open("persona/mood_adjustments.json", "w", encoding="utf-8") as f:
            json.dump(mood_data, f, indent=2)
    except Exception as e:
        print(f"[Mood update error]: {e}")
        return
    
    # Phase 3: Update session state
    _enhanced_tracker.session_state["current_mood"] = new_mood
    _enhanced_tracker.session_state["mood_intensity"] = intensity
    _enhanced_tracker.session_state["session_mood_changes"] += 1
    _enhanced_tracker._save_session_state()
    
    # Log to history with enhanced information
    history = load_mood_history()
    history_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "mood": new_mood,
        "intensity": intensity,
        "reason": reason,
        "session_id": _enhanced_tracker.session_state["session_id"],
        "is_hybrid": hormone_context.get("is_hybrid", False),
        "is_emergent": hormone_context.get("is_emergent", False),
        "stability": hormone_context.get("stability", "medium"),
        "personality_influence": _enhanced_tracker.config["enable_personality_integration"],
        "mood_type": "scalar"
    }
    
    # Add hormone levels snapshot for debugging contextual changes
    if reason and ("contextual" in reason or "hormone_event" in reason):
        history_entry["hormone_snapshot"] = load_hormone_levels()
    
    history.append(history_entry)
    
    # Keep only recent history
    max_history = _enhanced_tracker.config["max_mood_history"]
    if len(history) > max_history:
        history = history[-max_history:]
    
    save_mood_history(history)
    
    # Print mood change for debugging
    mood_type = ""
    if hormone_context.get("is_hybrid"):
        mood_type = " [HYBRID]"
    elif hormone_context.get("is_emergent"):
        mood_type = " [EMERGENT]"
    
    print(f"[Mood Update]: {new_mood}{mood_type} (intensity: {intensity:.2f}) - {reason}")

def _apply_cross_session_mood_decay(mood: str, intensity: float) -> Tuple[str, float]:
    """Phase 3: Apply mood decay across sessions."""
    try:
        session_state = _enhanced_tracker.session_state
        last_activity = datetime.fromisoformat(session_state["last_activity"])
        hours_elapsed = (datetime.now() - last_activity).total_seconds() / 3600
        
        decay_threshold = _enhanced_tracker.config["mood_decay_hours"]
        decay_rate = _enhanced_tracker.config["mood_decay_rate"]
        
        if hours_elapsed > decay_threshold:
            # Apply exponential decay
            decay_factor = (1 - decay_rate) ** (hours_elapsed - decay_threshold)
            decayed_intensity = intensity * decay_factor
            
            # If intensity drops too low, return to neutral
            if decayed_intensity < 0.2:
                return "neutral", 0.5
            else:
                return mood, decayed_intensity
        
        return mood, intensity
    except Exception as e:
        print(f"[Mood Decay Error]: {e}")
        return mood, intensity

def _apply_cross_session_vector_decay(mood_vector: Dict[str, float], intensity: float) -> Tuple[Dict[str, float], float]:
    """Phase 4: Apply mood vector decay across sessions."""
    try:
        session_state = _enhanced_tracker.session_state
        last_activity = datetime.fromisoformat(session_state["last_activity"])
        hours_elapsed = (datetime.now() - last_activity).total_seconds() / 3600
        
        decay_threshold = _enhanced_tracker.config["mood_decay_hours"]
        decay_rate = _enhanced_tracker.config["mood_decay_rate"]
        
        if hours_elapsed > decay_threshold:
            # Apply exponential decay toward neutral (0.5) for each dimension
            decay_factor = (1 - decay_rate) ** (hours_elapsed - decay_threshold)
            
            decayed_vector = {}
            for dim, value in mood_vector.items():
                # Decay toward neutral (0.5)
                decayed_vector[dim] = 0.5 + (value - 0.5) * decay_factor
            
            # Calculate new intensity
            decayed_intensity = intensity * decay_factor
            
            # If intensity drops too low, return to completely neutral vector
            if decayed_intensity < 0.2:
                neutral_vector = {dim: 0.5 for dim in mood_vector.keys()}
                return neutral_vector, 0.5
            else:
                return decayed_vector, decayed_intensity
        
        return mood_vector, intensity
    except Exception as e:
        print(f"[Vector Mood Decay Error]: {e}")
        return mood_vector, intensity

# ─────────────────── 📊 Phase 4: Advanced mood analysis ─────────────

def get_compound_mood_analysis() -> Dict[str, Any]:
    """Phase 4: Get comprehensive compound mood analysis."""
    if not _enhanced_tracker.compound_config["enable_vector_moods"]:
        return {"vector_moods_disabled": True}
    
    current_vector = _enhanced_tracker.session_state.get("current_mood_vector", {})
    if not current_vector:
        return {"no_current_vector": True}
    
    # Get recent mood vector history
    vector_history = _enhanced_tracker.session_state.get("vector_mood_history", [])
    
    # Analyze mood vector patterns
    analysis = {
        "current_vector": current_vector,
        "vector_history_length": len(vector_history),
        "dimension_analysis": {},
        "stability_analysis": {},
        "transition_patterns": {}
    }
    
    # Analyze each dimension
    for dimension in _compound_mood_calc.dimensions:
        current_value = current_vector.get(dimension, 0.5)
        
        # Get historical values for this dimension
        historical_values = [
            entry["vector"].get(dimension, 0.5) 
            for entry in vector_history[-20:]
            if "vector" in entry
        ]
        
        if historical_values:
            analysis["dimension_analysis"][dimension] = {
                "current": current_value,
                "mean": statistics.mean(historical_values),
                "variance": statistics.variance(historical_values) if len(historical_values) > 1 else 0,
                "trend": "increasing" if len(historical_values) > 3 and historical_values[-1] > statistics.mean(historical_values[:-1]) else "stable"
            }
    
    # Analyze transition patterns
    recent_transitions = _compound_mood_calc.transition_history[-10:]
    if recent_transitions:
        transition_speeds = [t["transition_speed"] for t in recent_transitions]
        analysis["transition_patterns"] = {
            "avg_transition_speed": statistics.mean(transition_speeds),
            "max_transition_speed": max(transition_speeds),
            "smoothed_transitions": sum(1 for t in recent_transitions if t.get("was_smoothed", False)),
            "total_transitions": len(recent_transitions)
        }
    
    return analysis

def get_mood_predictions() -> Dict[str, Any]:
    """Phase 4: Get mood predictions and trend analysis."""
    current_vector = _enhanced_tracker.session_state.get("current_mood_vector", {})
    if not current_vector:
        return {"predictions_unavailable": "no_current_vector"}
    
    # Get prediction from compound mood calculator
    prediction = _compound_mood_calc.predict_mood_evolution(current_vector)
    
    if not prediction.get("prediction_available", False):
        return prediction
    
    # Enhance prediction with additional analysis
    prediction["enhanced_analysis"] = {
        "mood_trajectory": _analyze_mood_trajectory(),
        "stability_forecast": _forecast_mood_stability(),
        "intervention_suggestions": _suggest_mood_interventions(prediction["predicted_vector"])
    }
    
    return prediction

def _analyze_mood_trajectory() -> Dict[str, Any]:
    """Analyze the trajectory of mood changes over time."""
    vector_history = _enhanced_tracker.session_state.get("vector_mood_history", [])
    if len(vector_history) < 5:
        return {"insufficient_data": True}
    
    # Analyze trajectory for each dimension
    trajectories = {}
    for dimension in _compound_mood_calc.dimensions:
        values = [
            entry["vector"].get(dimension, 0.5)
            for entry in vector_history[-10:]
            if "vector" in entry
        ]
        
        if len(values) >= 3:
            # Simple linear trend analysis
            x = list(range(len(values)))
            slope = (values[-1] - values[0]) / (len(values) - 1)
            trajectories[dimension] = {
                "slope": slope,
                "direction": "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable",
                "current_value": values[-1],
                "change_magnitude": abs(slope)
            }
    
    return trajectories

def _forecast_mood_stability() -> Dict[str, Any]:
    """Forecast mood stability based on recent patterns."""
    transition_history = _compound_mood_calc.transition_history[-20:]
    if len(transition_history) < 5:
        return {"insufficient_data": True}
    
    # Calculate stability metrics
    transition_speeds = [t["transition_speed"] for t in transition_history]
    smoothed_count = sum(1 for t in transition_history if t.get("was_smoothed", False))
    avg_speed = statistics.mean(transition_speeds)
    speed_variance = statistics.variance(transition_speeds) if len(transition_speeds) > 1 else 0
    
    # Determine stability level
    if avg_speed < 0.1 and speed_variance < 0.01:
        stability_level = "very_stable"
    elif avg_speed < 0.2 and speed_variance < 0.05:
        stability_level = "stable"
    elif avg_speed < 0.3:
        stability_level = "moderate"
    else:
        stability_level = "volatile"
    
    return {
        "stability_level": stability_level,
        "avg_transition_speed": avg_speed,
        "speed_variance": speed_variance,
        "smoothed_transitions_ratio": smoothed_count / len(transition_history),
        "forecast": "stability_expected" if stability_level in ["very_stable", "stable"] else "volatility_expected"
    }

def _suggest_mood_interventions(predicted_vector: Dict[str, float]) -> List[Dict[str, Any]]:
    """Suggest interventions based on predicted mood state."""
    suggestions = []
    
    # Analyze predicted vector for potential issues
    valence = predicted_vector.get("valence", 0.5)
    arousal = predicted_vector.get("arousal", 0.5)
    dominance = predicted_vector.get("dominance", 0.5)
    
    # Low valence suggestions
    if valence < 0.3:
        suggestions.append({
            "type": "mood_boost",
            "priority": "high",
            "description": "Consider engaging in positive activities to improve valence",
            "specific_actions": ["listen to uplifting music", "practice gratitude", "connect with loved ones"]
        })
    
    # High arousal + low dominance = anxiety
    if arousal > 0.7 and dominance < 0.3:
        suggestions.append({
            "type": "anxiety_reduction", 
            "priority": "high",
            "description": "High arousal with low dominance suggests anxiety - consider calming activities",
            "specific_actions": ["deep breathing exercises", "meditation", "gentle physical activity"]
        })
    
    # Low arousal + low valence = depression risk
    if arousal < 0.3 and valence < 0.3:
        suggestions.append({
            "type": "energy_boost",
            "priority": "medium", 
            "description": "Low energy and mood - consider energizing activities",
            "specific_actions": ["light exercise", "social interaction", "engaging hobbies"]
        })
    
    return suggestions

def get_mood_patterns_analysis() -> Dict[str, Any]:
    """Phase 3+4: Analyze mood patterns over time with vector support."""
    if not _enhanced_tracker.config["mood_pattern_analysis"]:
        return {"analysis_disabled": True}
    
    history = load_mood_history()
    if len(history) < 5:
        return {"insufficient_data": True, "entries": len(history)}
    
    # Analyze recent trends
    trend_window = _enhanced_tracker.config["mood_trend_window"]
    cutoff_time = datetime.now() - timedelta(hours=trend_window)
    recent_moods = [
        entry for entry in history[-50:]  # Last 50 entries
        if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
    ]
    
    if not recent_moods:
        return {"no_recent_data": True}
    
    # Separate scalar and vector moods
    scalar_moods = [entry for entry in recent_moods if entry.get("mood_type", "scalar") == "scalar"]
    vector_moods = [entry for entry in recent_moods if entry.get("mood_type") == "vector"]
    
    analysis = {
        "analysis_period_hours": trend_window,
        "total_mood_changes": len(recent_moods),
        "scalar_moods": len(scalar_moods),
        "vector_moods": len(vector_moods),
    }
    
    # Analyze scalar moods (legacy)
    if scalar_moods:
        mood_counts = {}
        intensity_sum = 0
        for entry in scalar_moods:
            mood = entry["mood"]
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
            intensity_sum += entry["intensity"]
        
        dominant_mood = max(mood_counts.items(), key=lambda x: x[1])
        mood_diversity = len(mood_counts) / len(scalar_moods)
        avg_intensity = intensity_sum / len(scalar_moods)
        
        analysis["scalar_analysis"] = {
            "dominant_mood": {
                "mood": dominant_mood[0],
                "frequency": dominant_mood[1],
                "percentage": (dominant_mood[1] / len(scalar_moods)) * 100
            },
            "mood_diversity": mood_diversity,
            "average_intensity": avg_intensity,
            "mood_distribution": {
                mood: (count / len(scalar_moods)) * 100
                for mood, count in mood_counts.items()
            },
            "stability_score": 1.0 - mood_diversity,
        }
    
    # Analyze vector moods (Phase 4)
    if vector_moods:
        # Analyze dimensional patterns
        dimension_patterns = {}
        for dimension in _compound_mood_calc.dimensions:
            values = [
                entry["mood_vector"].get(dimension, 0.5)
                for entry in vector_moods
                if "mood_vector" in entry
            ]
            
            if values:
                dimension_patterns[dimension] = {
                    "mean": statistics.mean(values),
                    "variance": statistics.variance(values) if len(values) > 1 else 0,
                    "trend": "increasing" if len(values) > 3 and values[-1] > statistics.mean(values[:-1]) else "stable"
                }
        
        analysis["vector_analysis"] = {
            "dimension_patterns": dimension_patterns,
            "vector_stability": sum(p["variance"] for p in dimension_patterns.values()) / len(dimension_patterns) if dimension_patterns else 0
        }
    
    analysis["personality_integration"] = _enhanced_tracker.config["enable_personality_integration"]
    return analysis

def get_enhanced_mood_summary() -> Dict[str, Any]:
    """Phase 4: Get comprehensive mood summary with all enhancements."""
    current_mood_data = get_current_mood()
    hormone_levels = load_hormone_levels()
    history = load_mood_history()
    patterns = get_mood_patterns_analysis()
    recent_moods = [entry["mood"] for entry in history[-10:]] if history else []
    
    summary = {
        "current_state": current_mood_data,
        "hormone_levels": hormone_levels,
        "personality_baselines": _get_personality_baselines(),
        "session_info": {
            "session_id": _enhanced_tracker.session_state["session_id"],
            "session_mood_changes": _enhanced_tracker.session_state["session_mood_changes"],
            "session_start": _enhanced_tracker.session_state["session_start"]
        },
        "recent_patterns": {
            "recent_moods": recent_moods,
            "total_mood_changes": len(history)
        },
        "pattern_analysis": patterns,
        "personality_integration": {
            "enabled": _enhanced_tracker.config["enable_personality_integration"],
            "personality_name": _enhanced_tracker.personality.get("name", "Unknown"),
            "personality_traits": _enhanced_tracker.personality.get("personality_traits", {}),
            "influence_strength": _enhanced_tracker.config["personality_influence_strength"]
        },
        "advanced_features": {
            "cross_session_mood": _enhanced_tracker.config["enable_cross_session_mood"],
            "mood_interpolation": _enhanced_tracker.config["enable_mood_interpolation"],
            "complex_emotion_detection": _enhanced_tracker.config["complex_emotion_detection"],
            "pattern_analysis": _enhanced_tracker.config["mood_pattern_analysis"]
        }
    }
    
    # Phase 4: Add compound mood information
    if _enhanced_tracker.compound_config["enable_vector_moods"]:
        summary["compound_mood"] = {
            "vector_moods_enabled": True,
            "current_vector": _enhanced_tracker.session_state.get("current_mood_vector", {}),
            "compound_analysis": get_compound_mood_analysis(),
            "predictions": get_mood_predictions(),
            "transition_tracking": _enhanced_tracker.compound_config["enable_mood_transitions"]
        }
    else:
        summary["compound_mood"] = {"vector_moods_enabled": False}
    
    return summary

# ─────────────────── 🔧 Enhanced compatibility functions ─────────────

def get_current_mood() -> Dict[str, Any]:
    """Enhanced current mood with Phase 4 features."""
    try:
        with open("persona/mood_adjustments.json", "r", encoding="utf-8") as f:
            mood_data = json.load(f)
        
        # Add Phase 4 enhancements if missing
        if "session_id" not in mood_data:
            mood_data["session_id"] = _enhanced_tracker.session_state["session_id"]
        
        if "personality_version" not in mood_data:
            mood_data["personality_version"] = _enhanced_tracker.personality.get("version", "4.0")
        
        if "mood_type" not in mood_data:
            mood_data["mood_type"] = "vector" if "mood_vector" in mood_data else "scalar"
        
        return mood_data
    except Exception as e:
        print(f"[Current mood load error]: {e}")
        default_vector = {dim: 0.5 for dim in _compound_mood_calc.dimensions}
        return {
            "current_mood": "neutral",
            "mood_vector": default_vector,
            "intensity": 0.5,
            "context": {"is_hybrid": False, "is_emergent": False, "stability": "medium"},
            "last_updated": datetime.utcnow().isoformat(),
            "session_id": _enhanced_tracker.session_state["session_id"],
            "personality_version": "4.0",
            "mood_type": "vector"
        }

def update_mood_from_hormones(reason="hormonal_shift"):
    """Enhanced hormone-based mood update with Phase 4 features."""
    print(f"[Mood From Hormones]: Starting mood update for reason '{reason}'")
    
    # Load current hormone levels
    hormone_levels = load_hormone_levels()
    print(f"[Mood From Hormones]: Current hormones - {hormone_levels}")
    
    # Use enhanced mood calculation with compound mood support
    result = calculate_mood_from_hormones(hormone_levels)
    
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict):
        # Phase 4: Compound mood result
        mood_vector, mood_analysis = result
        print(f"[Mood From Hormones]: Calculated compound mood vector with {len(mood_vector)} dimensions")
        
        # Update mood with vector
        update_mood(mood_vector, reason=reason, hormone_context=mood_analysis.get("mixed_moods", {}))
        return mood_vector, mood_analysis.get("mixed_moods", {}), mood_analysis
    else:
        # Legacy scalar mood result
        mood, intensity = result
        print(f"[Mood From Hormones]: Calculated mood '{mood}' with intensity {intensity:.3f}")
        
        # Get enhanced context
        context = get_mood_context(mood, intensity)
        
        # Update mood with scalar values
        update_mood(mood, intensity, reason, context)
        return mood, intensity, context

# ─────────────────── 🧪 Phase 4: Testing and utilities ─────────────

def test_phase4_mood_features():
    """Test Phase 4 compound mood tracking features."""
    print("🧪 Testing Phase 4 Compound Mood Tracking")
    print("=" * 55)
    
    # Test personality integration
    print(f"🎭 Personality: {_enhanced_tracker.personality.get('name', 'Unknown')}")
    print(f"🎯 Traits: {_enhanced_tracker.personality.get('personality_traits', {})}")
    
    # Test compound mood calculation
    test_hormones = [
        {"dopamine": 0.7, "serotonin": 0.6, "cortisol": 0.4, "oxytocin": 0.8},  # Happy
        {"dopamine": 0.3, "serotonin": 0.2, "cortisol": 0.8, "oxytocin": 0.3},  # Stressed
        {"dopamine": 0.5, "serotonin": 0.5, "cortisol": 0.5, "oxytocin": 0.5},  # Neutral
    ]
    
    for i, hormones in enumerate(test_hormones, 1):
        print(f"\n🧪 Test {i}: {hormones}")
        result = calculate_mood_from_hormones(hormones)
        
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict):
            # Compound mood result
            mood_vector, analysis = result
            print(f"  Vector Result: {len(mood_vector)} dimensions")
            dominant_dims = [f"{k}:{v:.2f}" for k, v in mood_vector.items() if abs(v - 0.5) > 0.1]
            print(f"  Dominant: {', '.join(dominant_dims)}")
            if analysis.get("mixed_moods", {}).get("mixed_mood", False):
                print(f"  Mixed mood detected: {analysis['mixed_moods']['dominant_emotions']}")
        else:
            # Scalar mood result
            mood, intensity = result
            print(f"  Scalar Result: {mood} (intensity: {intensity:.2f})")
    
    # Test compound mood analysis
    print(f"\n📊 Compound Mood Analysis:")
    analysis = get_compound_mood_analysis()
    if "dimension_analysis" in analysis:
        for dim, data in analysis["dimension_analysis"].items():
            print(f"  {dim}: current={data['current']:.2f}, trend={data['trend']}")
    
    # Test predictions
    print(f"\n🔮 Mood Predictions:")
    predictions = get_mood_predictions()
    if predictions.get("prediction_available", False):
        print(f"  Confidence: {predictions['confidence']:.2f}")
        print(f"  Horizon: {predictions['prediction_horizon_hours']} hours")
    
    print("\n✅ Phase 4 compound mood tracking test completed")

# Legacy compatibility functions
handle_event_and_update_mood = lambda event: update_mood_from_hormones(f"event:{event}")

def apply_sentiment_to_mood(text: str):
    """
    Phase 4: Complete sentiment analysis to mood pipeline with hormone adjustment.
    This function properly integrates sentiment analysis with hormone adjustment.
    """
    print(f"[Mood Tracker]: Starting sentiment-to-mood pipeline for: '{text}'")
    
    try:
        # Step 1: Apply sentiment analysis and hormone adjustments
        from persona.hormone_adjuster import apply_contextual_hormone_adjustments
        updated_hormones = apply_contextual_hormone_adjustments(text)
        print(f"[Mood Tracker]: Hormones updated by ML pipeline: {updated_hormones}")
        
        # Step 2: Save the updated hormone levels
        save_hormone_levels(updated_hormones)
        
        # Step 3: Calculate mood from the newly adjusted hormones
        result = calculate_mood_from_hormones(updated_hormones)
        
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict):
            mood_vector, mood_analysis = result
            update_mood(mood_vector, reason="sentiment_analysis", hormone_context=mood_analysis.get("mixed_moods", {}))
            return mood_vector, mood_analysis
        else:
            mood, intensity = result
            update_mood(mood, intensity, reason="sentiment_analysis", hormone_context={"source": "sentiment_pipeline"})
            return mood, intensity
            
    except Exception as e:
        print(f"[Mood Tracker Error]: Failed to apply sentiment to mood - {e}")
        return "error", 0.0

get_mood_summary = get_enhanced_mood_summary
force_mood_recalculation = lambda: update_mood_from_hormones("manual_recalculation")

if __name__ == "__main__":
    test_phase4_mood_features()
