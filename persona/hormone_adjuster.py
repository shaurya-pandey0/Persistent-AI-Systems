# persona/hormone_adjuster.py

"""
Enhanced hormone adjustment system with Phase 4 temporal decay models.
Phase 1-3: ML-based sentiment analysis with emotion detection (WORKING) ✅
Phase 4: Temporal decay models with realistic neurochemical half-lives (NEW) 🚀
"""

from __future__ import annotations
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

# Import from neutral API module to avoid circular imports
from .hormone_api import (
    load_hormone_levels,
    save_hormone_levels,
    load_mood_weights,
    infer_mood_from_hormones,
    get_mood_context
)

# Import emotion detection pipeline
from .emotion_nsfw_checker import detect_emotion, detect_toxicity, analyze_sentiment_confidence

# Import Phase 4 configuration
try:
    from config.constants import TEMPORAL_DECAY_CONFIG, ADAPTIVE_BASELINE_CONFIG
except ImportError:
    # Fallback configuration
    TEMPORAL_DECAY_CONFIG = {
        "enable_temporal_decay": True,
        "hormone_half_lives": {
            "dopamine": 2.0, "serotonin": 6.0, "cortisol": 4.0, "oxytocin": 3.0,
        },
        "decay_update_interval_minutes": 15,
        "min_decay_threshold": 0.01,
        "decay_baseline_target": 0.5,
        "enable_circadian_modulation": True,
        "enable_interaction_boost": True,
        "interaction_boost_factor": 0.5,
    }
    ADAPTIVE_BASELINE_CONFIG = {
        "enable_adaptive_baselines": True,
        "enable_circadian_rhythms": True,
        "circadian_amplitude": 0.15,
        "peak_energy_hour": 14,
        "low_energy_hour": 3,
    }

# ---------------------- Phase 4: Temporal Decay System --------------------- #

class TemporalDecayManager:
    """
    Phase 4: Manages realistic temporal decay of hormone levels using
    exponential decay with neurochemically-inspired half-lives.
    """
    
    def __init__(self):
        self.config = TEMPORAL_DECAY_CONFIG
        self.adaptive_config = ADAPTIVE_BASELINE_CONFIG
        self.last_decay_update = self._load_last_decay_time()
        self.decay_history = []
    
    def _load_last_decay_time(self) -> datetime:
        """Load the last decay update time from persistent storage."""
        try:
            decay_file = Path("persona/hormone_decay_state.json")
            if decay_file.exists():
                with open(decay_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return datetime.fromisoformat(data.get("last_update", datetime.now().isoformat()))
            else:
                return datetime.now()
        except Exception as e:
            print(f"[Decay Manager]: Error loading decay time: {e}")
            return datetime.now()
    
    def _save_decay_state(self, current_time: datetime) -> None:
        """Save current decay state to persistent storage."""
        try:
            decay_file = Path("persona/hormone_decay_state.json")
            decay_file.parent.mkdir(exist_ok=True)
            state = {
                "last_update": current_time.isoformat(),
                "decay_history": self.decay_history[-100:], # Keep last 100 entries
                "config_version": "4.0"
            }
            with open(decay_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[Decay Manager]: Error saving decay state: {e}")
    
    def should_apply_decay(self) -> bool:
        """Determine if decay should be applied based on time elapsed."""
        if not self.config["enable_temporal_decay"]:
            return False
        
        current_time = datetime.now()
        time_elapsed = (current_time - self.last_decay_update).total_seconds() / 60 # minutes
        return time_elapsed >= self.config["decay_update_interval_minutes"]
    
    def calculate_decay_factor(self, hormone: str, hours_elapsed: float) -> float:
        """
        Calculate exponential decay factor for a hormone based on its half-life.
        Formula: decay_factor = (1/2)^(time_elapsed / half_life)
        """
        half_life = self.config["hormone_half_lives"].get(hormone, 4.0)
        if hours_elapsed <= 0:
            return 1.0
        
        # Standard exponential decay formula
        decay_factor = math.pow(0.5, hours_elapsed / half_life)
        
        # Apply circadian modulation if enabled
        if self.config["enable_circadian_modulation"]:
            decay_factor = self._apply_circadian_modulation(decay_factor, hormone)
        
        return max(0.0, min(1.0, decay_factor))
    
    def _apply_circadian_modulation(self, decay_factor: float, hormone: str) -> float:
        """Apply circadian rhythm effects to decay rate."""
        if not self.adaptive_config["enable_circadian_rhythms"]:
            return decay_factor
        
        current_hour = datetime.now().hour
        peak_hour = self.adaptive_config["peak_energy_hour"]
        low_hour = self.adaptive_config["low_energy_hour"] 
        amplitude = self.adaptive_config["circadian_amplitude"]
        
        # Calculate circadian factor (sine wave with 24-hour period)
        circadian_phase = (current_hour - peak_hour) * (2 * math.pi / 24)
        circadian_factor = math.sin(circadian_phase) * amplitude
        
        # Hormone-specific circadian effects
        hormone_modifiers = {
            "dopamine": 1.2,  # Dopamine more affected by circadian rhythms
            "serotonin": 0.8,  # Serotonin less affected
            "cortisol": 1.5,   # Cortisol heavily affected (stress hormone)
            "oxytocin": 0.9,   # Oxytocin moderately affected
        }
        
        modifier = hormone_modifiers.get(hormone, 1.0)
        modulated_factor = decay_factor * (1 + circadian_factor * modifier)
        return max(0.0, min(1.0, modulated_factor))
    
    def apply_temporal_decay(self, hormone_levels: Dict[str, float]) -> Dict[str, float]:
        """
        Apply temporal decay to all hormone levels based on elapsed time.
        Returns updated hormone levels after decay.
        """
        if not self.should_apply_decay():
            return hormone_levels
        
        current_time = datetime.now()
        hours_elapsed = (current_time - self.last_decay_update).total_seconds() / 3600
        
        # Get adaptive baselines for decay targets
        baselines = self._get_adaptive_baselines()
        
        updated_levels = hormone_levels.copy()
        print(f"[Temporal Decay]: Applying decay after {hours_elapsed:.2f} hours")
        
        for hormone, current_level in hormone_levels.items():
            baseline = baselines.get(hormone, 0.5)
            decay_factor = self.calculate_decay_factor(hormone, hours_elapsed)
            
            # Calculate new level: decay toward baseline
            # Formula: new_level = baseline + (current_level - baseline) * decay_factor  
            new_level = baseline + (current_level - baseline) * decay_factor
            
            # Apply minimum threshold
            level_change = abs(new_level - current_level)
            if level_change >= self.config["min_decay_threshold"]:
                updated_levels[hormone] = max(0.0, min(1.0, new_level))
                print(f"[Temporal Decay]: {hormone} {current_level:.3f} -> {updated_levels[hormone]:.3f} (decay: {decay_factor:.3f})")
        
        # Record decay event
        decay_event = {
            "timestamp": current_time.isoformat(),
            "hours_elapsed": hours_elapsed,
            "changes": {
                hormone: {"before": hormone_levels[hormone], "after": updated_levels[hormone]}
                for hormone in hormone_levels
                if abs(updated_levels[hormone] - hormone_levels[hormone]) >= self.config["min_decay_threshold"]
            }
        }
        
        if decay_event["changes"]:
            self.decay_history.append(decay_event)
        
        # Update tracking
        self.last_decay_update = current_time
        self._save_decay_state(current_time)
        
        return updated_levels
    
    def _get_adaptive_baselines(self) -> Dict[str, float]:
        """Get adaptive baselines for decay targets."""
        # This will be enhanced with user-specific baselines in adaptive baseline system
        # For now, use personality-adjusted baselines
        try:
            from persona.personality import get_personality_baselines
            return get_personality_baselines()
        except ImportError:
            return {
                "dopamine": 0.5,
                "serotonin": 0.5,
                "cortisol": 0.5,
                "oxytocin": 0.5,
            }
    
    def boost_decay_resistance(self, interaction_type: str = "conversation") -> None:
        """
        Phase 4: Boost decay resistance during interactions.
        During active interactions, hormones decay more slowly to represent
        sustained engagement and stimulation.
        """
        if not self.config["enable_interaction_boost"]:
            return
        
        # Temporarily modify decay factors
        boost_factor = self.config["interaction_boost_factor"]
        
        # This could be implemented by adjusting the last_decay_update time
        # to effectively slow down decay during interactions
        boost_minutes = 30 # Boost lasts 30 minutes
        boost_time = timedelta(minutes=boost_minutes * boost_factor)
        self.last_decay_update = self.last_decay_update + boost_time
        
        print(f"[Temporal Decay]: Applied interaction boost ({interaction_type})")

# Create global decay manager
_decay_manager = TemporalDecayManager()

# ---------------------- FIXED: Emotion-to-Hormone Mapping (Phase 1-3) --------------------- #

_EMOTION_HORMONE_MAP = {
    # 🔧 FIXED: Positive emotions with CORTISOL REDUCTION
    "joy": {"dopamine": 0.12, "serotonin": 0.08, "cortisol": -0.08},  # ✅ Added cortisol reduction
    "love": {"oxytocin": 0.15, "dopamine": 0.08, "serotonin": 0.05, "cortisol": -0.10},  # ✅ Added cortisol reduction
    "admiration": {"dopamine": 0.06, "oxytocin": 0.04, "cortisol": -0.04},  # ✅ Added cortisol reduction
    "excitement": {"dopamine": 0.10, "cortisol": -0.02},  # ✅ Net cortisol effect now negative
    "gratitude": {"serotonin": 0.08, "oxytocin": 0.06, "cortisol": -0.06},  # ✅ Added cortisol reduction
    "relief": {"cortisol": -0.08, "serotonin": 0.05},  # ✅ Already had cortisol reduction
    "pride": {"dopamine": 0.08, "serotonin": 0.04, "cortisol": -0.05},  # ✅ Added cortisol reduction
    "optimism": {"dopamine": 0.06, "serotonin": 0.06, "cortisol": -0.04},  # ✅ Added cortisol reduction
    "caring": {"oxytocin": 0.10, "serotonin": 0.04, "cortisol": -0.05},  # ✅ Added cortisol reduction
    "approval": {"dopamine": 0.05, "oxytocin": 0.03, "cortisol": -0.03},  # ✅ Added cortisol reduction
    
    # Negative emotions (unchanged)
    "anger": {"cortisol": 0.12, "dopamine": -0.05, "serotonin": -0.04},
    "sadness": {"serotonin": -0.10, "cortisol": 0.06, "dopamine": -0.04},
    "fear": {"cortisol": 0.15, "serotonin": -0.06},
    "disgust": {"cortisol": 0.08, "serotonin": -0.05},
    "annoyance": {"cortisol": 0.06, "serotonin": -0.03},
    "disappointment": {"dopamine": -0.08, "serotonin": -0.05},
    "embarrassment": {"cortisol": 0.08, "oxytocin": -0.04},
    "grief": {"serotonin": -0.12, "cortisol": 0.10},
    "nervousness": {"cortisol": 0.10, "dopamine": -0.03},
    "remorse": {"serotonin": -0.06, "cortisol": 0.05},
    
    # Neutral/Complex emotions
    "surprise": {"dopamine": 0.04, "cortisol": 0.02},
    "curiosity": {"dopamine": 0.05},
    "confusion": {"cortisol": 0.03},
    "neutral": {}  # No hormone changes
}

# ---------------------- Toxicity-to-Hormone Mapping ------------------- #

_TOXICITY_HORMONE_MAP = {
    "TOXIC": {"cortisol": 0.15, "serotonin": -0.08},
    "toxic": {"cortisol": 0.15, "serotonin": -0.08},
    "severe_toxic": {"cortisol": 0.20, "serotonin": -0.12, "dopamine": -0.06},
    "obscene": {"cortisol": 0.10, "serotonin": -0.06},
    "threat": {"cortisol": 0.18, "dopamine": -0.08},
    "insult": {"cortisol": 0.12, "serotonin": -0.08},
    "identity_hate": {"cortisol": 0.15, "serotonin": -0.10}
}

# ------------------------- Processing Parameters ------------------- #

_RATE_LIMIT = 0.08  # Increased from 0.05 for more responsive changes
_DECAY_STEP = 0.01  # Decay toward baseline when neutral
_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to apply changes
_TOXICITY_THRESHOLD = 0.5  # Minimum toxicity score to trigger

def _apply_resistance(current: float, delta: float) -> float:
    """Smaller effective change near extremes (0/1)."""
    distance = 1 - abs(current - 0.5) * 2  # 1 at mid, 0 at extremes
    return delta * (0.4 + 0.6 * distance)  # min 40% efficacy at extremes

def analyze_contextual_sentiment(text: str) -> dict:
    """
    ML-based contextual sentiment analysis using emotion detection pipeline.
    Returns emotion analysis, toxicity analysis, and hormone adjustment recommendations.
    """
    print(f"[ML Sentiment]: Analyzing text: '{text}'")
    
    # Get emotion detection results
    emotions = detect_emotion(text)
    toxicity_result = detect_toxicity(text)
    confidence = analyze_sentiment_confidence(text, emotions, toxicity_result)
    
    # Determine primary emotion
    primary_emotion = None
    emotion_intensity = 0.0
    
    if emotions and confidence["overall_confidence"] > _CONFIDENCE_THRESHOLD:
        primary_emotion = emotions[0]["label"].lower()
        emotion_intensity = emotions[0]["score"]
        print(f"[ML Sentiment]: Primary emotion: {primary_emotion} (confidence: {emotion_intensity:.3f})")
    else:
        primary_emotion = "neutral"
        print(f"[ML Sentiment]: Low confidence or no strong emotions detected")
    
    # Check for toxicity
    toxicity_detected = []
    max_toxicity_score = 0.0
    
    if toxicity_result and toxicity_result.get("is_toxic", False):
        toxic_score = toxicity_result.get("score", 0.0)
        toxic_label = toxicity_result.get("label", "").lower()
        
        if toxic_score > _TOXICITY_THRESHOLD:
            toxicity_detected.append(toxic_label)
            max_toxicity_score = toxic_score
            print(f"[ML Sentiment]: Toxicity detected: {toxic_label} (score: {toxic_score:.3f})")
    
    # Calculate hormone adjustments
    hormone_deltas = {}
    adjustment_reasons = []
    
    # Apply emotion-based adjustments
    if primary_emotion in _EMOTION_HORMONE_MAP:
        emotion_deltas = _EMOTION_HORMONE_MAP[primary_emotion]
        for hormone, base_delta in emotion_deltas.items():
            # Scale by emotion intensity and confidence
            scaled_delta = base_delta * emotion_intensity * confidence["overall_confidence"]
            hormone_deltas[hormone] = hormone_deltas.get(hormone, 0) + scaled_delta
            adjustment_reasons.append(f"emotion_{primary_emotion}")
    
    # Apply toxicity-based adjustments
    if toxicity_detected:
        for tox_label in toxicity_detected:
            if tox_label in _TOXICITY_HORMONE_MAP:
                tox_deltas = _TOXICITY_HORMONE_MAP[tox_label]
                for hormone, base_delta in tox_deltas.items():
                    scaled_delta = base_delta * max_toxicity_score
                    hormone_deltas[hormone] = hormone_deltas.get(hormone, 0) + scaled_delta
                    adjustment_reasons.append(f"toxicity_{tox_label}")
    
    return {
        "primary_emotion": primary_emotion,
        "emotion_intensity": emotion_intensity,
        "emotions_detected": emotions[:3] if emotions else [],
        "toxicity_detected": toxicity_detected,
        "toxicity_score": max_toxicity_score,
        "confidence_metrics": confidence,
        "hormone_deltas": hormone_deltas,
        "adjustment_reasons": adjustment_reasons,
        "detected_text": text
    }

def apply_contextual_hormone_adjustments(text: str) -> Dict[str, float]:
    """
    Phase 4: Enhanced hormone adjustment with temporal decay integration.
    Combines ML-based sentiment analysis with realistic temporal decay models.
    """
    # Phase 4: Apply temporal decay first
    current_hormones = load_hormone_levels()
    decayed_hormones = _decay_manager.apply_temporal_decay(current_hormones)
    
    # Apply new sentiment-based adjustments
    analysis = analyze_contextual_sentiment(text)
    updated = decayed_hormones.copy()
    
    print(f"[ML Hormone Adjust]: {analysis['primary_emotion']} (intensity: {analysis['emotion_intensity']:.2f})")
    
    if analysis["adjustment_reasons"]:
        print(f"[ML Hormone Adjust]: Reasons: {', '.join(analysis['adjustment_reasons'])}")
    
    if analysis["hormone_deltas"]:
        # Phase 4: Boost decay resistance during interaction
        _decay_manager.boost_decay_resistance("sentiment_analysis")
        
        for hormone, base_delta in analysis["hormone_deltas"].items():
            # Apply rate limiting
            delta = max(-_RATE_LIMIT, min(_RATE_LIMIT, base_delta))
            
            # Apply resistance curve
            delta = _apply_resistance(decayed_hormones[hormone], delta)
            
            # Update hormone level
            updated[hormone] = max(0.0, min(1.0, decayed_hormones[hormone] + delta))
            
            print(f"[ML Hormone Adjust]: {hormone} {decayed_hormones[hormone]:.3f} -> {updated[hormone]:.3f} (delta: {delta:+.4f})")
    
    else:
        # Neutral message → gradual decay toward baseline (handled by temporal decay)
        print("[ML Hormone Adjust]: Neutral input - temporal decay already applied")
    
    save_hormone_levels(updated)
    return updated

# ---------------------- Phase 4: Enhanced hormone utilities ------------------- #

def get_temporal_decay_stats() -> Dict[str, any]:
    """Get comprehensive temporal decay statistics."""
    return {
        "decay_enabled": TEMPORAL_DECAY_CONFIG["enable_temporal_decay"],
        "hormone_half_lives": TEMPORAL_DECAY_CONFIG["hormone_half_lives"],
        "last_decay_update": _decay_manager.last_decay_update.isoformat(),
        "decay_events": len(_decay_manager.decay_history),
        "recent_decay_events": _decay_manager.decay_history[-5:] if _decay_manager.decay_history else [],
        "circadian_modulation": TEMPORAL_DECAY_CONFIG["enable_circadian_modulation"],
        "interaction_boost": TEMPORAL_DECAY_CONFIG["enable_interaction_boost"],
    }

def force_hormone_decay() -> Dict[str, float]:
    """Manually force hormone decay calculation."""
    print("[Temporal Decay]: Manual decay trigger")
    current_hormones = load_hormone_levels()
    decayed_hormones = _decay_manager.apply_temporal_decay(current_hormones)
    save_hormone_levels(decayed_hormones)
    return decayed_hormones

def reset_decay_timer() -> None:
    """Reset decay timer (useful for testing)."""
    _decay_manager.last_decay_update = datetime.now()
    _decay_manager._save_decay_state(datetime.now())
    print("[Temporal Decay]: Timer reset")

# 🔧 ADDED: Manual cortisol reduction function
def apply_manual_cortisol_reduction(reduction_amount: float = 0.1, reason: str = "manual_reduction") -> Dict[str, float]:
    """
    Manually reduce cortisol levels for testing or emergency situations.
    
    Args:
        reduction_amount: Amount to reduce cortisol (0.0-1.0)
        reason: Reason for manual reduction
    
    Returns:
        Updated hormone levels
    """
    current_hormones = load_hormone_levels()
    updated_hormones = current_hormones.copy()
    
    old_cortisol = current_hormones["cortisol"]
    new_cortisol = max(0.0, old_cortisol - reduction_amount)
    updated_hormones["cortisol"] = new_cortisol
    
    save_hormone_levels(updated_hormones)
    
    print(f"[Manual Cortisol Reduction]: {old_cortisol:.3f} -> {new_cortisol:.3f} (reduction: -{reduction_amount:.3f}) - {reason}")
    
    return updated_hormones

# Legacy function mappings for compatibility
def adjust_hormones(event: str) -> Dict[str, float]:
    """Legacy function for event-based hormone adjustment."""
    event_to_text = {
        "stress": "I feel terrible and anxious about this situation",
        "positive_feedback": "This is amazing and wonderful, I love it",
        "social_connection": "I feel love and deep connection with you",
        "neutral_interaction": "This is a normal conversation"
    }
    
    text = event_to_text.get(event, event)
    print(f"[Legacy Hormone Adjust]: Converting event '{event}' to text analysis")
    return apply_contextual_hormone_adjustments(text)

# Re-export functions with consistent naming for backward compatibility
load_hormones = load_hormone_levels
save_hormones = save_hormone_levels

def get_emotion_mapping_info() -> Dict:
    """Debugging function to get information about emotion-hormone mappings."""
    return {
        "emotion_count": len(_EMOTION_HORMONE_MAP),
        "toxicity_count": len(_TOXICITY_HORMONE_MAP),
        "emotions_mapped": list(_EMOTION_HORMONE_MAP.keys()),
        "toxicity_types": list(_TOXICITY_HORMONE_MAP.keys()),
        "rate_limit": _RATE_LIMIT,
        "confidence_threshold": _CONFIDENCE_THRESHOLD,
        "toxicity_threshold": _TOXICITY_THRESHOLD,
        "temporal_decay_enabled": TEMPORAL_DECAY_CONFIG["enable_temporal_decay"],
        "cortisol_reduction_emotions": [
            emotion for emotion, deltas in _EMOTION_HORMONE_MAP.items() 
            if deltas.get("cortisol", 0) < 0
        ]
    }

# ---------------------- Phase 4: Testing and debugging ------------------- #

def test_temporal_decay():
    """Test temporal decay functionality."""
    print("🧪 Testing Phase 4 Temporal Decay System")
    print("=" * 50)
    
    # Show current configuration
    stats = get_temporal_decay_stats()
    print(f"⏰ Decay enabled: {stats['decay_enabled']}")
    print(f"🧬 Hormone half-lives: {stats['hormone_half_lives']}")
    
    # Test decay calculation
    test_hormones = {"dopamine": 0.8, "serotonin": 0.3, "cortisol": 0.7, "oxytocin": 0.6}
    print(f"\n🧪 Testing decay on: {test_hormones}")
    
    # Simulate time passage
    original_time = _decay_manager.last_decay_update
    _decay_manager.last_decay_update = original_time - timedelta(hours=3)  # 3 hours ago
    
    decayed = _decay_manager.apply_temporal_decay(test_hormones)
    print(f"📊 After 3 hours: {decayed}")
    
    # Test with different time periods
    for hours in [1, 6, 12, 24]:
        _decay_manager.last_decay_update = original_time - timedelta(hours=hours)
        test_decay = _decay_manager.apply_temporal_decay(test_hormones)
        print(f"📈 After {hours}h: dopamine {test_decay['dopamine']:.3f}, serotonin {test_decay['serotonin']:.3f}")
    
    # Restore original time
    _decay_manager.last_decay_update = original_time
    print("\n✅ Temporal decay test completed")

def test_cortisol_reduction():
    """🔧 NEW: Test cortisol reduction functionality."""
    print("🧪 Testing Cortisol Reduction System")
    print("=" * 50)
    
    # Show cortisol-reducing emotions
    info = get_emotion_mapping_info()
    print(f"📊 Emotions that reduce cortisol: {info['cortisol_reduction_emotions']}")
    
    # Test positive emotions with cortisol reduction
    test_inputs = [
        "I love you so much!",
        "This brings me so much joy!",
        "I'm feeling grateful and happy",
        "What a relief, everything is okay"
    ]
    
    # Set high cortisol for testing
    high_cortisol = {"dopamine": 0.5, "serotonin": 0.5, "cortisol": 0.8, "oxytocin": 0.5}
    save_hormone_levels(high_cortisol)
    print(f"🧪 Starting with high cortisol: {high_cortisol}")
    
    for text in test_inputs:
        print(f"\n🧪 Testing: '{text}'")
        before_levels = load_hormone_levels()
        after_levels = apply_contextual_hormone_adjustments(text)
        
        cortisol_change = after_levels["cortisol"] - before_levels["cortisol"]
        print(f"   Cortisol: {before_levels['cortisol']:.3f} -> {after_levels['cortisol']:.3f} (change: {cortisol_change:+.3f})")
        
        if cortisol_change < 0:
            print("   ✅ Cortisol reduced correctly!")
        else:
            print("   ❌ Cortisol did not reduce")
    
    print("\n✅ Cortisol reduction test completed")

# CLI for testing
if __name__ == "__main__":
    print("🧪 Phase 4 Enhanced Hormone Adjuster — Type messages, 'test decay', 'test cortisol', or 'quit' to exit")
    print("🤖 Using GoEmotions + Toxicity Detection + Temporal Decay Models + CORTISOL REDUCTION\n")
    
    # Show mapping and decay info
    info = get_emotion_mapping_info()
    print(f"📊 Loaded {info['emotion_count']} emotion mappings and {info['toxicity_count']} toxicity types")
    print(f"⚙️ Rate limit: {info['rate_limit']}, Temporal decay: {info['temporal_decay_enabled']}")
    print(f"🔧 Cortisol-reducing emotions: {len(info['cortisol_reduction_emotions'])}\n")
    
    while True:
        msg = input("You: ")
        if msg.lower().startswith("quit"):
            break
        elif msg.lower() == "test decay":
            test_temporal_decay()
            continue
        elif msg.lower() == "test cortisol":
            test_cortisol_reduction()
            continue
        elif msg.lower() == "stats":
            stats = get_temporal_decay_stats()
            print(f"📊 Decay Stats: {json.dumps(stats, indent=2)}")
            continue
        elif msg.lower().startswith("reduce cortisol"):
            parts = msg.split()
            amount = float(parts[2]) if len(parts) > 2 else 0.1
            apply_manual_cortisol_reduction(amount, "user_command")
            continue
        
        print("\n" + "="*50)
        new_levels = apply_contextual_hormone_adjustments(msg)
        
        # Show resulting mood
        mood_weights = load_mood_weights()
        mood, intensity = infer_mood_from_hormones(new_levels, mood_weights)
        context = get_mood_context(mood, intensity)
        
        mood_type = ""
        if context["is_hybrid"]:
            mood_type = " [HYBRID]"
        elif context["is_emergent"]:
            mood_type = " [EMERGENT]"
        
        print(f"\n🧠 Final State:")
        print(f"  Hormones: {new_levels}")
        print(f"  Mood: {mood}{mood_type} (intensity: {intensity:.2f})")
        
        # Show decay stats
        decay_stats = get_temporal_decay_stats()
        print(f"  Last decay: {decay_stats['last_decay_update']}")
        print("="*50 + "\n")
