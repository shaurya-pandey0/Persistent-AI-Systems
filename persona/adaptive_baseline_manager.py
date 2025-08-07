# persona/adaptive_baseline_manager.py

"""
Phase 4: Adaptive Baseline Manager with User-Specific Calibration.

This module manages adaptive hormone baselines that learn from user interactions,
incorporates circadian rhythms, and provides multi-modal context integration.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import statistics

# Import Phase 4 configuration
try:
    from config.constants import ADAPTIVE_BASELINE_CONFIG, TEMPORAL_DECAY_CONFIG
except ImportError:
    ADAPTIVE_BASELINE_CONFIG = {
        "enable_adaptive_baselines": True,
        "baseline_adaptation_rate": 0.02,
        "baseline_stability_period": 72,
        "min_baseline_samples": 50,
        "enable_circadian_rhythms": True,
        "circadian_amplitude": 0.15,
        "peak_energy_hour": 14,
        "low_energy_hour": 3,
        "enable_context_adaptation": True,
        "context_influence_strength": 0.2,
        "supported_contexts": ["work", "social", "creative", "rest", "exercise", "stress"],
        "personality_baseline_influence": 0.4,
        "user_feedback_weight": 0.3,
        "environmental_factor_weight": 0.1,
    }

# Import hormone and personality systems
try:
    from persona.hormone_api import load_hormone_levels, save_hormone_levels
    from persona.personality import load_personality_traits
except ImportError:
    # Fallback functions
    def load_hormone_levels():
        return {"dopamine": 0.5, "serotonin": 0.5, "cortisol": 0.5, "oxytocin": 0.5}
    def save_hormone_levels(levels):
        pass
    def load_personality_traits():
        return {}

# File paths
ADAPTIVE_BASELINES_FILE = Path("persona/adaptive_baselines.json")
BASELINE_HISTORY_FILE = Path("persona/baseline_history.jsonl")
USER_FEEDBACK_FILE = Path("persona/user_feedback.json")
CIRCADIAN_PROFILE_FILE = Path("persona/circadian_profile.json")
CONTEXT_BASELINES_FILE = Path("persona/context_baselines.json")

# ─────────────────── 🔄 Adaptive Baseline System ─────────────────

class AdaptiveBaselineManager:
    """
    Manages user-specific hormone baselines that adapt over time based on
    interaction patterns, user feedback, and contextual factors.
    """
    
    def __init__(self):
        self.config = ADAPTIVE_BASELINE_CONFIG
        self.baselines = self._load_baselines()
        self.baseline_history = self._load_baseline_history()
        self.user_feedback = self._load_user_feedback()
        self.circadian_profile = self._load_circadian_profile()
        self.context_baselines = self._load_context_baselines()
        self.learning_samples = []
        
    def _load_baselines(self) -> Dict[str, float]:
        """Load current adaptive baselines."""
        try:
            if ADAPTIVE_BASELINES_FILE.exists():
                with open(ADAPTIVE_BASELINES_FILE, "r", encoding="utf-8") as f:
                    baselines = json.load(f)
                    # Ensure all hormones are present
                    default_baselines = {"dopamine": 0.5, "serotonin": 0.5, "cortisol": 0.5, "oxytocin": 0.5}
                    for hormone in default_baselines:
                        if hormone not in baselines:
                            baselines[hormone] = default_baselines[hormone]
                    return baselines
            else:
                return {"dopamine": 0.5, "serotonin": 0.5, "cortisol": 0.5, "oxytocin": 0.5}
        except Exception as e:
            print(f"[Adaptive Baselines Load Error]: {e}")
            return {"dopamine": 0.5, "serotonin": 0.5, "cortisol": 0.5, "oxytocin": 0.5}
    
    def _save_baselines(self) -> None:
        """Save current adaptive baselines."""
        try:
            ADAPTIVE_BASELINES_FILE.parent.mkdir(exist_ok=True)
            with open(ADAPTIVE_BASELINES_FILE, "w", encoding="utf-8") as f:
                json.dump(self.baselines, f, indent=2)
        except Exception as e:
            print(f"[Adaptive Baselines Save Error]: {e}")
    
    def _load_baseline_history(self) -> List[Dict]:
        """Load baseline adaptation history."""
        if not BASELINE_HISTORY_FILE.exists():
            return []
        
        try:
            history = []
            with open(BASELINE_HISTORY_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        history.append(json.loads(line))
            return history[-1000:]  # Keep last 1000 entries
        except Exception as e:
            print(f"[Baseline History Load Error]: {e}")
            return []
    
    def _save_baseline_history_entry(self, entry: Dict) -> None:
        """Save a single baseline history entry."""
        try:
            BASELINE_HISTORY_FILE.parent.mkdir(exist_ok=True)
            with open(BASELINE_HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[Baseline History Save Error]: {e}")
    
    def _load_user_feedback(self) -> Dict[str, Any]:
        """Load user feedback data."""
        try:
            if USER_FEEDBACK_FILE.exists():
                with open(USER_FEEDBACK_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return {"feedback_history": [], "mood_preferences": {}, "interaction_satisfaction": []}
        except Exception as e:
            print(f"[User Feedback Load Error]: {e}")
            return {"feedback_history": [], "mood_preferences": {}, "interaction_satisfaction": []}
    
    def _save_user_feedback(self) -> None:
        """Save user feedback data."""
        try:
            USER_FEEDBACK_FILE.parent.mkdir(exist_ok=True)
            with open(USER_FEEDBACK_FILE, "w", encoding="utf-8") as f:
                json.dump(self.user_feedback, f, indent=2)
        except Exception as e:
            print(f"[User Feedback Save Error]: {e}")
    
    def _load_circadian_profile(self) -> Dict[str, Any]:
        """Load user's circadian rhythm profile."""
        try:
            if CIRCADIAN_PROFILE_FILE.exists():
                with open(CIRCADIAN_PROFILE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return self._create_default_circadian_profile()
        except Exception as e:
            print(f"[Circadian Profile Load Error]: {e}")
            return self._create_default_circadian_profile()
    
    def _create_default_circadian_profile(self) -> Dict[str, Any]:
        """Create default circadian rhythm profile."""
        return {
            "peak_energy_hour": self.config["peak_energy_hour"],
            "low_energy_hour": self.config["low_energy_hour"],
            "amplitude": self.config["circadian_amplitude"],
            "learned_patterns": {},
            "adaptation_count": 0,
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_circadian_profile(self) -> None:
        """Save circadian rhythm profile."""
        try:
            CIRCADIAN_PROFILE_FILE.parent.mkdir(exist_ok=True)
            with open(CIRCADIAN_PROFILE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.circadian_profile, f, indent=2)
        except Exception as e:
            print(f"[Circadian Profile Save Error]: {e}")
    
    def _load_context_baselines(self) -> Dict[str, Dict[str, float]]:
        """Load context-specific baselines."""
        try:
            if CONTEXT_BASELINES_FILE.exists():
                with open(CONTEXT_BASELINES_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            print(f"[Context Baselines Load Error]: {e}")
            return {}
    
    def _save_context_baselines(self) -> None:
        """Save context-specific baselines."""
        try:
            CONTEXT_BASELINES_FILE.parent.mkdir(exist_ok=True)
            with open(CONTEXT_BASELINES_FILE, "w", encoding="utf-8") as f:
                json.dump(self.context_baselines, f, indent=2)
        except Exception as e:
            print(f"[Context Baselines Save Error]: {e}")
    
    def get_adaptive_baselines(self, context: str = None, current_time: datetime = None) -> Dict[str, float]:
        """
        Get adaptive baselines adjusted for current context and circadian rhythms.
        """
        if not self.config["enable_adaptive_baselines"]:
            return {"dopamine": 0.5, "serotonin": 0.5, "cortisol": 0.5, "oxytocin": 0.5}
        
        # Start with learned baselines
        adjusted_baselines = self.baselines.copy()
        
        # Apply circadian rhythm adjustments
        if self.config["enable_circadian_rhythms"]:
            circadian_adjustments = self._calculate_circadian_adjustments(current_time)
            for hormone, adjustment in circadian_adjustments.items():
                adjusted_baselines[hormone] = max(0.0, min(1.0, adjusted_baselines[hormone] + adjustment))
        
        # Apply context-specific adjustments
        if context and self.config["enable_context_adaptation"]:
            context_adjustments = self._get_context_adjustments(context)
            influence_strength = self.config["context_influence_strength"]
            
            for hormone, adjustment in context_adjustments.items():
                adjusted_baselines[hormone] = (
                    adjusted_baselines[hormone] * (1 - influence_strength) +
                    adjustment * influence_strength
                )
        
        # Apply personality influence
        personality_influence = self.config["personality_baseline_influence"]
        if personality_influence > 0:
            personality_adjustments = self._get_personality_baseline_adjustments()
            for hormone, adjustment in personality_adjustments.items():
                adjusted_baselines[hormone] = (
                    adjusted_baselines[hormone] * (1 - personality_influence) +
                    adjustment * personality_influence
                )
        
        return adjusted_baselines
    
    def _calculate_circadian_adjustments(self, current_time: datetime = None) -> Dict[str, float]:
        """Calculate circadian rhythm adjustments for baselines."""
        if current_time is None:
            current_time = datetime.now()
        
        current_hour = current_time.hour
        peak_hour = self.circadian_profile["peak_energy_hour"]
        low_hour = self.circadian_profile["low_energy_hour"]
        amplitude = self.circadian_profile["amplitude"]
        
        # Calculate circadian phase (sine wave with 24-hour period)
        # Peak at peak_hour, trough at low_hour
        phase_offset = (current_hour - peak_hour) * (2 * math.pi / 24)
        circadian_factor = math.sin(phase_offset) * amplitude
        
        # Hormone-specific circadian effects
        adjustments = {
            "dopamine": circadian_factor * 1.2,      # Dopamine more affected by energy cycles
            "serotonin": circadian_factor * 0.8,     # Serotonin less affected
            "cortisol": -circadian_factor * 1.5,     # Cortisol inversely related to energy
            "oxytocin": circadian_factor * 0.6,      # Oxytocin moderately affected
        }
        
        return adjustments
    
    def _get_context_adjustments(self, context: str) -> Dict[str, float]:
        """Get context-specific baseline adjustments."""
        if context not in self.context_baselines:
            # Initialize context baselines if not present
            self.context_baselines[context] = self.baselines.copy()
            self._save_context_baselines()
        
        return self.context_baselines[context]
    
    def _get_personality_baseline_adjustments(self) -> Dict[str, float]:
        """Get personality-based baseline adjustments."""
        try:
            personality_traits = load_personality_traits()
        except:
            personality_traits = {}
        
        # Default adjustments
        adjustments = {"dopamine": 0.5, "serotonin": 0.5, "cortisol": 0.5, "oxytocin": 0.5}
        
        # Adjust based on personality traits
        temperament = personality_traits.get("temperament", "balanced")
        if temperament == "playful":
            adjustments["dopamine"] += 0.1
            adjustments["oxytocin"] += 0.05
        elif temperament == "serious":
            adjustments["cortisol"] += 0.05
            adjustments["serotonin"] += 0.05
        
        tone = personality_traits.get("tone", "neutral")
        if tone == "optimistic":
            adjustments["serotonin"] += 0.08
            adjustments["dopamine"] += 0.05
        elif tone == "cautious":
            adjustments["cortisol"] += 0.03
        
        return adjustments
    
    def update_baselines_from_interaction(self, hormone_levels: Dict[str, float], 
                                        user_satisfaction: float = None,
                                        context: str = None) -> None:
        """
        Update adaptive baselines based on interaction data.
        
        Args:
            hormone_levels: Current hormone levels after interaction
            user_satisfaction: User satisfaction score (0-1)
            context: Current interaction context
        """
        if not self.config["enable_adaptive_baselines"]:
            return
        
        # Add learning sample
        learning_sample = {
            "timestamp": datetime.now().isoformat(),
            "hormone_levels": hormone_levels,
            "user_satisfaction": user_satisfaction,
            "context": context
        }
        
        self.learning_samples.append(learning_sample)
        
        # Keep only recent samples
        if len(self.learning_samples) > 200:
            self.learning_samples = self.learning_samples[-100:]
        
        # Update baselines if we have enough samples
        if len(self.learning_samples) >= self.config["min_baseline_samples"]:
            self._adapt_baselines()
        
        # Update context-specific baselines
        if context and self.config["enable_context_adaptation"]:
            self._update_context_baselines(context, hormone_levels, user_satisfaction)
    
    def _adapt_baselines(self) -> None:
        """Adapt baselines based on collected learning samples."""
        if len(self.learning_samples) < self.config["min_baseline_samples"]:
            return
        
        adaptation_rate = self.config["baseline_adaptation_rate"]
        
        # Calculate target baselines from recent successful interactions
        recent_samples = self.learning_samples[-100:]  # Use last 100 samples
        
        # Filter for high-satisfaction interactions
        good_interactions = [
            sample for sample in recent_samples
            if sample.get("user_satisfaction", 0.5) > 0.7
        ]
        
        if len(good_interactions) < 10:
            return  # Not enough good interactions to adapt
        
        # Calculate new baselines
        new_baselines = {}
        for hormone in self.baselines.keys():
            # Calculate mean hormone levels from good interactions
            hormone_values = [
                interaction["hormone_levels"].get(hormone, 0.5)
                for interaction in good_interactions
            ]
            
            if hormone_values:
                target_baseline = statistics.mean(hormone_values)
                
                # Gradually adapt toward target
                current_baseline = self.baselines[hormone]
                new_baseline = current_baseline + (target_baseline - current_baseline) * adaptation_rate
                
                # Ensure reasonable bounds
                new_baselines[hormone] = max(0.2, min(0.8, new_baseline))
            else:
                new_baselines[hormone] = self.baselines[hormone]
        
        # Check if adaptation is significant enough
        total_change = sum(abs(new_baselines[h] - self.baselines[h]) for h in self.baselines.keys())
        
        if total_change > 0.01:  # Only update if change is meaningful
            # Log baseline adaptation
            adaptation_entry = {
                "timestamp": datetime.now().isoformat(),
                "old_baselines": self.baselines.copy(),
                "new_baselines": new_baselines.copy(),
                "total_change": total_change,
                "samples_used": len(good_interactions),
                "adaptation_reason": "interaction_learning"
            }
            
            self._save_baseline_history_entry(adaptation_entry)
            
            # Update baselines
            self.baselines = new_baselines
            self._save_baselines()
            
            print(f"[Adaptive Baselines]: Updated baselines (total change: {total_change:.3f})")
            for hormone, new_value in new_baselines.items():
                old_value = adaptation_entry["old_baselines"][hormone]
                print(f"  {hormone}: {old_value:.3f} -> {new_value:.3f}")
    
    def _update_context_baselines(self, context: str, hormone_levels: Dict[str, float], 
                                user_satisfaction: float = None) -> None:
        """Update context-specific baselines."""
        if context not in self.context_baselines:
            self.context_baselines[context] = self.baselines.copy()
        
        # Adapt context baselines if user satisfaction is high
        if user_satisfaction and user_satisfaction > 0.7:
            adaptation_rate = self.config["baseline_adaptation_rate"] * 0.5  # Slower adaptation for context
            
            for hormone, level in hormone_levels.items():
                current_context_baseline = self.context_baselines[context].get(hormone, 0.5)
                new_context_baseline = current_context_baseline + (level - current_context_baseline) * adaptation_rate
                self.context_baselines[context][hormone] = max(0.2, min(0.8, new_context_baseline))
            
            self._save_context_baselines()
    
    def record_user_feedback(self, feedback_type: str, feedback_value: Any, context: str = None) -> None:
        """Record user feedback for baseline adaptation."""
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": feedback_type,
            "value": feedback_value,
            "context": context
        }
        
        self.user_feedback["feedback_history"].append(feedback_entry)
        
        # Process specific feedback types
        if feedback_type == "mood_preference":
            mood, preference_score = feedback_value
            self.user_feedback["mood_preferences"][mood] = preference_score
        elif feedback_type == "interaction_satisfaction":
            self.user_feedback["interaction_satisfaction"].append(feedback_value)
        
        # Keep feedback history manageable
        if len(self.user_feedback["feedback_history"]) > 500:
            self.user_feedback["feedback_history"] = self.user_feedback["feedback_history"][-250:]
        
        self._save_user_feedback()
        
        # Use feedback to influence baseline adaptation
        self._adapt_baselines_from_feedback()
    
    def _adapt_baselines_from_feedback(self) -> None:
        """Adapt baselines based on user feedback."""
        feedback_weight = self.config["user_feedback_weight"]
        
        # Get recent satisfaction scores
        recent_satisfaction = self.user_feedback["interaction_satisfaction"][-20:]
        if len(recent_satisfaction) < 5:
            return
        
        avg_satisfaction = statistics.mean(recent_satisfaction)
        
        # If satisfaction is consistently low, make small adjustments
        if avg_satisfaction < 0.4:
            print("[Adaptive Baselines]: Low satisfaction detected, applying corrective adjustments")
            
            # Make small random adjustments to explore better baselines
            for hormone in self.baselines.keys():
                adjustment = (hash(hormone + str(time.time())) % 100 - 50) / 1000  # ±0.05 random adjustment
                self.baselines[hormone] = max(0.2, min(0.8, self.baselines[hormone] + adjustment))
            
            self._save_baselines()
    
    def learn_circadian_patterns(self, hormone_levels: Dict[str, float], mood_score: float) -> None:
        """Learn user's circadian patterns from interaction data."""
        if not self.config["enable_circadian_rhythms"]:
            return
        
        current_hour = datetime.now().hour
        
        # Update learned patterns
        if "hourly_patterns" not in self.circadian_profile:
            self.circadian_profile["hourly_patterns"] = {}
        
        hour_key = str(current_hour)
        if hour_key not in self.circadian_profile["hourly_patterns"]:
            self.circadian_profile["hourly_patterns"][hour_key] = {
                "mood_scores": [],
                "hormone_averages": {h: [] for h in hormone_levels.keys()}
            }
        
        # Add data point
        self.circadian_profile["hourly_patterns"][hour_key]["mood_scores"].append(mood_score)
        for hormone, level in hormone_levels.items():
            self.circadian_profile["hourly_patterns"][hour_key]["hormone_averages"][hormone].append(level)
        
        # Keep limited history per hour
        for hour_data in self.circadian_profile["hourly_patterns"].values():
            if len(hour_data["mood_scores"]) > 50:
                hour_data["mood_scores"] = hour_data["mood_scores"][-25:]
            for hormone_list in hour_data["hormone_averages"].values():
                if len(hormone_list) > 50:
                    hormone_list[:] = hormone_list[-25:]
        
        # Adapt circadian profile if we have enough data
        total_data_points = sum(
            len(data["mood_scores"]) 
            for data in self.circadian_profile["hourly_patterns"].values()
        )
        
        if total_data_points > 100:
            self._adapt_circadian_profile()
    
    def _adapt_circadian_profile(self) -> None:
        """Adapt circadian profile based on learned patterns."""
        hourly_patterns = self.circadian_profile.get("hourly_patterns", {})
        
        if not hourly_patterns:
            return
        
        # Find peak and low energy hours based on mood scores
        hourly_mood_averages = {}
        for hour_str, data in hourly_patterns.items():
            if data["mood_scores"]:
                hourly_mood_averages[int(hour_str)] = statistics.mean(data["mood_scores"])
        
        if len(hourly_mood_averages) >= 12:  # Need data for at least half the day
            # Update peak and low energy hours
            peak_hour = max(hourly_mood_averages.items(), key=lambda x: x[1])[0]
            low_hour = min(hourly_mood_averages.items(), key=lambda x: x[1])[0]
            
            # Gradually adapt
            learning_rate = 0.1
            current_peak = self.circadian_profile["peak_energy_hour"]
            current_low = self.circadian_profile["low_energy_hour"]
            
            new_peak = current_peak + (peak_hour - current_peak) * learning_rate
            new_low = current_low + (low_hour - current_low) * learning_rate
            
            self.circadian_profile["peak_energy_hour"] = int(round(new_peak)) % 24
            self.circadian_profile["low_energy_hour"] = int(round(new_low)) % 24
            self.circadian_profile["adaptation_count"] += 1
            self.circadian_profile["last_updated"] = datetime.now().isoformat()
            
            self._save_circadian_profile()
            
            print(f"[Circadian Adaptation]: Peak hour: {self.circadian_profile['peak_energy_hour']}, Low hour: {self.circadian_profile['low_energy_hour']}")

# ─────────────────── 🌐 Global Adaptive Baseline Manager ─────────────────

# Create global adaptive baseline manager
_adaptive_baseline_manager = AdaptiveBaselineManager()

# ─────────────────── 📡 Public API Functions ─────────────────

def get_adaptive_baselines(context: str = None) -> Dict[str, float]:
    """Get current adaptive baselines for hormones."""
    return _adaptive_baseline_manager.get_adaptive_baselines(context)

def update_baselines_from_interaction(hormone_levels: Dict[str, float], 
                                    user_satisfaction: float = None,
                                    context: str = None) -> None:
    """Update baselines based on interaction outcome."""
    _adaptive_baseline_manager.update_baselines_from_interaction(hormone_levels, user_satisfaction, context)

def record_user_feedback(feedback_type: str, feedback_value: Any, context: str = None) -> None:
    """Record user feedback for baseline learning."""
    _adaptive_baseline_manager.record_user_feedback(feedback_type, feedback_value, context)

def learn_circadian_patterns(hormone_levels: Dict[str, float], mood_score: float) -> None:
    """Learn user's circadian rhythm patterns."""
    _adaptive_baseline_manager.learn_circadian_patterns(hormone_levels, mood_score)

def get_baseline_adaptation_stats() -> Dict[str, Any]:
    """Get statistics about baseline adaptation."""
    return {
        "current_baselines": _adaptive_baseline_manager.baselines,
        "adaptation_history": len(_adaptive_baseline_manager.baseline_history),
        "learning_samples": len(_adaptive_baseline_manager.learning_samples),
        "user_feedback_entries": len(_adaptive_baseline_manager.user_feedback["feedback_history"]),
        "context_baselines": len(_adaptive_baseline_manager.context_baselines),
        "circadian_profile": _adaptive_baseline_manager.circadian_profile,
        "configuration": {
            "adaptive_enabled": _adaptive_baseline_manager.config["enable_adaptive_baselines"],
            "circadian_enabled": _adaptive_baseline_manager.config["enable_circadian_rhythms"],
            "context_adaptation": _adaptive_baseline_manager.config["enable_context_adaptation"],
            "adaptation_rate": _adaptive_baseline_manager.config["baseline_adaptation_rate"]
        }
    }

def reset_adaptive_baselines() -> None:
    """Reset adaptive baselines to defaults (for testing)."""
    _adaptive_baseline_manager.baselines = {"dopamine": 0.5, "serotonin": 0.5, "cortisol": 0.5, "oxytocin": 0.5}
    _adaptive_baseline_manager.learning_samples = []
    _adaptive_baseline_manager._save_baselines()
    print("[Adaptive Baselines]: Reset to defaults")

# ─────────────────── 🧪 Testing and Debugging Functions ─────────────────

def test_adaptive_baselines():
    """Test adaptive baseline functionality."""
    print("🧪 Testing Phase 4 Adaptive Baseline System")
    print("=" * 55)
    
    # Test baseline retrieval
    print("📊 Current baselines:")
    baselines = get_adaptive_baselines()
    for hormone, level in baselines.items():
        print(f"  {hormone}: {level:.3f}")
    
    # Test circadian adjustments
    print(f"\n🌅 Circadian-adjusted baselines:")
    morning_baselines = get_adaptive_baselines()  # Current time
    print(f"Current time baselines:")
    for hormone, level in morning_baselines.items():
        print(f"  {hormone}: {level:.3f}")
    
    # Test context-specific baselines
    print(f"\n💼 Context-specific baselines:")
    contexts = ["work", "social", "creative", "rest"]
    for context in contexts:
        context_baselines = get_adaptive_baselines(context)
        print(f"{context.capitalize()} context:")
        for hormone, level in context_baselines.items():
            print(f"  {hormone}: {level:.3f}")
    
    # Test learning from interactions
    print(f"\n🧠 Testing baseline learning:")
    test_interactions = [
        ({"dopamine": 0.6, "serotonin": 0.7, "cortisol": 0.3, "oxytocin": 0.8}, 0.9, "social"),
        ({"dopamine": 0.8, "serotonin": 0.5, "cortisol": 0.4, "oxytocin": 0.6}, 0.8, "creative"),
        ({"dopamine": 0.4, "serotonin": 0.6, "cortisol": 0.7, "oxytocin": 0.4}, 0.3, "work"),
    ]
    
    for hormones, satisfaction, context in test_interactions:
        update_baselines_from_interaction(hormones, satisfaction, context)
        print(f"Recorded interaction: {context} (satisfaction: {satisfaction})")
    
    # Test user feedback
    print(f"\n💬 Testing user feedback:")
    record_user_feedback("mood_preference", ("cheerful", 0.9))
    record_user_feedback("interaction_satisfaction", 0.8)
    print("Recorded user feedback")
    
    # Test statistics
    print(f"\n📈 Adaptation statistics:")
    stats = get_baseline_adaptation_stats()
    print(f"Learning samples: {stats['learning_samples']}")
    print(f"Feedback entries: {stats['user_feedback_entries']}")
    print(f"Context baselines: {stats['context_baselines']}")
    
    print("\n✅ Phase 4 adaptive baseline test completed")

if __name__ == "__main__":
    test_adaptive_baselines()
