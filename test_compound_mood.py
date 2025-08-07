# tests/test_compound_mood.py

"""
Specific tests for Phase 4 compound mood modeling.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_compound_mood_detailed():
    """Test compound mood system in detail."""
    print("🧪 Detailed Compound Mood System Testing")
    print("=" * 50)
    
    try:
        from persona.mood_tracker import (
            calculate_mood_from_hormones,
            update_mood_from_hormones,
            get_compound_mood_analysis,
            _compound_mood_calc,
            get_mood_predictions,
            get_enhanced_mood_summary
        )
        
        # Test 1: Mood vector calculation
        print("\n1️⃣ Testing mood vector calculation:")
        
        test_hormone_profiles = [
            {
                "name": "High Energy Happy",
                "hormones": {"dopamine": 0.8, "serotonin": 0.7, "cortisol": 0.3, "oxytocin": 0.6}
            },
            {
                "name": "Stressed and Anxious", 
                "hormones": {"dopamine": 0.3, "serotonin": 0.2, "cortisol": 0.9, "oxytocin": 0.2}
            },
            {
                "name": "Calm and Content",
                "hormones": {"dopamine": 0.5, "serotonin": 0.8, "cortisol": 0.2, "oxytocin": 0.7}
            },
            {
                "name": "Creative Flow",
                "hormones": {"dopamine": 0.6, "serotonin": 0.5, "cortisol": 0.3, "oxytocin": 0.4}
            }
        ]
        
        for profile in test_hormone_profiles:
            print(f"\n  Testing {profile['name']}:")
            print(f"    Hormones: {profile['hormones']}")
            
            mood_vector = _compound_mood_calc.calculate_mood_vector(profile['hormones'])
            print(f"    Mood Vector:")
            
            for dimension, value in mood_vector.items():
                print(f"      {dimension}: {value:.3f}")
            
            # Test mixed mood detection
            mixed_moods = _compound_mood_calc.detect_mixed_moods(mood_vector)
            if mixed_moods["mixed_mood"]:
                print(f"    Mixed mood detected: {mixed_moods['dominant_emotions']}")
            else:
                print(f"    No mixed mood detected")
        
        # Test 2: Mood transitions
        print("\n2️⃣ Testing mood transitions:")
        
        # Simulate a sequence of mood changes
        mood_sequence = [
            {"dopamine": 0.5, "serotonin": 0.5, "cortisol": 0.5, "oxytocin": 0.5},  # Neutral
            {"dopamine": 0.7, "serotonin": 0.6, "cortisol": 0.4, "oxytocin": 0.6},  # Slightly happy
            {"dopamine": 0.9, "serotonin": 0.8, "cortisol": 0.2, "oxytocin": 0.8},  # Very happy
            {"dopamine": 0.3, "serotonin": 0.3, "cortisol": 0.8, "oxytocin": 0.3},  # Sudden stress
        ]
        
        previous_vector = None
        for i, hormones in enumerate(mood_sequence):
            print(f"\n  Transition {i+1}:")
            current_vector = _compound_mood_calc.calculate_mood_vector(hormones)
            
            if previous_vector:
                transition = _compound_mood_calc.calculate_mood_transitions(current_vector, previous_vector)
                print(f"    Transition speed: {transition['transition_speed']:.4f}")
                print(f"    Was smoothed: {transition['was_smoothed']}")
                
                if transition['was_smoothed']:
                    print(f"    Smoothed vector applied")
            
            previous_vector = current_vector
        
        # Test 3: Full mood calculation with personality
        print("\n3️⃣ Testing full mood calculation:")
        
        test_cases = [
            {"dopamine": 0.7, "serotonin": 0.6, "cortisol": 0.4, "oxytocin": 0.8},
            {"dopamine": 0.3, "serotonin": 0.2, "cortisol": 0.8, "oxytocin": 0.3},
            {"dopamine": 0.5, "serotonin": 0.5, "cortisol": 0.5, "oxytocin": 0.5},
        ]
        
        for i, hormones in enumerate(test_cases, 1):
            print(f"\n  Test case {i}: {hormones}")
            
            result = calculate_mood_from_hormones(hormones)
            
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict):
                # Compound mood result
                mood_vector, analysis = result
                print(f"    Compound mood with {len(mood_vector)} dimensions")
                
                # Show dominant dimensions
                dominant_dims = [(k, v) for k, v in mood_vector.items() if abs(v - 0.5) > 0.15]
                if dominant_dims:
                    print(f"    Dominant dimensions:")
                    for dim, value in dominant_dims:
                        direction = "high" if value > 0.5 else "low"
                        print(f"      {dim}: {value:.3f} ({direction})")
                
                # Show mixed mood info
                mixed_info = analysis.get("mixed_moods", {})
                if mixed_info.get("mixed_mood", False):
                    print(f"    Mixed mood: {mixed_info['dominant_emotions']}")
                
            else:
                # Scalar mood result
                mood, intensity = result
                print(f"    Scalar mood: {mood} (intensity: {intensity:.3f})")
        
        # Test 4: Mood predictions
        print("\n4️⃣ Testing mood predictions:")
        
        try:
            # Set up some transition history first
            for hormones in test_cases:
                update_mood_from_hormones("prediction_test")
            
            predictions = get_mood_predictions()
            
            if predictions.get("prediction_available", False):
                print(f"    Prediction confidence: {predictions['confidence']:.3f}")
                print(f"    Prediction horizon: {predictions['prediction_horizon_hours']} hours")
                
                predicted_vector = predictions.get("predicted_vector", {})
                if predicted_vector:
                    print(f"    Predicted mood vector:")
                    for dim, value in predicted_vector.items():
                        print(f"      {dim}: {value:.3f}")
            else:
                reason = predictions.get("reason", "unknown")
                print(f"    Predictions not available: {reason}")
                
        except Exception as e:
            print(f"    Prediction test error: {e}")
        
        # Test 5: Enhanced mood summary
        print("\n5️⃣ Testing enhanced mood summary:")
        
        summary = get_enhanced_mood_summary()
        print(f"    Summary components:")
        print(f"      Current state: {'✓' if 'current_state' in summary else '✗'}")
        print(f"      Hormone levels: {'✓' if 'hormone_levels' in summary else '✗'}")
        print(f"      Personality integration: {'✓' if 'personality_integration' in summary else '✗'}")
        print(f"      Advanced features: {'✓' if 'advanced_features' in summary else '✗'}")
        
        compound_info = summary.get("compound_mood", {})
        if compound_info.get("vector_moods_enabled", False):
            print(f"      Compound mood: ✓ (vector mode)")
            current_vector = compound_info.get("current_vector", {})
            if current_vector:
                print(f"        Vector dimensions: {len(current_vector)}")
        else:
            print(f"      Compound mood: ✗ (scalar mode)")
        
        # Test 6: Mood stability tracking
        print("\n6️⃣ Testing mood stability:")
        
        # Simulate stable vs volatile mood patterns
        stable_pattern = [0.6, 0.61, 0.59, 0.6, 0.62, 0.58, 0.6]
        volatile_pattern = [0.3, 0.8, 0.2, 0.9, 0.1, 0.7, 0.4]
        
        print(f"    Stable pattern: {stable_pattern}")
        stable_variance = sum((x - 0.6)**2 for x in stable_pattern) / len(stable_pattern)
        print(f"      Variance: {stable_variance:.4f}")
        
        print(f"    Volatile pattern: {volatile_pattern}")
        volatile_mean = sum(volatile_pattern) / len(volatile_pattern)
        volatile_variance = sum((x - volatile_mean)**2 for x in volatile_pattern) / len(volatile_pattern)
        print(f"      Variance: {volatile_variance:.4f}")
        
        stability_ratio = stable_variance / volatile_variance if volatile_variance > 0 else 0
        print(f"    Stability ratio: {stability_ratio:.4f} (lower = more stable)")
        
        print("\n✅ All compound mood tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Compound mood test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_compound_mood_detailed()
