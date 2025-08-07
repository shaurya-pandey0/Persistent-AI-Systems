# tests/test_temporal_decay.py

"""
Specific tests for Phase 4 temporal decay models.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_temporal_decay_models():
    """Test temporal decay functionality in detail."""
    print("🧪 Detailed Temporal Decay Model Testing")
    print("=" * 50)
    
    try:
        from persona.hormone_adjuster import (
            _decay_manager,
            get_temporal_decay_stats,
            force_hormone_decay,
            apply_contextual_hormone_adjustments
        )
        
        # Test 1: Decay factor calculation
        print("\n1️⃣ Testing decay factor calculation:")
        test_times = [0.5, 1.0, 2.0, 6.0, 12.0, 24.0]  # Hours
        
        for hormone in ["dopamine", "serotonin", "cortisol", "oxytocin"]:
            print(f"\n{hormone.title()} decay over time:")
            for hours in test_times:
                factor = _decay_manager.calculate_decay_factor(hormone, hours)
                print(f"  {hours:4.1f}h: {factor:.4f}")
        
        # Test 2: Circadian modulation
        print("\n2️⃣ Testing circadian modulation:")
        test_hours = [6, 12, 18, 24]  # Different times of day
        
        for hour in test_hours:
            # Simulate different times
            test_time = datetime.now().replace(hour=hour % 24, minute=0, second=0)
            print(f"\nTime {hour:02d}:00:")
            
            for hormone in ["dopamine", "cortisol"]:
                original_time = _decay_manager.last_decay_update
                _decay_manager.last_decay_update = test_time - timedelta(hours=2)
                
                factor = _decay_manager.calculate_decay_factor(hormone, 2.0)
                print(f"  {hormone}: {factor:.4f}")
                
                _decay_manager.last_decay_update = original_time
        
        # Test 3: Full decay process
        print("\n3️⃣ Testing full decay process:")
        
        # Set up test hormones
        test_hormones = {
            "dopamine": 0.8,
            "serotonin": 0.3, 
            "cortisol": 0.7,
            "oxytocin": 0.6
        }
        
        print(f"Initial hormones: {test_hormones}")
        
        # Simulate time passage
        original_time = _decay_manager.last_decay_update
        _decay_manager.last_decay_update = original_time - timedelta(hours=6)
        
        decayed = _decay_manager.apply_temporal_decay(test_hormones)
        print(f"After 6 hours: {decayed}")
        
        # Calculate changes
        changes = {h: decayed[h] - test_hormones[h] for h in test_hormones}
        print(f"Changes: {changes}")
        
        # Restore time
        _decay_manager.last_decay_update = original_time
        
        # Test 4: Interaction boost
        print("\n4️⃣ Testing interaction boost:")
        
        boost_time_before = _decay_manager.last_decay_update
        _decay_manager.boost_decay_resistance("conversation")
        boost_time_after = _decay_manager.last_decay_update
        
        time_diff = (boost_time_after - boost_time_before).total_seconds() / 60
        print(f"Interaction boost applied: {time_diff:.1f} minutes")
        
        # Test 5: Sentiment integration
        print("\n5️⃣ Testing sentiment with decay:")
        
        test_messages = [
            "I'm incredibly excited and happy!",
            "This is so frustrating and stressful",
            "I feel calm and peaceful"
        ]
        
        for message in test_messages:
            print(f"\nMessage: '{message}'")
            
            # Apply with decay
            result = apply_contextual_hormone_adjustments(message)
            print(f"Result: {result}")
        
        print("\n✅ All temporal decay tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Temporal decay test failed: {e}")
        return False

if __name__ == "__main__":
    test_temporal_decay_models()
