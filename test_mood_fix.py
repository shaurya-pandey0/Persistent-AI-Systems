# test_mood_fix.py
"""
Test script to verify the full mood adjustment pipeline is working.
Includes hormone-based scenarios and CLI for sentiment-based adjustment.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

def test_hormone_scenarios():
    """Test predefined hormone inputs"""
    from persona.hormone_api import save_hormone_levels
    from persona.mood_tracker import update_mood_from_hormones, get_current_mood

    print("\n🧪 Testing FIXED Mood Calculation System (Predefined Hormone Scenarios)")
    
    test_scenarios = [
        {
            "name": "High Cortisol (Stress/Anger)",
            "hormones": {"dopamine": 0.45, "serotonin": 0.47, "cortisol": 0.72, "oxytocin": 0.50}
        },
        {
            "name": "High Oxytocin + Dopamine (Love/Joy)",
            "hormones": {"dopamine": 0.65, "serotonin": 0.54, "cortisol": 0.50, "oxytocin": 0.68}
        },
        {
            "name": "Low Serotonin + High Cortisol (Depression)",
            "hormones": {"dopamine": 0.48, "serotonin": 0.35, "cortisol": 0.68, "oxytocin": 0.50}
        },
        {
            "name": "High Dopamine + Normal Others (Excitement)",
            "hormones": {"dopamine": 0.75, "serotonin": 0.55, "cortisol": 0.48, "oxytocin": 0.52}
        },
        {
            "name": "All Neutral (Baseline)",
            "hormones": {"dopamine": 0.50, "serotonin": 0.50, "cortisol": 0.50, "oxytocin": 0.50}
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"   Hormone Levels: {scenario['hormones']}")
        
        try:
            save_hormone_levels(scenario['hormones'])
            mood, intensity, context = update_mood_from_hormones(reason="test_scenario")
            current_mood_data = get_current_mood()
            
            mood_flags = []
            if context.get("is_hybrid"):
                mood_flags.append("HYBRID")
            if context.get("is_emergent"):
                mood_flags.append("EMERGENT")
            mood_type = f" [{'/'.join(mood_flags)}]" if mood_flags else ""
            
            print(f"   🎭 Calculated Mood: {mood}{mood_type}")
            print(f"   📊 Intensity: {intensity:.3f}")
            print(f"   🔬 Stability: {context.get('stability', 'unknown')}")
            print(f"   ✅ Test completed successfully!")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

def interactive_sentiment_test():
    """Interactive CLI test for sentiment-based hormone and mood update"""
    from persona.mood_tracker import apply_sentiment_to_mood, get_mood_summary

    print("\n🧠 Enter text to test the mood system via ML sentiment analysis")
    print("Type 'exit' to quit.")
    print("-" * 70)

    while True:
        user_input = input("\n🗣️  You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Exiting mood test. Goodbye!")
            break
        
        try:
            mood, intensity, context = apply_sentiment_to_mood(user_input)
            summary = get_mood_summary()

            print(f"\n🎭 Mood: {mood} {'[HYBRID]' if context['is_hybrid'] else ''}{'[EMERGENT]' if context['is_emergent'] else ''}")
            print(f"📈 Intensity: {intensity:.2f} | Stability: {context['stability']}")
            print(f"🧪 Hormones: {summary['hormone_levels']}")
            print(f"🕓 Last Mood Changes: {summary['recent_patterns']['recent_moods']}")
        except Exception as e:
            print(f"❌ Error during sentiment test: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("=" * 70)
    print("🔬 AI Mood Engine Diagnostic: Hormonal + Sentiment Integration Test")
    print("=" * 70)

    test_hormone_scenarios()
    interactive_sentiment_test()
