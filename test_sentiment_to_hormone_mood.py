import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def main():
    print("🧠 Enhanced Sentiment ➜ Hormone ➜ Mood Test Pipeline")
    print("Type 'exit' to quit.\n")

    try:
        from persona.emotion_nsfw_checker import detect_emotion, detect_toxicity, analyze_sentiment_confidence
        from persona.hormone_adjuster import (
            analyze_contextual_sentiment,
            apply_contextual_hormone_adjustments,
            get_emotion_mapping_info,
        )
        from persona.hormone_api import (
            load_hormone_levels,
            save_hormone_levels,
            infer_mood_from_hormones,
            get_mood_context,
            load_mood_weights,
        )
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return

    info = get_emotion_mapping_info()
    print(f"📊 Emotion Mappings: {info['emotion_count']} | Toxicity Types: {info['toxicity_count']}")
    print(f"⚙️  Rate Limit: {info['rate_limit']} | Confidence Threshold: {info['confidence_threshold']}\n")

    while True:
        user_input = input("🗣️  You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break

        try:
            print("\n🔍 Running Full Pipeline...\n")
            # Analyze raw
            emotions = detect_emotion(user_input)
            toxicity = detect_toxicity(user_input)
            confidence_metrics = analyze_sentiment_confidence(user_input, emotions, toxicity)

            print("📥 Emotion Detection:")
            if emotions:
                for e in emotions[:3]:
                    print(f" - {e['label']} ({e['score']:.2f})")
            else:
                print(" - No strong emotions detected.")

            print("\n☢️ Toxicity Detection:")
            if toxicity.get("is_toxic"):
                print(f" - Label: {toxicity.get('label')} (score: {toxicity.get('score'):.2f})")
            else:
                print(" - Not toxic.")

            print("\n📐 Confidence Metrics:")
            for k, v in confidence_metrics.items():
                print(f" - {k}: {v:.3f}")

            # Run full adjustment
            hormones_after = apply_contextual_hormone_adjustments(user_input)

            print("\n🧪 Updated Hormones:")
            for h, v in hormones_after.items():
                print(f" - {h}: {v:.3f}")

            # Infer mood
            mood_weights = load_mood_weights()
            mood, intensity = infer_mood_from_hormones(hormones_after, mood_weights)
            context = get_mood_context(mood, intensity)

            mood_flags = []
            if context["is_hybrid"]: mood_flags.append("HYBRID")
            if context["is_emergent"]: mood_flags.append("EMERGENT")
            mood_label = f"{mood} [{' / '.join(mood_flags)}]" if mood_flags else mood

            print("\n🎭 Inferred Mood:")
            print(f" - Mood: {mood_label}")
            print(f" - Intensity: {intensity:.2f}")
            print(f" - Stability: {context['stability']}")
            print(f" - Hormone Variance: {context['hormone_variance']:.3f}")

            print("\n✅ Full test pass.\n" + "=" * 60 + "\n")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
