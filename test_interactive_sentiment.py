import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def main():
    print("🧠 Sentiment Analysis & Mood Pipeline (Interactive CLI)")
    print("Type 'exit' to quit.\n")

    try:
        from persona.emotion_nsfw_checker import detect_emotion, detect_toxicity
        from persona.hormone_adjuster import apply_contextual_hormone_adjustments
        from persona.mood_tracker import update_mood_from_hormones
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return

    while True:
        user_input = input("👤 You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break

        try:
            emotions = detect_emotion(user_input)
            toxicity = detect_toxicity(user_input)
            hormones_after = apply_contextual_hormone_adjustments(user_input)
            mood, intensity, context = update_mood_from_hormones(reason="cli_test")

            emotion_list = [f"{e['label']}({e['score']:.2f})" for e in emotions[:2]] if emotions else ["None"]
            mood_flags = []
            if context.get("is_hybrid"): mood_flags.append("HYBRID")
            if context.get("is_emergent"): mood_flags.append("EMERGENT")
            mood_type = f" [{'/'.join(mood_flags)}]" if mood_flags else ""

            print("\n📊 Results:")
            print(f"🎭 Emotions: {', '.join(emotion_list)}")
            print(f"☢️  Toxicity: {toxicity.get('is_toxic')} (score: {toxicity.get('score'):.3f})")
            print(f"🧪 Hormones: {hormones_after}")
            print(f"🧠 Mood: {mood}{mood_type} (intensity: {intensity:.2f})\n")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
