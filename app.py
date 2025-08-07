# app.py – FIXED VERSION with correct parameter names
import json
import os
from pathlib import Path
import streamlit as st

# Load environment variables early
from dotenv import load_dotenv
load_dotenv(Path.cwd() / ".env", override=False)

# Import helpers
from utils.session_id import get_or_create_session_file, save_turn_to_session
from utils.ui_helpers import render_message
from core.context_block_builder import build_session_init_prompt, build_turn_prompt
from core.fact_extractor import store_fact, load_facts
from memory.turn_memory import dump_turn, load_memory
from memory.session_summarizer import summarize_session
from persona.mood_tracker import apply_sentiment_to_mood, get_current_mood, update_mood

# Import the prompt builder for AI responses
from core.prompt_builder import generate_ai_response, get_prompt_debug_info, clear_session_cache

# Preload vectorstore index
from vectorstore import get_store
get_store()

# -- Streamlit Page Config --
st.set_page_config(page_title="🧠 Memory Agent", layout="wide")
st.title("🧠 Your Memory-Driven AI Agent (FAISS + Persona + AI Responses)")

# -- Session Boot & Mood Initialization --
session_file = get_or_create_session_file(st.session_state)
session_id = st.session_state["session_id"]

if "turns" not in st.session_state:
    st.session_state.turns = []

if "session_prompt_printed" not in st.session_state:
    st.session_state.session_prompt_printed = False

if "mood_initialized" not in st.session_state:
    APP_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    PERSONA_DIR = APP_DIR / "persona"
    mood_file = PERSONA_DIR / "mood_adjustments.json"
    if not mood_file.exists():
        try:
            mood_file.parent.mkdir(exist_ok=True)
            default_mood = {
                "current_mood": "neutral",
                "intensity": 0.5,
                "context": {
                    "is_hybrid": False,
                    "is_emergent": False,
                    "stability": "medium"
                },
                "last_updated": "2025-01-01T00:00:00.000000"
            }
            with open(mood_file, "w", encoding="utf-8") as f:
                json.dump(default_mood, f, indent=2)
        except Exception as e:
            print(f"[Mood Init Error]: {e}")
            st.error(f"❌ Failed to initialize mood system: {e}")
    st.session_state.mood_initialized = True

# Load prior history if available
if not st.session_state.turns and Path(session_file).exists():
    try:
        with open(session_file, encoding="utf-8") as f:
            history = json.load(f)
        for turn in history:
            st.session_state.turns.append({"role": "user", "content": turn["user"]})
            st.session_state.turns.append({"role": "assistant", "content": turn["assistant"]})
    except Exception as e:
        st.error(f"❌ Error loading session: {e}")

# Show prior messages
for m in st.session_state.turns:
    render_message(m)

# Optional: Show context blocks
if not st.session_state.session_prompt_printed:
    if st.sidebar.checkbox("Show Context Blocks (Debug)", value=False):
        session_block = build_session_init_prompt(session_id)
        with st.chat_message("assistant"):
            st.markdown("**📋 Session Context Preview:**")
            st.code(session_block, language="markdown")
    st.session_state.session_prompt_printed = True

# ---------------- User Input & Response ----------------
if user_msg := st.chat_input("Type your message…"):
    # Add user message to session
    st.session_state.turns.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # Mood processing with error handling
    try:
        print(f"\n=== MOOD PROCESSING START ===")
        print(f"[App.py]: Processing user input: '{user_msg}'")
        from persona.hormone_api import load_hormone_levels
        hormones_before = load_hormone_levels()
        print(f"[App.py]: Hormones BEFORE processing: {hormones_before}")
        apply_sentiment_to_mood(user_msg)
        hormones_after = load_hormone_levels()
        print(f"[App.py]: Hormones AFTER processing: {hormones_after}")
        if any(abs(hormones_after[h]-hormones_before[h]) > 0.001 for h in hormones_before):
            print("[App.py]: ✅ Hormones successfully updated!")
        else:
            print("[App.py]: ❌ Hormones did NOT change!")
        print(f"[App.py]: Mood processing completed")
        print(f"=== MOOD PROCESSING END ===\n")
    except Exception as e:
        st.error(f"❌ Mood processing error: {e}")
        import traceback
        traceback.print_exc()

    # Generate response via prompt builder, passing correct params
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Build recent conversation history for context
                conv_history = []
                for turn in st.session_state.turns[-10:]:
                    conv_history.append({"role": turn["role"], "content": turn["content"]})

                # 🔧 FIXED: Call prompt builder with correct parameter name
                response_data = generate_ai_response(
                    user_input=user_msg,              # ✅ FIXED: Changed from 'user_text=' to 'user_input='
                    session_id=session_id,
                    conversation_history=conv_history[:-1],  # exclude current user msg for context
                    temperature=0.7,
                    max_tokens=1024
                )

                if response_data["success"]:
                    ai_response = response_data["response"]
                    st.markdown(ai_response)
                    st.session_state.turns.append({"role": "assistant", "content": ai_response})
                    # Save turn
                    save_turn_to_session({"user": user_msg, "assistant": ai_response}, st.session_state)
                else:
                    err_msg = response_data.get("response", "Error in AI response.")
                    st.error(err_msg)
                    st.session_state.turns.append({"role": "assistant", "content": err_msg})
                    save_turn_to_session({"user": user_msg, "assistant": err_msg}, st.session_state)

            except Exception as e:
                st.error(f"❌ Failed to generate response: {e}")
                import traceback
                traceback.print_exc()
                fallback = "I'm sorry, I encountered an error. Please try again."
                st.markdown(fallback)
                st.session_state.turns.append({"role": "assistant", "content": fallback})
                save_turn_to_session({"user": user_msg, "assistant": fallback}, st.session_state)

    # Update FAISS memory state
    try:
        from persona.faiss_memory_writer import update_faiss_memory_state_from_session
        update_faiss_memory_state_from_session(session_id)
    except Exception as e:
        st.warning(f"⚠️ FAISS memory update error: {e}")

    # Update tiny model
    try:
        from persona.tiny_model_writer import update_tiny_model_state_from_session
        update_tiny_model_state_from_session(session_id)
    except Exception as e:
        st.warning(f"⚠️ Tiny model update error: {e}")

# ---------------- Sidebar controls ----------------
st.sidebar.header("🧠 Memory Management")
if st.sidebar.button("End Chat & Save to Long-Term Memory"):
    try:
        with open(session_file, encoding="utf-8") as f:
            data = json.load(f)
        summary = summarize_session(session_id, data)
        st.sidebar.success("✅ Session summarized and stored.")
        st.sidebar.json(summary)
    except Exception as e:
        st.sidebar.error(f"❌ Could not summarize: {e}")

# AI Response Controls
st.sidebar.header("🤖 AI Response Settings")
temperature = st.sidebar.slider("Response Creativity (Temperature)", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Response Length", 256, 2048, 1024)

show_context_preview = st.sidebar.checkbox("Show Context Preview", value=False)
if show_context_preview:
    try:
        debug_info = get_prompt_debug_info("preview", session_id)
        st.sidebar.write(f"**Total prompt length:** {debug_info['total_prompt_length']} chars")
        st.sidebar.write(f"**System message length:** {debug_info['system_message_length']} chars")
        st.sidebar.write(f"**Message count:** {debug_info['messages_count']}")
    except Exception:
        pass

if st.sidebar.button("🔄 Clear Context Cache"):
    from core.prompt_builder import clear_session_cache
    clear_session_cache()
    st.sidebar.info("Context cache cleared.")

# Mood Status
try:
    mood_data = get_current_mood()
    st.sidebar.write(f"**Current Mood:** {mood_data['current_mood']}")
    st.sidebar.write(f"**Intensity:** {mood_data['intensity']:.2f}")
    c = mood_data.get('context', {})
    if c.get('is_hybrid'):
        st.sidebar.write("🔀 **Hybrid State**")
    if c.get('is_emergent'):
        st.sidebar.write("⚡ **Emergent State**")
    st.sidebar.write(f"**Stability:** {c.get('stability', 'medium')}")
    # Hormone levels
    from persona.hormone_api import load_hormone_levels, save_hormone_levels
    hormones = load_hormone_levels()
    with st.sidebar.expander("🧪 Hormone Levels"):
        for h, v in hormones.items():
            color = "🔴" if v > 0.7 else "🔵" if v < 0.3 else "⚪"
            st.sidebar.write(f"{color} {h.title()}: {v:.3f}")
    # Manual buttons
    with st.sidebar.expander("🧪 Manual Mood Testing"):
        if st.button("Test Positive"):
            apply_sentiment_to_mood("I love this!")
            st.experimental_rerun()
        if st.button("Test Negative"):
            apply_sentiment_to_mood("I hate this!")
            st.experimental_rerun()
        if st.button("Reset to Neutral"):
            update_mood("neutral", 0.5, "manual reset")
            save_hormone_levels({"dopamine":0.5, "serotonin":0.5, "cortisol":0.5, "oxytocin":0.5})
            st.experimental_rerun()
except Exception:
    pass

# Debug info
st.sidebar.header("🔧 Debug")
st.sidebar.write(f"Session ID: `{session_id[:8]}`")
st.sidebar.write(f"Turns: {len(st.session_state.get('turns', []))}")
st.sidebar.write(f"Memory turns: {len(load_memory())}")
facts = load_facts()
if facts:
    with st.sidebar.expander("📝 Last Facts"):
        for idx, fact in enumerate(facts[-5:], 1):
            st.sidebar.write(f"{idx}. {fact}")