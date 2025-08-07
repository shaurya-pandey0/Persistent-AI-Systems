# utils/ui_helpers.py (Updated: Renamed to render_message, adjusted to handle single messages without skipping system)
import streamlit as st

def render_message(msg):
    role = msg["role"]
    label = "User:" if role == "user" else "Assistant:" if role == "assistant" else "System:"
    with st.chat_message(role):
        st.markdown(f"**{label}** {msg['content']}")
__all__ = ["render_message"]