"""
core/context_block_builder.py

Builds the two layered prompt blocks REQUIRED by the specification:
1. Session-level injection → printed once at session start
2. Turn-level injection → printed every user turn

Nothing here writes to disk; it only assembles strings.
"""

import json
from pathlib import Path
from typing import List, Tuple

from memory.long_term_memory import load_long_term_memory
from memory.context_retriever import retrieve_top_memories
from persona.mood_tracker import get_current_mood, get_mood_summary
from persona.hormone_api import load_hormone_levels # Use neutral API
from persona.relationship_status import get_relationship_summary

# ------------------------------------------------------------------ #
# Helpers to load stable persona & personality traits
# ------------------------------------------------------------------ #

def _load_persona_traits():
    with open("persona/persona.json", encoding="utf-8") as f:
        persona = json.load(f)
    with open("persona/personality.json", encoding="utf-8") as f:
        personality_data = json.load(f)
    
    # Extract personality traits from Phase 4 structure
    traits = personality_data.get("personality_traits", {})
    
    # If traits is empty, fallback to old flat structure for backward compatibility
    if not traits:
        traits = personality_data
    
    return persona, traits

# ------------------------------------------------------------------ #
# Public builders
# ------------------------------------------------------------------ #

def build_session_init_prompt(user_id: str) -> str:
    """Create the SESSION-LEVEL block (print once)."""
    persona, traits = _load_persona_traits()
    relationship = get_relationship_summary(user_id) or "No relationship data yet."
    
    # Top-3 long-term memories (if any)
    ltm = [e["summary"] for e in load_long_term_memory(user_id) if e.get("summary")]
    long_term_block = "\n- ".join(ltm[:3]) if ltm else "None yet."
    
    prompt = f"""🔰 [SESSION-LEVEL INJECTION] (Sent once at session start)

[PERSONA INFO]
Name: {persona["name"]}
Identity: {persona["identity"]}
Goals: {", ".join(persona["goals"])}
Pronouns: {persona["pronouns"]}

[PERSONALITY]
Tone: {traits.get("tone", "friendly")}
Style: {traits.get("style", "helpful")}
Temperament: {traits.get("temperament", "balanced")}
Formality: {traits.get("formality", "moderate")}

[RELATIONSHIP STATUS]
{relationship}

[LONG-TERM MEMORIES]
- {long_term_block}

→ [END INIT BLOCK]
"""
    return prompt

def build_turn_prompt(
    user_input: str,
    user_id: str,
    k_short: int = 3,
    k_long: int = 2,
) -> str:
    """Create the TURN-LEVEL block for every user message."""
    print(f"[Context Builder]: Reading current mood data (mood already processed in app.py)")
    
    # Load persona and get current mood data (already updated by app.py)
    persona, traits = _load_persona_traits()
    
    # Get enhanced mood data with full context (already updated)
    mood = get_current_mood()
    hormone_levels = load_hormone_levels()
    
    print(f"[Context Builder]: Current mood: {mood['current_mood']} ({mood['intensity']:.2f})")
    print(f"[Context Builder]: Hormone levels: {hormone_levels}")
    
    # Extract mood context information
    mood_context = mood.get("context", {})
    is_hybrid = mood_context.get("is_hybrid", False)
    is_emergent = mood_context.get("is_emergent", False)
    stability = mood_context.get("stability", "medium")
    
    # Format hormone levels for display
    hormone_str = f"dopamine={hormone_levels.get('dopamine', 0.5):.2f}, " \
                  f"serotonin={hormone_levels.get('serotonin', 0.5):.2f}, " \
                  f"oxytocin={hormone_levels.get('oxytocin', 0.5):.2f}, " \
                  f"cortisol={hormone_levels.get('cortisol', 0.5):.2f}"
    
    # Create mood context summary
    mood_flags = []
    if is_hybrid:
        mood_flags.append("HYBRID")
    if is_emergent:
        mood_flags.append("EMERGENT")
    mood_type_str = f" [{'/'.join(mood_flags)}]" if mood_flags else ""
    
    # Generate mood context description
    if is_hybrid:
        mood_context_desc = f"Hybrid emotional state combining multiple mood influences"
    elif is_emergent:
        mood_context_desc = f"Emergent emotional state arising from unique hormone combinations"
    else:
        mood_context_desc = f"Standard emotional state mapped from current hormone levels"
    
    # Fetch combined short + long memories (already deduped)
    short_rels, long_rels = retrieve_top_memories(
        user_input, k_short=k_short, k_long=k_long, user_id=user_id
    )
    
    def _mk_bullet_list(lines: List[str]) -> str:
        return "\n".join(lines) if lines else "None."
    
    # Build the enhanced prompt string
    prompt = f"""🔁 [TURN-LEVEL INJECTION] (Sent every user message)

[MINDSET]
[MINDSET]
Isabella’s mood right now is: {mood['current_mood'].replace('_', ' ')}{mood_type_str}, and it colors the way she talks, reacts, and engages.

[EMOTIONAL CONTEXT]
Isabella is currently feeling {mood['current_mood'].replace('_', ' ')} with an emotional intensity of {mood['intensity']:.2f}.
She doesn't consciously recognize hybrid or emergent emotional dynamics, but her behavior may subtly reflect them.
She's currently in a {stability.lower()} emotional phase.

[RELEVANT MEMORY]
{_mk_bullet_list(short_rels + long_rels)}

User: {user_input}

→ [GENERATE RESPONSE TO LAST USER MESSAGE]
"""
    return prompt

def get_detailed_mood_status() -> str:
    """
    Helper function to get a detailed mood status summary.
    Can be used for debugging or detailed mood analysis.
    """
    from persona.mood_tracker import get_mood_summary
    
    summary = get_mood_summary()
    current_state = summary["current_state"]
    hormones = summary["hormone_levels"]
    patterns = summary["recent_patterns"]
    
    status = f"""=== DETAILED MOOD STATUS ===
Current Mood: {current_state['current_mood']} (Intensity: {current_state['intensity']:.2f})
Mood Type: {'Hybrid' if current_state['context']['is_hybrid'] else 'Emergent' if current_state['context']['is_emergent'] else 'Standard'}
Stability: {current_state['context']['stability']}

Hormone Levels:
  Dopamine: {hormones['dopamine']:.2f}
  Serotonin: {hormones['serotonin']:.2f}
  Oxytocin: {hormones['oxytocin']:.2f}
  Cortisol: {hormones['cortisol']:.2f}

Recent Activity:
  Recent Moods: {', '.join(patterns['recent_moods'][-5:])}
  Hybrid States: {patterns['hybrid_states_count']}/20 recent
  Emergent States: {patterns['emergent_states_count']}/20 recent
  Mood Volatility: {summary['complexity_indicators']['mood_volatility']}

Last Updated: {current_state.get('last_updated', 'Unknown')}
"""
    return status
