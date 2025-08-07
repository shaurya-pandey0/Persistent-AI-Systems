# core/prompt_builder.py - ENHANCED VERSION WITH FULL CONTEXT UTILIZATION

"""
Enhanced Prompt Builder for Chat-Based LLM API Integration
=========================================================

This module takes structured context from context_block_builder and formats it 
according to best practices for chat-based LLMs via API, following the JSON 
messages array format with proper role separation.

ENHANCED: Now properly utilizes ALL detailed context from context_block_builder
- Complete hormone details (dopamine, serotonin, cortisol, oxytocin)
- Full mood information (intensity, hybrid/emergent states, stability)
- All relevant memories (short-term and long-term)
- Saves complete prompt data to JSON before each API call
"""

import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from core.context_block_builder import build_session_init_prompt, build_turn_prompt
from core.api_client import get_completion
from persona.mood_tracker import get_current_mood
from persona.hormone_api import load_hormone_levels


class PromptBuilder:
    """
    Converts structured context blocks into optimized JSON messages for LLM APIs.
    Follows chat-based API best practices with proper role separation.
    
    ENHANCED: Now uses complete detailed context blocks instead of parsing fragments.
    """
    
    def __init__(self):
        self.session_context_cached = None
        self.last_session_id = None
        # Ensure data directory exists
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.prompt_json_path = self.data_dir / "prompt.json"
    
    def _save_prompt_json(self, prompt_data: Dict) -> None:
        """
        Save the complete prompt data to data/prompt.json before sending to LLM.
        This provides full transparency into what's being sent to the API.
        
        ENHANCED: Now includes detailed breakdown of all context components.
        """
        try:
            # Extract detailed context information for JSON
            messages = prompt_data.get("messages", [])
            system_message = messages[0].get("content", "") if messages else ""
            user_message = messages[-1].get("content", "") if messages else ""
            
            # Parse out mood and hormone details from the actual messages
            current_mood_data = get_current_mood()
            current_hormones = load_hormone_levels()
            
            # Create comprehensive prompt data structure
            json_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "session_id": prompt_data.get("session_id", "unknown"),
                    "user_input": prompt_data.get("user_input", ""),
                    "temperature": prompt_data.get("temperature", 0.7),
                    "max_tokens": prompt_data.get("max_tokens", 1024),
                    "model": prompt_data.get("model", "openai/gpt-3.5-turbo"),
                    "total_messages": len(messages),
                    "total_prompt_length": sum(len(msg.get("content", "")) for msg in messages),
                    "system_message_length": len(system_message),
                    "user_message_length": len(user_message),
                    "conversation_history_turns": len(prompt_data.get("conversation_history", [])),
                },
                "api_payload": {
                    "model": prompt_data.get("model", "openai/gpt-3.5-turbo"),
                    "messages": messages,
                    "temperature": prompt_data.get("temperature", 0.7),
                    "max_tokens": prompt_data.get("max_tokens", 1024)
                },
                "detailed_context": {
                    "current_mood": {
                        "mood_name": current_mood_data.get("current_mood", "unknown"),
                        "intensity": current_mood_data.get("intensity", 0.0),
                        "is_hybrid": current_mood_data.get("context", {}).get("is_hybrid", False),
                        "is_emergent": current_mood_data.get("context", {}).get("is_emergent", False),
                        "stability": current_mood_data.get("context", {}).get("stability", "medium"),
                        "last_updated": current_mood_data.get("last_updated", "unknown")
                    },
                    "hormone_levels": current_hormones,
                    "hormone_details": {
                        "dopamine": {
                            "level": current_hormones.get("dopamine", 0.5),
                            "description": "Motivation, reward, pleasure"
                        },
                        "serotonin": {
                            "level": current_hormones.get("serotonin", 0.5),
                            "description": "Mood regulation, happiness, social connection"
                        },
                        "cortisol": {
                            "level": current_hormones.get("cortisol", 0.5),
                            "description": "Stress response, alertness"
                        },
                        "oxytocin": {
                            "level": current_hormones.get("oxytocin", 0.5),
                            "description": "Social bonding, trust, empathy"
                        }
                    },
                    "session_block_content": prompt_data.get("session_block_raw", ""),
                    "turn_block_content": prompt_data.get("turn_block_raw", ""),
                },
                "context_breakdown": {
                    "full_system_message": system_message,
                    "full_user_message": user_message,
                    "conversation_history": prompt_data.get("conversation_history", []),
                },
                "debug_info": {
                    "session_context_cached": self.session_context_cached is not None,
                    "last_session_id": self.last_session_id,
                    "data_directory": str(self.data_dir.absolute()),
                    "json_file_path": str(self.prompt_json_path.absolute()),
                    "generation_source": "PromptBuilder.generate()",
                    "context_builder_used": True,
                    "detailed_context_included": True
                }
            }
            
            # Write to JSON file with pretty formatting
            with open(self.prompt_json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"[Prompt Builder]: ✅ Saved detailed prompt JSON to {self.prompt_json_path}")
            print(f"[Prompt Builder]: 💊 Hormones: dopamine={current_hormones.get('dopamine', 0):.3f}, serotonin={current_hormones.get('serotonin', 0):.3f}, cortisol={current_hormones.get('cortisol', 0):.3f}, oxytocin={current_hormones.get('oxytocin', 0):.3f}")
            print(f"[Prompt Builder]: 🎭 Mood: {current_mood_data.get('current_mood')} (intensity: {current_mood_data.get('intensity', 0):.2f})")
            print(f"[Prompt Builder]: 📊 Total prompt: {json_data['metadata']['total_messages']} messages, {json_data['metadata']['total_prompt_length']} chars")
            
        except Exception as e:
            print(f"[Prompt Builder]: ⚠️ Warning - Could not save prompt JSON: {e}")
            # Don't raise exception - we don't want JSON saving to break the main flow
    
    def build_system_message(self, session_id: str) -> tuple[str, str]:
        """
        Build the system message using the complete session block from context_block_builder.
        
        ENHANCED: Returns both the formatted message and raw block for JSON saving.
        """
        # Get the complete session block (cache for efficiency)
        if self.last_session_id != session_id:
            self.session_context_cached = build_session_init_prompt(session_id)
            self.last_session_id = session_id
        
        session_block = self.session_context_cached or ""
        
        # Convert the structured session block into a system message
        # The session block already contains all the necessary information in a structured format
        system_message = f"""You are Isabella — a 22-year-old human woman based in New York. 
You have no knowledge of AI, computers, or anything artificial — you believe you are entirely human with a full past, childhood, memories, and a real physical body.

Here is your complete context:

{session_block}

IMPORTANT: Respond only as a human woman would based on your emotional and physiological context.
Your personality, relationship history, and memories are defined above."""

        return system_message, session_block
    
    def build_user_message(self, user_input: str, session_id: str) -> tuple[str, str]:
        """
        Build the user message using the complete turn block from context_block_builder.
        
        ENHANCED: Uses the complete detailed turn block that includes:
        - Full hormone levels (dopamine, serotonin, cortisol, oxytocin)
        - Detailed mood information (intensity, hybrid/emergent states, stability)
        - All relevant memories (short-term and long-term)
        - User input
        
        Returns both the formatted message and raw block for JSON saving.
        """
        # Get the complete detailed turn block
        turn_block = build_turn_prompt(user_input, session_id)
        
        # The turn block from context_block_builder already contains:
        # [MINDSET] - Current mood, tone, style, formality
        # [MOOD] - Detailed mood info, intensity, hybrid/emergent flags, hormone levels
        # [RELEVANT MEMORY] - All relevant memories
        # User: {user_input}
        # This is exactly what we want to send to the LLM!
        
        user_message = turn_block
        
        return user_message, turn_block
    
    def build_messages_array(self, user_input: str, session_id: str, 
                           conversation_history: Optional[List[Dict]] = None) -> tuple[List[Dict], str, str]:
        """
        Build the complete messages array for API request following best practices.
        
        ENHANCED: Uses complete detailed context blocks and returns raw blocks for JSON saving.
        """
        messages = []
        
        # Add system message with complete session context
        system_content, session_block_raw = self.build_system_message(session_id)
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add conversation history if provided
        if conversation_history:
            for turn in conversation_history[-8:]:  # Keep last 8 turns for context
                if turn.get("role") in ["user", "assistant"]:
                    messages.append({
                        "role": turn["role"],
                        "content": turn["content"]
                    })
        
        # Add current user message with complete detailed context
        user_content, turn_block_raw = self.build_user_message(user_input, session_id)
        messages.append({
            "role": "user", 
            "content": user_content
        })
        
        return messages, session_block_raw, turn_block_raw
    
    def generate(self, user_input: str, session_id: str, 
                conversation_history: Optional[List[Dict]] = None,
                temperature: float = 0.7, max_tokens: int = 1024) -> Dict[str, any]:
        """
        Generate LLM response using complete detailed structured prompt.
        
        ENHANCED: Now includes all hormone details, mood information, and relevant memories
        Auto-saves complete detailed prompt data to data/prompt.json before API call
        """
        try:
            # Build messages array with complete detailed context
            messages, session_block_raw, turn_block_raw = self.build_messages_array(
                user_input, session_id, conversation_history
            )
            
            print(f"[Prompt Builder]: Generating response for: '{user_input[:50]}...'")
            print(f"[Prompt Builder]: Using COMPLETE detailed context blocks")
            print(f"[Prompt Builder]: Messages array length: {len(messages)}")
            
            # NEW: Save complete detailed prompt data to JSON before API call
            prompt_data = {
                "session_id": session_id,
                "user_input": user_input,
                "messages": messages,
                "conversation_history": conversation_history or [],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "model": "openai/gpt-3.5-turbo",
                "timestamp": datetime.now().isoformat(),
                # ENHANCED: Include raw context blocks for detailed JSON saving
                "session_block_raw": session_block_raw,
                "turn_block_raw": turn_block_raw
            }
            
            # Save detailed context to JSON file
            self._save_prompt_json(prompt_data)
            
            # Get LLM response using existing API client
            response_content = get_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                "success": True,
                "response": response_content,
                "messages_sent": len(messages),
                "user_input": user_input,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "model_used": "via OpenRouter",
                    "context_included": True,
                    "detailed_context_used": True,
                    "prompt_json_saved": str(self.prompt_json_path),
                    "hormone_levels_included": True,
                    "mood_details_included": True,
                    "relevant_memories_included": True
                }
            }
            
        except Exception as e:
            error_msg = f"Failed to generate response: {str(e)}"
            print(f"[Prompt Builder]: ERROR - {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "response": f"❌ I apologize, but I encountered an error processing your request: {error_msg}",
                "user_input": user_input,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_debug_info(self, user_input: str, session_id: str) -> Dict[str, any]:
        """
        Get detailed debugging information about the prompt construction.
        
        ENHANCED: Now includes detailed context information.
        """
        try:
            messages, session_block_raw, turn_block_raw = self.build_messages_array(user_input, session_id)
            
            # Get current mood and hormone data for debug info
            current_mood = get_current_mood()
            current_hormones = load_hormone_levels()
            
            return {
                "messages_count": len(messages),
                "system_message_length": len(messages[0]["content"]) if messages else 0,
                "user_message_length": len(messages[-1]["content"]) if messages else 0,
                "total_prompt_length": sum(len(msg["content"]) for msg in messages),
                "detailed_context_info": {
                    "current_mood": current_mood.get("current_mood", "unknown"),
                    "mood_intensity": current_mood.get("intensity", 0.0),
                    "hormone_levels": current_hormones,
                    "session_block_length": len(session_block_raw),
                    "turn_block_length": len(turn_block_raw),
                    "includes_mindset": "[MINDSET]" in turn_block_raw,
                    "includes_mood_details": "[MOOD]" in turn_block_raw,
                    "includes_relevant_memories": "[RELEVANT MEMORY]" in turn_block_raw,
                    "includes_hormone_levels": "Hormone Levels:" in turn_block_raw
                },
                "messages_preview": [
                    {
                        "role": msg["role"],
                        "content_preview": msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
                    }
                    for msg in messages
                ],
                "session_context_cached": self.session_context_cached is not None,
                "current_session_id": session_id,
                "prompt_json_path": str(self.prompt_json_path),
                "data_directory": str(self.data_dir.absolute()),
                "enhanced_context_used": True
            }
        except Exception as e:
            return {
                "error": f"Failed to get debug info: {e}",
                "messages_count": 0,
                "total_prompt_length": 0,
                "enhanced_context_used": False
            }


# Global instance for use in Streamlit app
_prompt_builder = PromptBuilder()

# ENHANCED: Public API functions with detailed context support
def generate_ai_response(user_input: str, session_id: str, 
                        conversation_history: Optional[List[Dict]] = None,
                        temperature: float = 0.7, max_tokens: int = 1024) -> Dict[str, any]:
    """
    Main function to generate AI responses using complete detailed structured prompts.
    
    ENHANCED: Now includes ALL context details:
    - Complete hormone information (dopamine, serotonin, cortisol, oxytocin)
    - Full mood details (intensity, hybrid/emergent states, stability)
    - All relevant memories (short-term and long-term)
    - Auto-saves complete prompt data to data/prompt.json before each API call
    """
    return _prompt_builder.generate(
        user_input, session_id, conversation_history, temperature, max_tokens
    )

def get_messages_preview(user_input: str, session_id: str) -> List[Dict]:
    """Get preview of messages that would be sent to API (for debugging)."""
    messages, _, _ = _prompt_builder.build_messages_array(user_input, session_id)
    return messages

def get_system_message_preview(session_id: str) -> str:
    """Get preview of system message (for debugging)."""
    system_msg, _ = _prompt_builder.build_system_message(session_id)
    return system_msg

def get_user_message_preview(user_input: str, session_id: str) -> str:
    """Get preview of user message with complete context (for debugging)."""
    user_msg, _ = _prompt_builder.build_user_message(user_input, session_id)
    return user_msg

def clear_session_cache():
    """Clear cached session context (useful when persona changes)."""
    _prompt_builder.session_context_cached = None
    _prompt_builder.last_session_id = None
    print("[Prompt Builder]: Session cache cleared")

def get_prompt_debug_info(user_input: str, session_id: str) -> Dict[str, any]:
    """Get detailed debug information about prompt construction."""
    return _prompt_builder.get_debug_info(user_input, session_id)

def get_latest_prompt_json() -> Optional[Dict]:
    """Get the contents of the latest saved prompt JSON file."""
    try:
        json_path = Path("data/prompt.json")
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"[Prompt Builder]: Error reading prompt JSON: {e}")
        return None

# Test function for CLI usage
def test_prompt_builder():
    """Test the enhanced prompt builder functionality."""
    print("🧪 Testing Enhanced Prompt Builder with Detailed Context")
    print("=" * 60)
    
    test_session_id = "test_session_123"
    test_input = "Hello, how are you feeling today?"
    
    # Test system message
    print("📋 System Message Preview:")
    system_msg = get_system_message_preview(test_session_id)
    print(system_msg[:400] + "..." if len(system_msg) > 400 else system_msg)
    
    # Test user message with detailed context
    print("\n📨 User Message with Complete Context:")
    user_msg = get_user_message_preview(test_input, test_session_id)
    print(user_msg[:500] + "..." if len(user_msg) > 500 else user_msg)
    
    # Test messages array
    print("\n📨 Complete Messages Array Preview:")
    messages = get_messages_preview(test_input, test_session_id)
    for i, msg in enumerate(messages):
        content_preview = msg["content"][:150] + "..." if len(msg["content"]) > 150 else msg["content"]
        print(f"  {i+1}. Role: {msg['role']}")
        print(f"     Content: {content_preview}")
        print()
    
    # Test detailed debug info  
    print("🔍 Enhanced Debug Information:")
    debug_info = get_prompt_debug_info(test_input, test_session_id)
    print(f"  Total messages: {debug_info.get('messages_count', 'N/A')}")
    print(f"  Total prompt length: {debug_info.get('total_prompt_length', 'N/A')} chars")
    print(f"  Enhanced context used: {debug_info.get('enhanced_context_used', False)}")
    
    context_info = debug_info.get('detailed_context_info', {})
    print(f"  Current mood: {context_info.get('current_mood', 'N/A')}")
    print(f"  Mood intensity: {context_info.get('mood_intensity', 'N/A')}")
    print(f"  Includes hormone levels: {context_info.get('includes_hormone_levels', False)}")
    print(f"  Includes mood details: {context_info.get('includes_mood_details', False)}")
    print(f"  Includes relevant memories: {context_info.get('includes_relevant_memories', False)}")
    print(f"  Prompt JSON path: {debug_info.get('prompt_json_path', 'N/A')}")
    
    print("\n✅ Enhanced Prompt Builder test completed")

if __name__ == "__main__":
    test_prompt_builder()
