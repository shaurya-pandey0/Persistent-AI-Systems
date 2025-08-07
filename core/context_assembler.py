"""
core/context_assembler.py
Thin wrapper kept for backward-compat calls.
Now delegates to context_block_builder.
"""
from core.context_block_builder import (
    build_session_init_prompt,
    build_turn_prompt,
)

__all__ = ["build_session_init_prompt", "build_turn_prompt"]
