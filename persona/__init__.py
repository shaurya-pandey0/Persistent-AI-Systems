# persona/__init__.py

"""
Persona module for AI personality, mood tracking, and hormone management.

This __init__.py file makes the persona directory a proper Python package,
allowing imports between modules like mood_tracker.py and hormone_api.py.

Phase 4 Features:
- Enhanced mood tracking with compound mood vectors
- Cross-session mood persistence
- Personality integration
- Advanced hormone management
- Temporal decay models
- Hierarchical memory system
"""

# Version information
__version__ = "4.0.0"
__author__ = "AI Development Team"

# Make key functions available at package level
try:
    from .hormone_api import (
        load_hormone_levels,
        save_hormone_levels,
        infer_mood_from_hormones,
        get_mood_context
    )
    
    from .mood_tracker import (
        update_mood_from_hormones,
        get_current_mood,
        get_enhanced_mood_summary,
        calculate_mood_from_hormones
    )
    
    __all__ = [
        'load_hormone_levels',
        'save_hormone_levels', 
        'infer_mood_from_hormones',
        'get_mood_context',
        'update_mood_from_hormones',
        'get_current_mood',
        'get_enhanced_mood_summary',
        'calculate_mood_from_hormones'
    ]
    
except ImportError as e:
    # If imports fail, log but don't crash
    print(f"[Persona Package Warning]: Some imports failed - {e}")
    __all__ = []

# Package metadata
PACKAGE_INFO = {
    "name": "persona",
    "version": __version__,
    "description": "AI personality and mood tracking system",
    "features": [
        "compound_mood_vectors",
        "cross_session_persistence", 
        "personality_integration",
        "temporal_decay_models",
        "hierarchical_memory"
    ]
}