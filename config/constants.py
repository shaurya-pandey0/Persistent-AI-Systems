# config/constants.py
from __future__ import annotations

"""
Global configuration constants.

Phase 1: Threshold calibration (WORKING) ✅
Phase 2: FAISS optimization, batch processing, auto-sync (WORKING) ✅
Phase 3: Conversation management, automated retraining, mood persistence (WORKING) ✅
Phase 4: Temporal decay, adaptive baselines, hierarchical memory, compound moods (NEW) 🚀
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────── 🔐 API-related constants ──────────────────────────
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "meta-llama/llama-4-maverick"
MAX_TURNS = 6

# ─────────────────── 📚 FAISS / Memory constants ───────────────────────
TOP_K = 3               # # of chunks to retrieve per query
VECTOR_DIM = 768        # Embedding dimensionality (BGE-M3)
MEMORY_FILE = Path("data/memory.jsonl")

# Hormone / mood persistence
FAISS_MEMORY_JSON = str(Path("persona/faiss_memory_state.json"))

# ─────────────────── 🧠 Phase 1: Threshold configuration (WORKING) ──────
MEMORY_THRESHOLD_CONFIG: dict[str, object] = {
    # Master switch
    "enabled": True,

    # Static default threshold (L2 distance: lower = better similarity)
    # Based on test results: good=0.5982, bad=1.1302, so threshold=1.0 filters noise
    "base_threshold": 1.0,

    # Turn on adaptive thresholding
    "dynamic_adjustment": True,

    # Clamp range for adaptive algorithm (L2 distance bounds)
    "min_threshold": 0.5,   # Very strict (only very similar results)
    "max_threshold": 1.5,   # More lenient (allows somewhat similar results)

    # Context we almost never want to keep
    "trivial_responses": {
        "hi", "hello", "hey", "ok", "okay", "thanks", "thank you",
    },
}

# ─────────────────── 🚀 Phase 2: FAISS optimization configuration ──────
FAISS_OPTIMIZATION_CONFIG = {
    # Index optimization
    "use_ivf_index": True,                    # Enable IndexIVFFlat for speed
    "auto_optimize_threshold": 1000,          # Auto-optimize when >1000 vectors
    "ncentroids": None,                       # Auto-calculate optimal centroids
    
    # Batch processing
    "batch_processing": True,                 # Enable batch embedding processing
    "batch_size": 32,                         # Optimal batch size for BGE-M3
    "max_batch_memory_mb": 512,               # Memory limit for batch processing
    
    # Performance monitoring
    "performance_monitoring": True,           # Track embedding and indexing performance
    "log_optimization_events": True,          # Log when optimizations occur
}

# ─────────────────── 🔄 Phase 2: Auto-sync configuration ───────────────
AUTO_SYNC_CONFIG = {
    # Synchronization behavior
    "enabled": True,                          # Master switch for auto-sync
    "sync_on_dump": True,                     # Sync after each conversation turn
    "sync_threshold": 10,                     # Auto-sync every N turns
    "background_sync": False,                 # Use background threads
    
    # Deduplication
    "dedupe_on_startup": True,                # Remove duplicates at startup
    "dedupe_enabled": True,                   # Enable duplicate detection
    
    # Performance
    "sync_timeout_seconds": 30,               # Timeout for sync operations
    "max_sync_retries": 3,                    # Retry failed syncs
}

# ─────────────────── 📊 Phase 2: Performance monitoring ────────────────
PERFORMANCE_CONFIG = {
    # Latency thresholds (milliseconds)
    "search_latency_threshold": 500,          # Max acceptable search time
    "embedding_latency_threshold": 1000,      # Max acceptable embedding time
    "sync_latency_threshold": 2000,           # Max acceptable sync time
    
    # Quality thresholds
    "precision_threshold": 0.75,              # Minimum precision required
    "relevance_threshold": 0.80,              # Minimum relevance score
    
    # Monitoring behavior
    "enable_performance_tracking": True,      # Track all performance metrics
    "alert_on_degradation": True,             # Send alerts on performance drops
    "degradation_threshold": 0.15,            # 15% performance drop triggers alert
    "performance_history_days": 30,           # Days of performance data to keep
}

# ─────────────────── 🏗️ Phase 3: Conversation management ──────────────
CONVERSATION_MANAGEMENT_CONFIG = {
    # Sliding window management
    "enable_sliding_window": True,            # Enable conversation window management
    "max_conversation_length": 50,            # Maximum turns to keep in active memory
    "sliding_window_size": 30,                # Active conversation window size
    "archive_threshold": 100,                 # Archive conversations older than N turns
    
    # Importance weighting
    "enable_importance_weighting": True,      # Enable importance-based retention
    "emotion_weight_multiplier": 2.0,         # Multiply importance for emotional content
    "user_mention_weight": 1.5,               # Weight for user name mentions
    "question_weight": 1.3,                   # Weight for questions
    "factual_weight": 1.4,                    # Weight for factual information
    
    # Semantic clustering
    "enable_semantic_clustering": True,       # Enable topic-based clustering
    "similarity_cluster_threshold": 0.8,      # Similarity threshold for clustering
    "max_clusters": 10,                       # Maximum number of conversation clusters
    "cluster_merge_threshold": 0.85,          # Threshold for merging similar clusters
    
    # Context management
    "context_relevance_decay": 0.95,          # Decay factor for older context
    "min_context_relevance": 0.3,             # Minimum relevance to keep context
}

# ─────────────────── 🔄 Phase 3: Automated retraining ─────────────────
AUTOMATED_RETRAINING_CONFIG = {
    # Distribution drift monitoring
    "enable_drift_monitoring": True,          # Enable automatic drift detection
    "drift_check_interval": 100,              # Check drift every N new vectors
    "drift_threshold": 0.15,                  # Similarity drift threshold (15%)
    "drift_sample_size": 50,                  # Sample size for drift calculation
    
    # Automatic retraining triggers
    "enable_auto_retraining": True,           # Enable automatic retraining
    "retrain_on_drift": True,                 # Retrain when drift detected
    "retrain_performance_threshold": 0.20,    # Retrain on 20% performance drop
    "max_retraining_frequency": 24,           # Max retraining once per 24 hours
    
    # Index optimization triggers
    "optimize_on_size_growth": True,          # Optimize when index grows significantly
    "size_growth_threshold": 0.30,            # 30% size increase triggers optimization
    "optimize_on_query_pattern_change": True, # Optimize when query patterns change
}

# ─────────────────── 🎭 Phase 3: Mood persistence ─────────────────────
MOOD_PERSISTENCE_CONFIG = {
    # Cross-session persistence
    "enable_cross_session_mood": True,        # Enable mood persistence across sessions
    "mood_decay_hours": 24,                   # Hours before mood starts decaying
    "mood_decay_rate": 0.1,                   # Hourly mood decay rate toward baseline
    "session_break_threshold": 4,             # Hours to consider a new session
    
    # Personality integration
    "enable_personality_integration": True,   # Enable personality-based mood adjustments
    "personality_influence_strength": 0.3,    # How much personality affects mood (0-1)
    "trait_adaptation_rate": 0.05,            # Rate of personality trait adaptation
    
    # Mood history tracking
    "max_mood_history": 1000,                 # Maximum mood history entries
    "mood_pattern_analysis": True,            # Enable mood pattern analysis
    "mood_trend_window": 24,                  # Hours for trend analysis
    
    # Advanced mood features
    "enable_mood_interpolation": True,        # Smooth mood transitions
    "mood_stability_factor": 0.8,             # Resistance to rapid mood changes
    "complex_emotion_detection": True,        # Enable complex emotion states
}

# ─────────────────── ⏰ Phase 4: Temporal decay models ─────────────────
TEMPORAL_DECAY_CONFIG = {
    # Hormone decay parameters
    "enable_temporal_decay": True,            # Enable temporal hormone decay
    "hormone_half_lives": {                   # Half-life in hours for each hormone
        "dopamine": 2.0,                      # Dopamine decays quickly (2 hours)
        "serotonin": 6.0,                     # Serotonin moderate decay (6 hours)
        "cortisol": 4.0,                      # Cortisol moderate-fast decay (4 hours)
        "oxytocin": 3.0,                      # Oxytocin moderate decay (3 hours)
    },
    
    # Decay calculation settings
    "decay_update_interval_minutes": 15,     # Update decay every 15 minutes
    "min_decay_threshold": 0.01,             # Minimum change before decay stops
    "decay_baseline_target": 0.5,            # Target baseline for decay
    
    # Advanced decay features
    "enable_circadian_modulation": True,     # Modulate decay by circadian rhythms
    "enable_interaction_boost": True,        # Boost decay resistance during interactions
    "interaction_boost_factor": 0.5,         # Reduce decay by 50% during interactions
    "max_decay_history": 1000,               # Maximum decay history entries
}

# ─────────────────── 🔄 Phase 4: Adaptive baseline system ─────────────
ADAPTIVE_BASELINE_CONFIG = {
    # User-specific baseline adaptation
    "enable_adaptive_baselines": True,       # Enable per-user baseline adaptation
    "baseline_adaptation_rate": 0.02,        # Rate of baseline learning (2% per update)
    "baseline_stability_period": 72,         # Hours to stabilize new baselines
    "min_baseline_samples": 50,              # Minimum interactions before adaptation
    
    # Circadian rhythm integration
    "enable_circadian_rhythms": True,        # Enable circadian rhythm effects
    "circadian_amplitude": 0.15,             # Maximum circadian variation (±15%)
    "peak_energy_hour": 14,                  # Hour of peak energy (2 PM)
    "low_energy_hour": 3,                    # Hour of lowest energy (3 AM)
    
    # Multi-modal context integration
    "enable_context_adaptation": True,       # Enable context-based adaptation
    "context_influence_strength": 0.2,       # Context influence on baselines (20%)
    "supported_contexts": [                  # Supported contextual factors
        "work", "social", "creative", "rest", "exercise", "stress"
    ],
    
    # Baseline personalization
    "personality_baseline_influence": 0.4,   # Personality influence on baselines (40%)
    "user_feedback_weight": 0.3,             # Weight of user feedback in adaptation
    "environmental_factor_weight": 0.1,      # Weight of environmental factors
}

# ─────────────────── 🏗️ Phase 4: Hierarchical memory system ────────────
HIERARCHICAL_MEMORY_CONFIG = {
    # Memory tier configuration
    "enable_hierarchical_memory": True,      # Enable multi-tier memory system
    "working_memory_size": 20,               # Number of items in working memory
    "short_term_memory_size": 200,           # Number of items in short-term memory
    "long_term_memory_threshold": 0.7,       # Importance threshold for long-term storage
    
    # Memory type separation
    "enable_episodic_semantic_separation": True,  # Separate episodic vs semantic memory
    "episodic_memory_weight": 0.8,           # Weight for episodic memories
    "semantic_memory_weight": 1.2,           # Weight for semantic memories (facts, knowledge)
    "procedural_memory_weight": 0.9,         # Weight for procedural memories (how-to)
    
    # Memory consolidation
    "enable_memory_consolidation": True,     # Enable memory consolidation process
    "consolidation_interval_hours": 12,      # Hours between consolidation cycles
    "consolidation_threshold": 0.6,          # Minimum importance for consolidation
    "memory_interference_factor": 0.1,       # Factor for memory interference effects
    
    # Advanced memory features
    "enable_associative_memory": True,       # Enable associative memory links
    "association_strength_threshold": 0.5,   # Minimum strength for memory associations
    "max_associations_per_memory": 10,       # Maximum associations per memory item
    "memory_rehearsal_boost": 1.5,           # Boost factor for rehearsed memories
}

# ─────────────────── 🎭 Phase 4: Compound mood system ─────────────────
COMPOUND_MOOD_CONFIG = {
    # Vector mood representation
    "enable_vector_moods": True,             # Enable multi-dimensional mood vectors
    "mood_vector_dimensions": [              # Mood vector dimensions
        "valence",      # Positive vs negative (happiness-sadness axis)
        "arousal",      # High vs low energy (excitement-calm axis)  
        "dominance",    # Control vs submission (confidence-anxiety axis)
        "sociability",  # Social vs solitary tendencies
        "creativity",   # Creative vs analytical mindset
        "focus"         # Focused vs scattered attention
    ],
    
    # Mood transition modeling
    "enable_mood_transitions": True,         # Enable mood transition tracking
    "transition_smoothing_factor": 0.7,      # Smoothing factor for mood transitions
    "max_transition_speed": 0.3,             # Maximum mood change per update
    "transition_history_length": 100,        # Number of transitions to track
    
    # Complex mood states
    "enable_mixed_moods": True,              # Enable mixed emotional states
    "mood_conflict_threshold": 0.4,          # Threshold for conflicting moods
    "dominant_mood_threshold": 0.6,          # Threshold for mood dominance
    "mood_stability_tracking": True,         # Track mood stability over time
    
    # Mood prediction and modeling
    "enable_mood_prediction": True,          # Enable mood state prediction
    "prediction_horizon_hours": 6,           # Hours ahead to predict mood
    "mood_pattern_learning": True,           # Learn user mood patterns
    "pattern_learning_rate": 0.05,           # Rate of pattern learning
}

def _validate_threshold_config(cfg: dict[str, object]) -> None:
    """Fail fast if a bad override sneaks in via env-vars or runtime edit."""
    if not (0.0 < cfg["min_threshold"] <= cfg["base_threshold"] <= cfg["max_threshold"]):
        raise ValueError(
            "MEMORY_THRESHOLD_CONFIG values are inconsistent: "
            f"{cfg['min_threshold']=}, {cfg['base_threshold']=}, {cfg['max_threshold']=}"
        )

def _validate_faiss_config(cfg: dict[str, object]) -> None:
    """Validate FAISS optimization configuration."""
    if cfg["batch_size"] < 1 or cfg["batch_size"] > 256:
        raise ValueError(f"Invalid batch_size: {cfg['batch_size']} (must be 1-256)")
    
    if cfg["auto_optimize_threshold"] < 100:
        raise ValueError(f"auto_optimize_threshold too low: {cfg['auto_optimize_threshold']} (min: 100)")

def _validate_performance_config(cfg: dict[str, object]) -> None:
    """Validate performance monitoring configuration."""
    if not (0.0 < cfg["degradation_threshold"] < 1.0):
        raise ValueError(f"Invalid degradation_threshold: {cfg['degradation_threshold']} (must be 0-1)")

def _validate_conversation_config(cfg: dict[str, object]) -> None:
    """Phase 3: Validate conversation management configuration."""
    if cfg["max_conversation_length"] < cfg["sliding_window_size"]:
        raise ValueError("max_conversation_length must be >= sliding_window_size")
    
    if not (0.0 < cfg["similarity_cluster_threshold"] <= 1.0):
        raise ValueError("similarity_cluster_threshold must be between 0 and 1")

def _validate_retraining_config(cfg: dict[str, object]) -> None:
    """Phase 3: Validate automated retraining configuration."""
    if not (0.0 < cfg["drift_threshold"] < 1.0):
        raise ValueError("drift_threshold must be between 0 and 1")
    
    if cfg["drift_sample_size"] < 10:
        raise ValueError("drift_sample_size must be at least 10")

def _validate_mood_config(cfg: dict[str, object]) -> None:
    """Phase 3: Validate mood persistence configuration."""
    if not (0.0 < cfg["personality_influence_strength"] <= 1.0):
        raise ValueError("personality_influence_strength must be between 0 and 1")
    
    if cfg["mood_decay_hours"] < 1:
        raise ValueError("mood_decay_hours must be at least 1")

def _validate_temporal_decay_config(cfg: dict[str, object]) -> None:
    """Phase 4: Validate temporal decay configuration."""
    for hormone, half_life in cfg["hormone_half_lives"].items():
        if half_life <= 0:
            raise ValueError(f"Invalid half_life for {hormone}: {half_life} (must be > 0)")
    
    if cfg["decay_update_interval_minutes"] < 1:
        raise ValueError("decay_update_interval_minutes must be at least 1")

def _validate_adaptive_config(cfg: dict[str, object]) -> None:
    """Phase 4: Validate adaptive baseline configuration."""
    if not (0.0 < cfg["baseline_adaptation_rate"] < 1.0):
        raise ValueError("baseline_adaptation_rate must be between 0 and 1")
    
    if not (0.0 < cfg["circadian_amplitude"] < 1.0):
        raise ValueError("circadian_amplitude must be between 0 and 1")

def _validate_hierarchical_config(cfg: dict[str, object]) -> None:
    """Phase 4: Validate hierarchical memory configuration."""
    if cfg["working_memory_size"] <= 0:
        raise ValueError("working_memory_size must be positive")
    
    if cfg["short_term_memory_size"] <= cfg["working_memory_size"]:
        raise ValueError("short_term_memory_size must be > working_memory_size")

def _validate_compound_mood_config(cfg: dict[str, object]) -> None:
    """Phase 4: Validate compound mood configuration."""
    if len(cfg["mood_vector_dimensions"]) < 2:
        raise ValueError("Must have at least 2 mood vector dimensions")
    
    if not (0.0 < cfg["transition_smoothing_factor"] <= 1.0):
        raise ValueError("transition_smoothing_factor must be between 0 and 1")

# Validate all configurations on import
_validate_threshold_config(MEMORY_THRESHOLD_CONFIG)
_validate_faiss_config(FAISS_OPTIMIZATION_CONFIG)  
_validate_performance_config(PERFORMANCE_CONFIG)
_validate_conversation_config(CONVERSATION_MANAGEMENT_CONFIG)
_validate_retraining_config(AUTOMATED_RETRAINING_CONFIG)
_validate_mood_config(MOOD_PERSISTENCE_CONFIG)
_validate_temporal_decay_config(TEMPORAL_DECAY_CONFIG)
_validate_adaptive_config(ADAPTIVE_BASELINE_CONFIG)
_validate_hierarchical_config(HIERARCHICAL_MEMORY_CONFIG)
_validate_compound_mood_config(COMPOUND_MOOD_CONFIG)

# ─────────────────── 🔥 Safety check ───────────────────────────────────
if not API_KEY:
    print("⚠️  Warning: OPENROUTER_API_KEY not found in .env file")

print("✅ Phase 4 configuration loaded and validated")
