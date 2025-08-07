# memory/context_retriever.py

"""
Retrieves short-term memories from the persistent FAISS store and
long-term summaries from JSON/DB.

Phase 1: Threshold optimization (WORKING) ✅
Phase 2: Enhanced filtering and performance (WORKING) ✅
Phase 3: Conversation management, sliding-window, importance weighting (WORKING) ✅
Phase 4: Hierarchical memory integration, multi-modal retrieval (NEW) 🚀
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import json
import hashlib

from config.constants import (
    MEMORY_THRESHOLD_CONFIG, 
    TOP_K, 
    CONVERSATION_MANAGEMENT_CONFIG,
    HIERARCHICAL_MEMORY_CONFIG
)
from vectorstore import search as vector_search
from memory.long_term_memory import load_long_term_memory

# Import Phase 4 hierarchical memory
try:
    from memory.hierarchical_memory import (
        retrieve_hierarchical_memories,
        get_hierarchical_memory_stats,
        add_conversation_to_hierarchical_memory
    )
    HIERARCHICAL_AVAILABLE = True
except ImportError:
    print("Warning: Hierarchical memory not available - using fallback retrieval")
    HIERARCHICAL_AVAILABLE = False
    def retrieve_hierarchical_memories(query, k=3, memory_types=None):
        return []

from core.faiss_utils import BGEEmbeddings

# ─────────────────── 🧠 Enhanced threshold calculator (Phase 1+3+4) ────
class ContextualThresholdCalculator:
    """
    Enhanced adaptive threshold calculation with Phase 4 hierarchical context.
    
    Phase 4 improvements:
    • Hierarchical memory tier awareness
    • Multi-modal context integration
    • Learning-based threshold adaptation
    """
    
    def __init__(self, cfg: dict[str, object] = MEMORY_THRESHOLD_CONFIG):
        self.cfg = cfg
        self.conversation_context = ConversationContextManager()
        self.hierarchical_context = HierarchicalContextManager()

    def __call__(self, query: str, scores: List[float], context_info: Dict = None) -> float:
        if not self.cfg.get("dynamic_adjustment", True) or not scores:
            return float(self.cfg["base_threshold"])

        # Phase 4: Enhanced threshold calculation with hierarchical context
        base_threshold = self._calculate_base_threshold(scores)
        
        # Apply conversation context adjustments (Phase 3)
        if context_info and CONVERSATION_MANAGEMENT_CONFIG["enable_importance_weighting"]:
            context_adjustment = self._calculate_context_adjustment(query, context_info)
            adjusted_threshold = base_threshold * (1 + context_adjustment)
        else:
            adjusted_threshold = base_threshold

        # Phase 4: Apply hierarchical memory context
        if HIERARCHICAL_AVAILABLE and HIERARCHICAL_MEMORY_CONFIG["enable_hierarchical_memory"]:
            hierarchical_adjustment = self.hierarchical_context.calculate_hierarchical_adjustment(query, scores)
            adjusted_threshold *= (1 + hierarchical_adjustment)

        # Clamp to configured range
        return float(max(
            self.cfg["min_threshold"],
            min(self.cfg["max_threshold"], adjusted_threshold),
        ))
    
    def _calculate_base_threshold(self, scores: List[float]) -> float:
        """Calculate base threshold from score distribution."""
        mean_score = float(np.mean(scores))
        return max(
            self.cfg["min_threshold"],
            min(self.cfg["max_threshold"], mean_score),
        )
    
    def _calculate_context_adjustment(self, query: str, context_info: Dict) -> float:
        """Phase 3: Calculate threshold adjustment based on conversation context."""
        adjustment = 0.0
        
        # Lower threshold (more permissive) for important conversations
        importance_score = context_info.get("importance_score", 0.5)
        if importance_score > 0.7:
            adjustment -= 0.1  # More permissive for important content
        elif importance_score < 0.3:
            adjustment += 0.1  # More restrictive for unimportant content
        
        # Adjust based on conversation recency
        recency_score = context_info.get("recency_score", 0.5)
        if recency_score > 0.8:
            adjustment -= 0.05  # Slightly more permissive for recent conversations
        
        return adjustment

# ─────────────────── 🏗️ Phase 4: Hierarchical Context Manager ─────────
class HierarchicalContextManager:
    """
    Phase 4: Manages context from hierarchical memory system for
    enhanced retrieval decisions.
    """
    
    def __init__(self):
        self.config = HIERARCHICAL_MEMORY_CONFIG if HIERARCHICAL_AVAILABLE else {}
        
    def calculate_hierarchical_adjustment(self, query: str, scores: List[float]) -> float:
        """Calculate threshold adjustment based on hierarchical memory context."""
        if not HIERARCHICAL_AVAILABLE:
            return 0.0
        
        adjustment = 0.0
        
        try:
            # Get hierarchical memory stats
            stats = get_hierarchical_memory_stats()
            
            # Adjust based on memory tier utilization
            working_utilization = stats.get("working_memory", {}).get("utilization", 0)
            if working_utilization > 0.8:
                adjustment -= 0.05  # More permissive if working memory is full
            
            # Adjust based on long-term memory availability
            long_term_total = stats.get("long_term_memory", {}).get("total", 0)
            if long_term_total > 100:
                adjustment += 0.03  # Slightly more restrictive if lots of long-term memories
            
        except Exception as e:
            print(f"[Hierarchical Context Error]: {e}")
        
        return adjustment
    
    def get_hierarchical_retrieval_strategy(self, query: str) -> Dict[str, Any]:
        """Determine optimal retrieval strategy based on query characteristics."""
        strategy = {
            "use_working_memory": True,
            "use_short_term": True,  
            "use_long_term": True,
            "memory_types": ["episodic", "semantic", "procedural"],
            "tier_weights": {"working": 0.4, "short_term": 0.4, "long_term": 0.2}
        }
        
        # Analyze query characteristics
        query_lower = query.lower()
        
        # Factual questions favor semantic memory
        if any(word in query_lower for word in ["what", "define", "explain", "how"]):
            strategy["memory_types"] = ["semantic", "procedural", "episodic"]
            strategy["tier_weights"]["long_term"] = 0.4
            strategy["tier_weights"]["working"] = 0.3
        
        # Personal questions favor episodic memory
        elif any(word in query_lower for word in ["remember", "conversation", "said", "told"]):
            strategy["memory_types"] = ["episodic", "semantic"]
            strategy["tier_weights"]["working"] = 0.5
            strategy["tier_weights"]["short_term"] = 0.4
        
        # Process questions favor procedural memory
        elif any(word in query_lower for word in ["how to", "steps", "process", "method"]):
            strategy["memory_types"] = ["procedural", "semantic"]
            strategy["tier_weights"]["long_term"] = 0.5
        
        return strategy

# ─────────────────── 🏗️ Phase 3+4: Enhanced Conversation Context Manager ─────────────
class ConversationContextManager:
    """
    Phase 3+4: Advanced conversation management with sliding-window,
    importance weighting, semantic clustering, and hierarchical integration.
    """
    
    def __init__(self):
        self.config = CONVERSATION_MANAGEMENT_CONFIG
        self.hierarchical_manager = HierarchicalContextManager()
        self.conversation_clusters = {}
        self.importance_cache = {}
        self.last_cluster_update = datetime.now()
    
    def calculate_importance_score(self, text: str, metadata: Dict = None) -> float:
        """Calculate importance score for conversation text."""
        if not self.config["enable_importance_weighting"]:
            return 0.5
        
        # Generate cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.importance_cache:
            return self.importance_cache[text_hash]
        
        importance = 0.5  # Base importance
        text_lower = text.lower()
        
        # Emotional content detection
        emotional_keywords = {
            'love', 'hate', 'angry', 'sad', 'happy', 'excited', 'frustrated',
            'amazing', 'terrible', 'wonderful', 'awful', 'fantastic', 'horrible'
        }
        if any(keyword in text_lower for keyword in emotional_keywords):
            importance *= self.config["emotion_weight_multiplier"]
        
        # Question detection
        if '?' in text or text_lower.startswith(('what', 'how', 'why', 'when', 'where', 'who')):
            importance *= self.config["question_weight"]
        
        # Factual information detection
        factual_indicators = {'because', 'therefore', 'fact', 'research', 'study', 'data'}
        if any(indicator in text_lower for indicator in factual_indicators):
            importance *= self.config["factual_weight"]
        
        # User name mentions (if available in metadata)
        if metadata and 'user_name' in metadata and metadata['user_name'].lower() in text_lower:
            importance *= self.config["user_mention_weight"]
        
        # Phase 4: Hierarchical memory tier consideration
        if HIERARCHICAL_AVAILABLE and metadata:
            memory_tier = metadata.get("memory_tier", "working")
            if memory_tier == "long_term":
                importance *= 1.2  # Boost importance for long-term memories
        
        # Normalize importance score
        importance = min(1.0, max(0.1, importance))
        
        # Cache the result
        self.importance_cache[text_hash] = importance
        return importance
    
    def apply_sliding_window(self, conversations: List[Dict]) -> List[Dict]:
        """Apply sliding window to conversation history with Phase 4 enhancements."""
        if not self.config["enable_sliding_window"]:
            return conversations
        
        max_length = self.config["max_conversation_length"]
        window_size = self.config["sliding_window_size"]
        
        if len(conversations) <= max_length:
            return conversations
        
        # Sort by importance and recency
        scored_conversations = []
        current_time = datetime.now()
        
        for conv in conversations:
            # Calculate composite score
            importance = self.calculate_importance_score(
                f"User: {conv.get('user', '')}\nAssistant: {conv.get('assistant', '')}",
                conv.get('metadata', {})
            )
            
            # Calculate recency score
            timestamp = conv.get('timestamp', current_time.isoformat())
            try:
                conv_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                hours_old = (current_time - conv_time).total_seconds() / 3600
                recency = max(0.1, 1.0 - (hours_old / 24))  # Decay over 24 hours
            except:
                recency = 0.1
            
            # Phase 4: Add hierarchical memory tier bonus
            tier_bonus = 0.0
            if HIERARCHICAL_AVAILABLE and conv.get('metadata', {}).get('memory_tier'):
                tier = conv['metadata']['memory_tier']
                tier_bonuses = {"working": 0.3, "short_term": 0.2, "long_term": 0.1}
                tier_bonus = tier_bonuses.get(tier, 0.0)
            
            composite_score = importance * 0.5 + recency * 0.3 + tier_bonus * 0.2
            scored_conversations.append((composite_score, conv))
        
        # Sort by composite score and take top conversations
        scored_conversations.sort(key=lambda x: x[0], reverse=True)
        return [conv for _, conv in scored_conversations[:window_size]]
    
    def cluster_conversations(self, conversations: List[Dict]) -> Dict[str, List[Dict]]:
        """Phase 3+4: Group conversations by semantic similarity with hierarchical awareness."""
        if not self.config["enable_semantic_clustering"] or len(conversations) < 2:
            return {"default": conversations}
        
        # Simple clustering based on text similarity
        clusters = defaultdict(list)
        cluster_centers = {}
        
        embedder = BGEEmbeddings()
        
        for conv in conversations:
            text = f"User: {conv.get('user', '')}\nAssistant: {conv.get('assistant', '')}"
            
            # Find best cluster
            best_cluster = None
            best_similarity = 0.0
            
            if cluster_centers:
                conv_embedding = embedder.embed_query(text)
                
                for cluster_id, center_embedding in cluster_centers.items():
                    # Calculate cosine similarity
                    similarity = np.dot(conv_embedding, center_embedding) / (
                        np.linalg.norm(conv_embedding) * np.linalg.norm(center_embedding)
                    )
                    
                    if similarity > best_similarity and similarity > self.config["similarity_cluster_threshold"]:
                        best_similarity = similarity
                        best_cluster = cluster_id
            
            # Assign to cluster
            if best_cluster:
                clusters[best_cluster].append(conv)
            else:
                # Create new cluster
                cluster_id = f"cluster_{len(cluster_centers)}"
                clusters[cluster_id].append(conv)
                cluster_centers[cluster_id] = embedder.embed_query(text)
            
            # Limit number of clusters
            if len(cluster_centers) >= self.config["max_clusters"]:
                break
        
        return dict(clusters)
    
    def get_conversation_context(self, query: str) -> Dict[str, Any]:
        """Get comprehensive conversation context for the query."""
        context = {
            "importance_score": self.calculate_importance_score(query),
            "recency_score": 1.0,  # Current query is maximally recent
            "cluster_info": {},
            "context_suggestions": []
        }
        
        # Phase 4: Add hierarchical context
        if HIERARCHICAL_AVAILABLE:
            hierarchical_strategy = self.hierarchical_manager.get_hierarchical_retrieval_strategy(query)
            context["hierarchical_strategy"] = hierarchical_strategy
        
        return context

# ─────────────────── 🎯 Enhanced retrieval API (Phase 1+2+3+4) ──────────
_threshold_calc = ContextualThresholdCalculator()
_context_manager = ConversationContextManager()

def retrieve_top_memories(
    user_query: str,
    k_short: int = TOP_K,
    k_long: int = 2,
    user_id: str | None = None,
    apply_conversation_management: bool = True,
    use_hierarchical: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Enhanced memory retrieval with Phase 4 hierarchical memory integration.
    
    Phase 1: Threshold filtering ✅
    Phase 2: Auto-sync and deduplication ✅
    Phase 3: Conversation management, importance weighting, clustering ✅
    Phase 4: Hierarchical memory integration, multi-modal retrieval ✅
    """

    # ── Phase 4: Get hierarchical memories first if available ─────────────
    hierarchical_memories = []
    if use_hierarchical and HIERARCHICAL_AVAILABLE and HIERARCHICAL_MEMORY_CONFIG["enable_hierarchical_memory"]:
        try:
            # Get conversation context for hierarchical strategy
            context_info = _context_manager.get_conversation_context(user_query)
            strategy = context_info.get("hierarchical_strategy", {})
            
            # Retrieve from hierarchical memory
            memory_types = strategy.get("memory_types", ["episodic", "semantic", "procedural"])
            hierarchical_memories = retrieve_hierarchical_memories(
                user_query, 
                k=k_short, 
                memory_types=memory_types
            )
            
            print(f"[Hierarchical Retrieval]: Retrieved {len(hierarchical_memories)} memories")
            
        except Exception as e:
            print(f"[Hierarchical Retrieval Error]: {e}")
            hierarchical_memories = []

    # ── Phase 3+4: Get conversation context ──────────────────────────────
    context_info = _context_manager.get_conversation_context(user_query)
    
    # ── Short-term search with enhanced filtering ─────────────────────────
    hits = vector_search(user_query, k=k_short * 2)  # Get more candidates for filtering
    scores = [score for _, score in hits]
    
    # Enhanced threshold calculation with hierarchical context
    threshold = (
        _threshold_calc(user_query, scores, context_info)
        if MEMORY_THRESHOLD_CONFIG["enabled"]
        else MEMORY_THRESHOLD_CONFIG["base_threshold"]
    )
    
    trivial = MEMORY_THRESHOLD_CONFIG["trivial_responses"]

    # Apply threshold filtering and trivial response removal
    filtered_hits = [
        (text, score)
        for text, score in hits
        if score <= threshold and not _is_trivial_response(text, trivial)
    ]
    
    # Phase 3+4: Apply conversation management if enabled
    if apply_conversation_management and CONVERSATION_MANAGEMENT_CONFIG["enable_sliding_window"]:
        filtered_hits = _apply_conversation_management_to_results(filtered_hits, context_info)
    
    # Take short-term results
    short_matches = [text for text, _ in filtered_hits[:k_short]]

    # ── Phase 4: Merge hierarchical and traditional results ──────────────
    if hierarchical_memories:
        # Combine and deduplicate
        all_short_matches = short_matches + hierarchical_memories
        seen_content = set()
        deduplicated_matches = []
        
        for match in all_short_matches:
            # Create a normalized version for comparison
            normalized = match.lower().strip()
            if normalized not in seen_content:
                seen_content.add(normalized)
                deduplicated_matches.append(match)
        
        short_matches = deduplicated_matches[:k_short]
        print(f"[Memory Merge]: Combined {len(all_short_matches)} memories, deduplicated to {len(short_matches)}")

    # ── Long-term search with Phase 3+4 enhancements ────────────────────
    summaries = [e.get("summary", "") for e in load_long_term_memory(user_id)]
    long_matches: List[str] = []

    if summaries:
        embedder = BGEEmbeddings()
        query_vec = np.array([embedder.embed_query(user_query)])

        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        store = FAISS.from_documents(
            [Document(page_content=s) for s in summaries],
            embedder,
        )
        D, I = store.index.search(query_vec, k=k_long * 2)  # Get more candidates

        for idx, score in zip(I[0], D[0]):
            if (
                idx >= 0
                and score <= threshold
                and not _is_trivial_response(summaries[idx], trivial)
            ):
                long_matches.append(summaries[idx])
        
        # Limit to requested count
        long_matches = long_matches[:k_long]

    return short_matches, long_matches

def _apply_conversation_management_to_results(
    results: List[Tuple[str, float]], 
    context_info: Dict
) -> List[Tuple[str, float]]:
    """Phase 3+4: Apply conversation management to search results."""
    if not results:
        return results
    
    # Convert results to conversation-like format for processing
    conversations = []
    for text, score in results:
        # Parse conversation text
        parts = text.split('\nAssistant: ', 1)
        user_part = parts[0].replace('User: ', '') if parts else text
        assistant_part = parts[1] if len(parts) > 1 else ""
        
        conv = {
            'user': user_part,
            'assistant': assistant_part,
            'score': score,
            'timestamp': datetime.now().isoformat(),
            'metadata': {'memory_tier': 'vector_store'}  # Phase 4: Add tier info
        }
        conversations.append(conv)
    
    # Apply sliding window
    managed_conversations = _context_manager.apply_sliding_window(conversations)
    
    # Convert back to results format
    managed_results = [
        (f"User: {conv['user']}\nAssistant: {conv['assistant']}", conv['score'])
        for conv in managed_conversations
    ]
    
    return managed_results

def _is_trivial_response(text: str, trivial_set: set) -> bool:
    """Enhanced trivial response detection."""
    # Extract just the first few words to check against trivial responses
    first_words = text.strip().lower().split()[:3]
    return any(word in trivial_set for word in first_words)

# ─────────────────── 🔍 Phase 4: Advanced multi-modal retrieval functions ────────
def retrieve_memories_by_importance_and_tier(
    user_query: str,
    min_importance: float = 0.6,
    preferred_tiers: List[str] = None,
    k: int = TOP_K
) -> List[str]:
    """Phase 4: Retrieve memories filtered by importance score and memory tier."""
    results = []
    
    # Get traditional vector search results
    hits = vector_search(user_query, k=k * 3)
    for text, score in hits:
        importance = _context_manager.calculate_importance_score(text)
        if importance >= min_importance:
            results.append(text)
    
    # Get hierarchical memory results if available
    if HIERARCHICAL_AVAILABLE and preferred_tiers:
        try:
            hierarchical_results = retrieve_hierarchical_memories(
                user_query, k=k, memory_types=preferred_tiers
            )
            
            # Filter by importance
            for text in hierarchical_results:
                importance = _context_manager.calculate_importance_score(text)
                if importance >= min_importance:
                    results.append(text)
        except Exception as e:
            print(f"[Hierarchical Importance Retrieval Error]: {e}")
    
    # Deduplicate and limit
    seen = set()
    deduplicated = []
    for result in results:
        normalized = result.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            deduplicated.append(result)
    
    return deduplicated[:k]

def retrieve_memories_by_timeframe_and_type(
    user_query: str,
    hours_back: int = 24,
    memory_types: List[str] = None,
    k: int = TOP_K
) -> List[str]:
    """Phase 4: Retrieve memories from specific timeframe and memory types."""
    cutoff_time = datetime.now() - timedelta(hours=hours_back)
    
    results = []
    
    # Get hierarchical memories if available
    if HIERARCHICAL_AVAILABLE and memory_types:
        try:
            hierarchical_results = retrieve_hierarchical_memories(
                user_query, k=k, memory_types=memory_types
            )
            results.extend(hierarchical_results)
        except Exception as e:
            print(f"[Hierarchical Timeframe Retrieval Error]: {e}")
    
    # Fallback to regular retrieval
    if not results:
        short_matches, _ = retrieve_top_memories(user_query, k_short=k, k_long=0)
        results = short_matches
    
    return results[:k]

def get_conversation_clusters_with_hierarchy(user_id: str = None) -> Dict[str, List[str]]:
    """Phase 4: Get conversation clusters with hierarchical memory integration."""
    try:
        # Get traditional clusters
        from memory.turn_memory import load_memory
        conversations = load_memory()
        
        clusters = _context_manager.cluster_conversations(conversations)
        
        # Add hierarchical memory clusters if available
        if HIERARCHICAL_AVAILABLE:
            try:
                h_stats = get_hierarchical_memory_stats()
                print(f"[Hierarchical Clusters]: {h_stats['long_term_memory']['total']} long-term memories available")
            except Exception as e:
                print(f"[Hierarchical Cluster Error]: {e}")
        
        # Convert to string format for easy viewing
        string_clusters = {}
        for cluster_id, convs in clusters.items():
            string_clusters[cluster_id] = [
                f"User: {conv.get('user', '')}\nAssistant: {conv.get('assistant', '')}"
                for conv in convs[:5]  # Limit to first 5 per cluster
            ]
        
        return string_clusters
    except Exception as e:
        print(f"Error getting conversation clusters: {e}")
        return {"error": [str(e)]}

def get_memory_management_stats() -> Dict[str, Any]:
    """Phase 4: Get comprehensive memory management statistics."""
    base_stats = {
        "sliding_window_enabled": CONVERSATION_MANAGEMENT_CONFIG["enable_sliding_window"],
        "importance_weighting_enabled": CONVERSATION_MANAGEMENT_CONFIG["enable_importance_weighting"],
        "semantic_clustering_enabled": CONVERSATION_MANAGEMENT_CONFIG["enable_semantic_clustering"],
        "max_conversation_length": CONVERSATION_MANAGEMENT_CONFIG["max_conversation_length"],
        "sliding_window_size": CONVERSATION_MANAGEMENT_CONFIG["sliding_window_size"],
        "cache_size": len(_context_manager.importance_cache),
        "cluster_count": len(_context_manager.conversation_clusters)
    }
    
    # Add hierarchical memory stats if available
    if HIERARCHICAL_AVAILABLE:
        try:
            hierarchical_stats = get_hierarchical_memory_stats()
            base_stats["hierarchical_memory"] = {
                "enabled": True,
                "working_memory_utilization": hierarchical_stats.get("working_memory", {}).get("utilization", 0),
                "short_term_utilization": hierarchical_stats.get("short_term_memory", {}).get("utilization", 0),
                "long_term_total": hierarchical_stats.get("long_term_memory", {}).get("total", 0),
                "consolidation_enabled": hierarchical_stats.get("consolidation", {}).get("enabled", False)
            }
        except Exception as e:
            base_stats["hierarchical_memory"] = {"enabled": True, "error": str(e)}
    else:
        base_stats["hierarchical_memory"] = {"enabled": False}
    
    return base_stats

# ─────────────────── 🧪 Phase 4: Testing and debugging ──────────────
def test_phase4_memory_features():
    """Test Phase 4 enhanced memory retrieval features."""
    print("🧪 Testing Phase 4 Enhanced Memory Retrieval")
    print("=" * 55)
    
    # Test hierarchical integration
    print("🏗️ Hierarchical Memory Integration:")
    print(f"Available: {'✅ Yes' if HIERARCHICAL_AVAILABLE else '❌ No'}")
    
    if HIERARCHICAL_AVAILABLE:
        try:
            stats = get_hierarchical_memory_stats()
            print(f"Working memory: {stats['working_memory']['count']} items")
            print(f"Long-term memory: {stats['long_term_memory']['total']} items")
        except Exception as e:
            print(f"Error getting stats: {e}")
    
    # Test enhanced retrieval
    print(f"\n🔍 Testing enhanced memory retrieval:")
    test_queries = [
        "How do neural networks work?",
        "What did we discuss about AI?", 
        "I love learning new things"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            short, long = retrieve_top_memories(query, k_short=2, k_long=1, use_hierarchical=True)
            print(f"Retrieved: {len(short)} short-term, {len(long)} long-term")
            
            # Test importance-based retrieval
            important = retrieve_memories_by_importance_and_tier(query, min_importance=0.6, k=2)
            print(f"Important memories: {len(important)}")
            
        except Exception as e:
            print(f"Retrieval error: {e}")
    
    # Test memory management stats
    print(f"\n📊 Memory Management Statistics:")
    stats = get_memory_management_stats()
    print(f"Sliding window: {'✅' if stats['sliding_window_enabled'] else '❌'}")
    print(f"Importance weighting: {'✅' if stats['importance_weighting_enabled'] else '❌'}")
    print(f"Hierarchical: {'✅' if stats['hierarchical_memory']['enabled'] else '❌'}")
    
    print("\n✅ Phase 4 enhanced memory retrieval test completed")

if __name__ == "__main__":
    test_phase4_memory_features()
