# memory/hierarchical_memory.py

"""
Phase 4: Hierarchical Memory System with Working, Short-term, and Long-term tiers.

This module implements a sophisticated multi-tier memory architecture that
separates episodic vs semantic memory and includes memory consolidation processes.
"""

from __future__ import annotations

import json
import threading
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib

# Import configuration
try:
    from config.constants import HIERARCHICAL_MEMORY_CONFIG, TOP_K
except ImportError:
    HIERARCHICAL_MEMORY_CONFIG = {
        "enable_hierarchical_memory": True,
        "working_memory_size": 20,
        "short_term_memory_size": 200,
        "long_term_memory_threshold": 0.7,
        "enable_episodic_semantic_separation": True,
        "episodic_memory_weight": 0.8,
        "semantic_memory_weight": 1.2,
        "procedural_memory_weight": 0.9,
        "enable_memory_consolidation": True,
        "consolidation_interval_hours": 12,
        "consolidation_threshold": 0.6,
        "memory_interference_factor": 0.1,
        "enable_associative_memory": True,
        "association_strength_threshold": 0.5,
        "max_associations_per_memory": 10,
        "memory_rehearsal_boost": 1.5,
    }
    TOP_K = 3

# Import vectorstore for embeddings
try:
    from vectorstore import BGEEmbeddings, search as vector_search
    from memory.turn_memory import load_memory
except ImportError:
    print("Warning: Could not import vectorstore - hierarchical memory will run in limited mode")
    BGEEmbeddings = None
    vector_search = None
    load_memory = lambda: []

# File paths
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
HIERARCHICAL_MEMORY_DIR = DATA_DIR / "hierarchical_memory"
HIERARCHICAL_MEMORY_DIR.mkdir(exist_ok=True)

WORKING_MEMORY_FILE = HIERARCHICAL_MEMORY_DIR / "working_memory.json"
SHORT_TERM_MEMORY_FILE = HIERARCHICAL_MEMORY_DIR / "short_term_memory.jsonl"
LONG_TERM_MEMORY_FILE = HIERARCHICAL_MEMORY_DIR / "long_term_memory.jsonl"
EPISODIC_MEMORY_FILE = HIERARCHICAL_MEMORY_DIR / "episodic_memory.jsonl"
SEMANTIC_MEMORY_FILE = HIERARCHICAL_MEMORY_DIR / "semantic_memory.jsonl"
PROCEDURAL_MEMORY_FILE = HIERARCHICAL_MEMORY_DIR / "procedural_memory.jsonl"
MEMORY_ASSOCIATIONS_FILE = HIERARCHICAL_MEMORY_DIR / "memory_associations.json"
CONSOLIDATION_LOG_FILE = HIERARCHICAL_MEMORY_DIR / "consolidation_log.jsonl"

# ─────────────────── 🧠 Memory Item Classes ─────────────────────

class MemoryItem:
    """
    Base class for memory items with importance scoring and metadata.
    """
    
    def __init__(self, content: str, memory_type: str = "general", metadata: Dict = None):
        self.content = content
        self.memory_type = memory_type  # episodic, semantic, procedural
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.access_count = 0
        self.last_accessed = self.timestamp
        self.importance_score = 0.5
        self.consolidation_score = 0.0
        self.associations = []
        self.id = self._generate_id()
        
    def _generate_id(self) -> str:
        """Generate unique ID for memory item."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        timestamp_hash = hashlib.md5(self.timestamp.isoformat().encode()).hexdigest()[:4]
        return f"{self.memory_type}_{content_hash}_{timestamp_hash}"
    
    def access(self) -> None:
        """Record memory access and update statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()
        
        # Boost importance based on access frequency
        access_boost = min(0.1, self.access_count * 0.01)
        self.importance_score = min(1.0, self.importance_score + access_boost)
    
    def calculate_importance(self, context: Dict = None) -> float:
        """Calculate dynamic importance score based on various factors."""
        base_importance = self.importance_score
        
        # Recency factor
        hours_old = (datetime.now() - self.timestamp).total_seconds() / 3600
        recency_factor = max(0.1, 1.0 - (hours_old / 168))  # Decay over 1 week
        
        # Access frequency factor
        access_factor = min(1.0, 0.5 + (self.access_count * 0.1))
        
        # Memory type weight
        type_weights = {
            "episodic": HIERARCHICAL_MEMORY_CONFIG["episodic_memory_weight"],
            "semantic": HIERARCHICAL_MEMORY_CONFIG["semantic_memory_weight"],
            "procedural": HIERARCHICAL_MEMORY_CONFIG["procedural_memory_weight"]
        }
        type_weight = type_weights.get(self.memory_type, 1.0)
        
        # Consolidation bonus
        consolidation_bonus = self.consolidation_score * 0.2
        
        # Calculate final importance
        final_importance = (
            base_importance * 0.4 +
            recency_factor * 0.2 +
            access_factor * 0.2 +
            consolidation_bonus * 0.2
        ) * type_weight
        
        return min(1.0, max(0.0, final_importance))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory item to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "importance_score": self.importance_score,
            "consolidation_score": self.consolidation_score,
            "associations": self.associations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create memory item from dictionary."""
        item = cls(data["content"], data["memory_type"], data["metadata"])
        item.id = data["id"]
        item.timestamp = datetime.fromisoformat(data["timestamp"])
        item.access_count = data["access_count"]
        item.last_accessed = datetime.fromisoformat(data["last_accessed"])
        item.importance_score = data["importance_score"]
        item.consolidation_score = data["consolidation_score"]
        item.associations = data["associations"]
        return item

# ─────────────────── 🏗️ Memory Tier Classes ─────────────────────

class WorkingMemory:
    """
    Working memory - holds the most recent and immediately relevant information.
    Limited capacity, fast access.
    """
    
    def __init__(self, max_size: int = None):
        self.max_size = max_size or HIERARCHICAL_MEMORY_CONFIG["working_memory_size"]
        self.items = deque(maxlen=self.max_size)
        self.lock = threading.Lock()
        self._load_from_disk()
    
    def add_item(self, item: MemoryItem) -> None:
        """Add item to working memory."""
        with self.lock:
            # Check for duplicates
            for existing_item in self.items:
                if existing_item.content == item.content:
                    existing_item.access()
                    return
            
            self.items.append(item)
            self._save_to_disk()
    
    def get_items(self, query: str = None, k: int = None) -> List[MemoryItem]:
        """Get items from working memory, optionally filtered by query."""
        with self.lock:
            items = list(self.items)
            
            if query and BGEEmbeddings:
                # Rank by semantic similarity
                embedder = BGEEmbeddings()
                query_embedding = embedder.embed_query(query)
                
                scored_items = []
                for item in items:
                    item_embedding = embedder.embed_query(item.content)
                    similarity = np.dot(query_embedding, item_embedding)
                    scored_items.append((similarity, item))
                
                scored_items.sort(key=lambda x: x[0], reverse=True)
                items = [item for _, item in scored_items]
            
            # Sort by recency and importance
            items.sort(key=lambda x: (x.last_accessed, x.calculate_importance()), reverse=True)
            
            return items[:k] if k else items
    
    def promote_to_short_term(self, importance_threshold: float = 0.6) -> List[MemoryItem]:
        """Promote important items from working memory to short-term memory."""
        with self.lock:
            promoted_items = []
            remaining_items = []
            
            for item in self.items:
                if item.calculate_importance() >= importance_threshold:
                    promoted_items.append(item)
                else:
                    remaining_items.append(item)
            
            # Update working memory
            self.items.clear()
            self.items.extend(remaining_items)
            self._save_to_disk()
            
            return promoted_items
    
    def _load_from_disk(self) -> None:
        """Load working memory from disk."""
        try:
            if WORKING_MEMORY_FILE.exists():
                with open(WORKING_MEMORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item_data in data:
                        item = MemoryItem.from_dict(item_data)
                        self.items.append(item)
        except Exception as e:
            print(f"[Working Memory Load Error]: {e}")
    
    def _save_to_disk(self) -> None:
        """Save working memory to disk."""
        try:
            with open(WORKING_MEMORY_FILE, "w", encoding="utf-8") as f:
                data = [item.to_dict() for item in self.items]
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[Working Memory Save Error]: {e}")

class ShortTermMemory:
    """
    Short-term memory - holds recently important information.
    Larger capacity than working memory, items can be consolidated to long-term.
    """
    
    def __init__(self, max_size: int = None):
        self.max_size = max_size or HIERARCHICAL_MEMORY_CONFIG["short_term_memory_size"]
        self.items = []
        self.lock = threading.Lock()
        self._load_from_disk()
    
    def add_item(self, item: MemoryItem) -> None:
        """Add item to short-term memory."""
        with self.lock:
            # Check for duplicates
            for existing_item in self.items:
                if existing_item.content == item.content:
                    existing_item.access()
                    return
            
            self.items.append(item)
            
            # Maintain size limit
            if len(self.items) > self.max_size:
                # Remove least important items
                self.items.sort(key=lambda x: x.calculate_importance())
                self.items = self.items[-(self.max_size):]
            
            self._save_to_disk()
    
    def get_items(self, query: str = None, k: int = None) -> List[MemoryItem]:
        """Get items from short-term memory."""
        with self.lock:
            items = self.items.copy()
            
            if query and BGEEmbeddings:
                # Rank by semantic similarity
                embedder = BGEEmbeddings()
                query_embedding = embedder.embed_query(query)
                
                scored_items = []
                for item in items:
                    item_embedding = embedder.embed_query(item.content)
                    similarity = np.dot(query_embedding, item_embedding)
                    scored_items.append((similarity, item))
                
                scored_items.sort(key=lambda x: x[0], reverse=True)
                items = [item for _, item in scored_items]
            
            # Sort by importance and recency
            items.sort(key=lambda x: (x.calculate_importance(), x.last_accessed), reverse=True)
            
            return items[:k] if k else items
    
    def consolidate_to_long_term(self, consolidation_threshold: float = None) -> List[MemoryItem]:
        """Consolidate important items to long-term memory."""
        threshold = consolidation_threshold or HIERARCHICAL_MEMORY_CONFIG["consolidation_threshold"]
        
        with self.lock:
            consolidated_items = []
            remaining_items = []
            
            for item in self.items:
                importance = item.calculate_importance()
                if importance >= threshold:
                    item.consolidation_score = importance
                    consolidated_items.append(item)
                else:
                    remaining_items.append(item)
            
            # Update short-term memory
            self.items = remaining_items
            self._save_to_disk()
            
            return consolidated_items
    
    def _load_from_disk(self) -> None:
        """Load short-term memory from disk."""
        try:
            if SHORT_TERM_MEMORY_FILE.exists():
                with open(SHORT_TERM_MEMORY_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            item_data = json.loads(line)
                            item = MemoryItem.from_dict(item_data)
                            self.items.append(item)
        except Exception as e:
            print(f"[Short-term Memory Load Error]: {e}")
    
    def _save_to_disk(self) -> None:
        """Save short-term memory to disk."""
        try:
            with open(SHORT_TERM_MEMORY_FILE, "w", encoding="utf-8") as f:
                for item in self.items:
                    f.write(json.dumps(item.to_dict()) + "\n")
        except Exception as e:
            print(f"[Short-term Memory Save Error]: {e}")

class LongTermMemory:
    """
    Long-term memory - holds consolidated, important information.
    Organized by memory type (episodic, semantic, procedural).
    """
    
    def __init__(self):
        self.items = {"episodic": [], "semantic": [], "procedural": []}
        self.associations = {}
        self.lock = threading.Lock()
        self._load_from_disk()
    
    def add_item(self, item: MemoryItem) -> None:
        """Add item to long-term memory."""
        with self.lock:
            memory_type = item.memory_type
            if memory_type not in self.items:
                memory_type = "episodic"  # Default fallback
            
            # Check for duplicates
            for existing_item in self.items[memory_type]:
                if existing_item.content == item.content:
                    existing_item.access()
                    return
            
            self.items[memory_type].append(item)
            self._create_associations(item)
            self._save_to_disk()
    
    def get_items(self, query: str = None, memory_type: str = None, k: int = None) -> List[MemoryItem]:
        """Get items from long-term memory."""
        with self.lock:
            # Select items by type
            if memory_type and memory_type in self.items:
                items = self.items[memory_type].copy()
            else:
                items = []
                for type_items in self.items.values():
                    items.extend(type_items)
            
            # Filter by query if provided
            if query and BGEEmbeddings:
                embedder = BGEEmbeddings()
                query_embedding = embedder.embed_query(query)
                
                scored_items = []
                for item in items:
                    item_embedding = embedder.embed_query(item.content)
                    similarity = np.dot(query_embedding, item_embedding)
                    scored_items.append((similarity, item))
                
                scored_items.sort(key=lambda x: x[0], reverse=True)
                items = [item for _, item in scored_items]
            
            # Sort by consolidation score and importance
            items.sort(key=lambda x: (x.consolidation_score, x.calculate_importance()), reverse=True)
            
            return items[:k] if k else items
    
    def _create_associations(self, new_item: MemoryItem) -> None:
        """Create associations between memory items."""
        if not HIERARCHICAL_MEMORY_CONFIG["enable_associative_memory"]:
            return
        
        if not BGEEmbeddings:
            return
        
        embedder = BGEEmbeddings()
        new_embedding = embedder.embed_query(new_item.content)
        
        # Find associations with existing items
        all_items = []
        for type_items in self.items.values():
            all_items.extend(type_items)
        
        associations = []
        threshold = HIERARCHICAL_MEMORY_CONFIG["association_strength_threshold"]
        max_associations = HIERARCHICAL_MEMORY_CONFIG["max_associations_per_memory"]
        
        for item in all_items[-100:]:  # Check last 100 items for efficiency
            if item.id == new_item.id:
                continue
            
            item_embedding = embedder.embed_query(item.content)
            similarity = np.dot(new_embedding, item_embedding)
            
            if similarity > threshold:
                associations.append({
                    "item_id": item.id,
                    "strength": similarity,
                    "type": "semantic_similarity"
                })
        
        # Sort by strength and limit
        associations.sort(key=lambda x: x["strength"], reverse=True)
        new_item.associations = associations[:max_associations]
        
        # Also add reverse associations
        for assoc in new_item.associations:
            for type_items in self.items.values():
                for item in type_items:
                    if item.id == assoc["item_id"]:
                        item.associations.append({
                            "item_id": new_item.id,
                            "strength": assoc["strength"],
                            "type": "semantic_similarity"
                        })
                        # Limit associations per item
                        if len(item.associations) > max_associations:
                            item.associations = sorted(item.associations, key=lambda x: x["strength"], reverse=True)[:max_associations]
    
    def get_associated_items(self, item_id: str, max_items: int = 5) -> List[MemoryItem]:
        """Get items associated with a given item."""
        associated_items = []
        
        # Find the source item
        source_item = None
        for type_items in self.items.values():
            for item in type_items:
                if item.id == item_id:
                    source_item = item
                    break
        
        if not source_item:
            return []
        
        # Get associated items
        for assoc in source_item.associations:
            for type_items in self.items.values():
                for item in type_items:
                    if item.id == assoc["item_id"]:
                        associated_items.append((assoc["strength"], item))
        
        # Sort by association strength
        associated_items.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in associated_items[:max_items]]
    
    def _load_from_disk(self) -> None:
        """Load long-term memory from disk."""
        try:
            # Load each memory type
            for memory_type in ["episodic", "semantic", "procedural"]:
                file_path = HIERARCHICAL_MEMORY_DIR / f"{memory_type}_memory.jsonl"
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                item_data = json.loads(line)
                                item = MemoryItem.from_dict(item_data)
                                self.items[memory_type].append(item)
        except Exception as e:
            print(f"[Long-term Memory Load Error]: {e}")
    
    def _save_to_disk(self) -> None:
        """Save long-term memory to disk."""
        try:
            # Save each memory type
            for memory_type, items in self.items.items():
                file_path = HIERARCHICAL_MEMORY_DIR / f"{memory_type}_memory.jsonl"
                with open(file_path, "w", encoding="utf-8") as f:
                    for item in items:
                        f.write(json.dumps(item.to_dict()) + "\n")
        except Exception as e:
            print(f"[Long-term Memory Save Error]: {e}")

# ─────────────────── 🔄 Memory Consolidation System ─────────────────

class MemoryConsolidator:
    """
    Handles memory consolidation processes - moving items between tiers
    based on importance, access patterns, and time.
    """
    
    def __init__(self):
        self.config = HIERARCHICAL_MEMORY_CONFIG
        self.last_consolidation = datetime.now()
        self.consolidation_history = []
        
    def should_consolidate(self) -> bool:
        """Determine if consolidation should be performed."""
        if not self.config["enable_memory_consolidation"]:
            return False
        
        hours_since = (datetime.now() - self.last_consolidation).total_seconds() / 3600
        return hours_since >= self.config["consolidation_interval_hours"]
    
    def perform_consolidation(self, working_memory: WorkingMemory, 
                            short_term_memory: ShortTermMemory, 
                            long_term_memory: LongTermMemory) -> Dict[str, Any]:
        """Perform complete memory consolidation across all tiers."""
        consolidation_start = time.time()
        
        print("[Memory Consolidation]: Starting consolidation process...")
        
        # Step 1: Promote from working to short-term memory
        promoted_to_short = working_memory.promote_to_short_term()
        for item in promoted_to_short:
            short_term_memory.add_item(item)
        
        # Step 2: Consolidate from short-term to long-term memory
        consolidated_to_long = short_term_memory.consolidate_to_long_term()
        for item in consolidated_to_long:
            long_term_memory.add_item(item)
        
        # Step 3: Apply memory interference and decay
        self._apply_memory_interference(short_term_memory, long_term_memory)
        
        consolidation_time = time.time() - consolidation_start
        self.last_consolidation = datetime.now()
        
        # Log consolidation event
        consolidation_event = {
            "timestamp": self.last_consolidation.isoformat(),
            "promoted_to_short_term": len(promoted_to_short),
            "consolidated_to_long_term": len(consolidated_to_long),
            "consolidation_time_seconds": consolidation_time,
            "total_memory_items": {
                "working": len(working_memory.items),
                "short_term": len(short_term_memory.items),
                "long_term": sum(len(items) for items in long_term_memory.items.values())
            }
        }
        
        self.consolidation_history.append(consolidation_event)
        self._log_consolidation(consolidation_event)
        
        print(f"[Memory Consolidation]: Completed in {consolidation_time:.2f}s")
        print(f"  Promoted to short-term: {len(promoted_to_short)}")
        print(f"  Consolidated to long-term: {len(consolidated_to_long)}")
        
        return consolidation_event
    
    def _apply_memory_interference(self, short_term: ShortTermMemory, long_term: LongTermMemory) -> None:
        """Apply memory interference effects to reduce similar memories."""
        interference_factor = self.config["memory_interference_factor"]
        
        if interference_factor <= 0:
            return
        
        # Find similar items in short-term memory and reduce their importance
        if BGEEmbeddings:
            embedder = BGEEmbeddings()
            
            items = short_term.items.copy()
            for i, item1 in enumerate(items):
                for j, item2 in enumerate(items[i+1:], i+1):
                    # Calculate similarity
                    emb1 = embedder.embed_query(item1.content)
                    emb2 = embedder.embed_query(item2.content)
                    similarity = np.dot(emb1, emb2)
                    
                    # Apply interference if items are very similar
                    if similarity > 0.9:
                        # Reduce importance of the less accessed item
                        if item1.access_count < item2.access_count:
                            item1.importance_score *= (1 - interference_factor)
                        else:
                            item2.importance_score *= (1 - interference_factor)
    
    def _log_consolidation(self, event: Dict[str, Any]) -> None:
        """Log consolidation event to file."""
        try:
            with open(CONSOLIDATION_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"[Consolidation Log Error]: {e}")

# ─────────────────── 🧠 Hierarchical Memory Manager ─────────────────

class HierarchicalMemoryManager:
    """
    Main manager for the hierarchical memory system.
    Coordinates all memory tiers and provides unified access.
    """
    
    def __init__(self):
        self.config = HIERARCHICAL_MEMORY_CONFIG
        self.working_memory = WorkingMemory()
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory()
        self.consolidator = MemoryConsolidator()
        self.memory_classifier = MemoryClassifier()
        
    def add_memory(self, content: str, context: Dict = None, memory_type: str = None) -> MemoryItem:
        """Add new memory to the hierarchical system."""
        
        # Classify memory type if not provided
        if memory_type is None:
            memory_type = self.memory_classifier.classify_memory_type(content, context)
        
        # Create memory item
        memory_item = MemoryItem(content, memory_type, context)
        
        # Calculate initial importance
        memory_item.importance_score = self.memory_classifier.calculate_initial_importance(content, context)
        
        # Add to working memory
        self.working_memory.add_item(memory_item)
        
        # Check if consolidation is needed
        if self.consolidator.should_consolidate():
            self.consolidator.perform_consolidation(
                self.working_memory, 
                self.short_term_memory, 
                self.long_term_memory
            )
        
        return memory_item
    
    def retrieve_memories(self, query: str, k: int = TOP_K, memory_types: List[str] = None) -> List[MemoryItem]:
        """Retrieve memories from all tiers based on query."""
        
        # Collect memories from all tiers
        all_memories = []
        
        # Working memory (most recent and relevant)
        working_items = self.working_memory.get_items(query, k * 2)
        all_memories.extend(working_items)
        
        # Short-term memory
        short_term_items = self.short_term_memory.get_items(query, k * 2)
        all_memories.extend(short_term_items)
        
        # Long-term memory
        for memory_type in (memory_types or ["episodic", "semantic", "procedural"]):
            long_term_items = self.long_term_memory.get_items(query, memory_type, k)
            all_memories.extend(long_term_items)
        
        # Remove duplicates
        seen_ids = set()
        unique_memories = []
        for memory in all_memories:
            if memory.id not in seen_ids:
                seen_ids.add(memory.id)
                unique_memories.append(memory)
                memory.access()  # Record access
        
        # Rank by relevance and importance
        if BGEEmbeddings and query:
            scored_memories = self._rank_memories_by_relevance(query, unique_memories)
        else:
            scored_memories = [(m.calculate_importance(), m) for m in unique_memories]
            scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        return [memory for _, memory in scored_memories[:k]]
    
    def _rank_memories_by_relevance(self, query: str, memories: List[MemoryItem]) -> List[Tuple[float, MemoryItem]]:
        """Rank memories by semantic relevance to query."""
        embedder = BGEEmbeddings()
        query_embedding = embedder.embed_query(query)
        
        scored_memories = []
        for memory in memories:
            # Calculate semantic similarity
            memory_embedding = embedder.embed_query(memory.content)
            similarity = np.dot(query_embedding, memory_embedding)
            
            # Combine with importance score
            importance = memory.calculate_importance()
            combined_score = similarity * 0.7 + importance * 0.3
            
            scored_memories.append((combined_score, memory))
        
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return scored_memories
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        return {
            "working_memory": {
                "count": len(self.working_memory.items),
                "max_size": self.working_memory.max_size,
                "utilization": len(self.working_memory.items) / self.working_memory.max_size
            },
            "short_term_memory": {
                "count": len(self.short_term_memory.items),
                "max_size": self.short_term_memory.max_size,
                "utilization": len(self.short_term_memory.items) / self.short_term_memory.max_size
            },
            "long_term_memory": {
                "episodic": len(self.long_term_memory.items["episodic"]),
                "semantic": len(self.long_term_memory.items["semantic"]),
                "procedural": len(self.long_term_memory.items["procedural"]),
                "total": sum(len(items) for items in self.long_term_memory.items.values())
            },
            "consolidation": {
                "last_consolidation": self.consolidator.last_consolidation.isoformat(),
                "consolidation_events": len(self.consolidator.consolidation_history),
                "enabled": self.config["enable_memory_consolidation"]
            },
            "configuration": {
                "hierarchical_enabled": self.config["enable_hierarchical_memory"],
                "associative_enabled": self.config["enable_associative_memory"],
                "episodic_semantic_separation": self.config["enable_episodic_semantic_separation"]
            }
        }
    
    def force_consolidation(self) -> Dict[str, Any]:
        """Manually trigger memory consolidation."""
        return self.consolidator.perform_consolidation(
            self.working_memory, 
            self.short_term_memory, 
            self.long_term_memory
        )

# ─────────────────── 🔍 Memory Classification System ─────────────────

class MemoryClassifier:
    """
    Classifies memories into episodic, semantic, or procedural categories
    and calculates initial importance scores.
    """
    
    def classify_memory_type(self, content: str, context: Dict = None) -> str:
        """Classify memory content into episodic, semantic, or procedural."""
        content_lower = content.lower()
        
        # Episodic memory indicators (personal experiences, events)
        episodic_indicators = [
            "i remember", "yesterday", "last week", "when i", "we went",
            "happened", "conversation", "meeting", "said", "told me"
        ]
        
        # Semantic memory indicators (facts, knowledge)
        semantic_indicators = [
            "is", "are", "means", "definition", "fact", "always", "never",
            "rule", "principle", "concept", "theory", "because", "therefore"
        ]
        
        # Procedural memory indicators (how-to, processes)
        procedural_indicators = [
            "how to", "step", "first", "then", "next", "process", "method",
            "procedure", "instructions", "way to", "in order to"
        ]
        
        # Count indicators
        episodic_score = sum(1 for indicator in episodic_indicators if indicator in content_lower)
        semantic_score = sum(1 for indicator in semantic_indicators if indicator in content_lower)
        procedural_score = sum(1 for indicator in procedural_indicators if indicator in content_lower)
        
        # Use context if available
        if context:
            if context.get("is_conversation", False):
                episodic_score += 2
            if context.get("is_factual", False):
                semantic_score += 2
            if context.get("is_instructional", False):
                procedural_score += 2
        
        # Determine memory type
        max_score = max(episodic_score, semantic_score, procedural_score)
        if max_score == 0:
            return "episodic"  # Default to episodic
        elif episodic_score == max_score:
            return "episodic"
        elif semantic_score == max_score:
            return "semantic"
        else:
            return "procedural"
    
    def calculate_initial_importance(self, content: str, context: Dict = None) -> float:
        """Calculate initial importance score for memory content."""
        base_importance = 0.5
        
        # Content-based factors
        content_lower = content.lower()
        
        # Emotional content increases importance
        emotional_keywords = [
            "love", "hate", "excited", "angry", "sad", "happy", "frustrated",
            "amazing", "terrible", "wonderful", "awful"
        ]
        emotion_boost = sum(0.1 for keyword in emotional_keywords if keyword in content_lower)
        
        # Questions are often important
        question_boost = 0.1 if "?" in content else 0.0
        
        # Factual information
        factual_boost = 0.1 if any(word in content_lower for word in ["fact", "research", "study", "data"]) else 0.0
        
        # Context-based factors
        context_boost = 0.0
        if context:
            if context.get("user_initiated", False):
                context_boost += 0.1
            if context.get("high_emotion", False):
                context_boost += 0.2
            if context.get("important_topic", False):
                context_boost += 0.15
        
        # Calculate final importance
        final_importance = base_importance + emotion_boost + question_boost + factual_boost + context_boost
        return min(1.0, max(0.1, final_importance))

# ─────────────────── 🌐 Global Memory Manager Instance ─────────────────

# Create global hierarchical memory manager
_hierarchical_memory = HierarchicalMemoryManager()

# ─────────────────── 📡 Public API Functions ─────────────────

def add_conversation_to_hierarchical_memory(user_input: str, assistant_response: str, context: Dict = None) -> None:
    """Add a conversation turn to hierarchical memory."""
    if not HIERARCHICAL_MEMORY_CONFIG["enable_hierarchical_memory"]:
        return
    
    conversation_content = f"User: {user_input}\nAssistant: {assistant_response}"
    conversation_context = {
        "is_conversation": True,
        "user_input": user_input,
        "assistant_response": assistant_response,
        "timestamp": datetime.now().isoformat()
    }
    
    if context:
        conversation_context.update(context)
    
    _hierarchical_memory.add_memory(conversation_content, conversation_context, "episodic")

def retrieve_hierarchical_memories(query: str, k: int = TOP_K, memory_types: List[str] = None) -> List[str]:
    """Retrieve memories from hierarchical memory system."""
    if not HIERARCHICAL_MEMORY_CONFIG["enable_hierarchical_memory"]:
        return []
    
    memory_items = _hierarchical_memory.retrieve_memories(query, k, memory_types)
    return [item.content for item in memory_items]

def add_fact_to_semantic_memory(fact: str, context: Dict = None) -> None:
    """Add a fact to semantic memory."""
    fact_context = {"is_factual": True}
    if context:
        fact_context.update(context)
    
    _hierarchical_memory.add_memory(fact, fact_context, "semantic")

def add_procedure_to_memory(procedure: str, context: Dict = None) -> None:
    """Add a procedure or how-to to procedural memory."""
    proc_context = {"is_instructional": True}
    if context:
        proc_context.update(context)
    
    _hierarchical_memory.add_memory(procedure, proc_context, "procedural")

def get_hierarchical_memory_stats() -> Dict[str, Any]:
    """Get hierarchical memory system statistics."""
    return _hierarchical_memory.get_memory_statistics()

def force_memory_consolidation() -> Dict[str, Any]:
    """Manually trigger memory consolidation."""
    return _hierarchical_memory.force_consolidation()

def get_associated_memories(memory_content: str, max_items: int = 5) -> List[str]:
    """Get memories associated with given content."""
    # Find memory item by content
    all_memories = []
    all_memories.extend(_hierarchical_memory.working_memory.get_items())
    all_memories.extend(_hierarchical_memory.short_term_memory.get_items())
    for memory_type in ["episodic", "semantic", "procedural"]:
        all_memories.extend(_hierarchical_memory.long_term_memory.get_items(memory_type=memory_type))
    
    # Find matching memory
    target_memory = None
    for memory in all_memories:
        if memory.content == memory_content:
            target_memory = memory
            break
    
    if not target_memory:
        return []
    
    # Get associated memories
    associated_items = _hierarchical_memory.long_term_memory.get_associated_items(target_memory.id, max_items)
    return [item.content for item in associated_items]

# ─────────────────── 🧪 Testing and Debugging ─────────────────

def test_hierarchical_memory():
    """Test hierarchical memory system functionality."""
    print("🧪 Testing Phase 4 Hierarchical Memory System")
    print("=" * 55)
    
    # Test memory addition
    test_conversations = [
        ("What is machine learning?", "Machine learning is a method of data analysis that automates analytical model building."),
        ("I love learning about AI", "That's wonderful! AI is a fascinating field with many applications."),
        ("How do I train a neural network?", "To train a neural network, you need data, define the architecture, and use backpropagation.")
    ]
    
    for user_input, assistant_response in test_conversations:
        add_conversation_to_hierarchical_memory(user_input, assistant_response)
        print(f"Added: {user_input[:30]}...")
    
    # Test memory retrieval
    print(f"\n🔍 Testing memory retrieval:")
    
    queries = ["machine learning", "neural network", "AI applications"]
    for query in queries:
        results = retrieve_hierarchical_memories(query, k=2)
        print(f"Query '{query}': {len(results)} results")
        for result in results:
            print(f"  - {result[:50]}...")
    
    # Test memory statistics
    print(f"\n📊 Memory Statistics:")
    stats = get_hierarchical_memory_stats()
    print(f"Working memory: {stats['working_memory']['count']}/{stats['working_memory']['max_size']}")
    print(f"Short-term memory: {stats['short_term_memory']['count']}/{stats['short_term_memory']['max_size']}")
    print(f"Long-term memory: {stats['long_term_memory']['total']} items")
    
    # Test consolidation
    print(f"\n🔄 Testing memory consolidation:")
    consolidation_result = force_memory_consolidation()
    print(f"Consolidation completed in {consolidation_result['consolidation_time_seconds']:.2f}s")
    
    print("\n✅ Phase 4 hierarchical memory test completed")

if __name__ == "__main__":
    test_hierarchical_memory()
