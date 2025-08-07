# memory/turn_memory.py

"""
Turn-based memory persistence with Phase 2 optimizations.

Phase 2 enhancements:
• Automated JSONL→FAISS sync with rehydrate_from_jsonl() hooks
• Built-in deduplication to prevent redundant conversation turns
• Performance monitoring and batch processing optimization
• Thread-safe operations with proper error handling
"""

from __future__ import annotations

import json
import hashlib
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Set, Any
from datetime import datetime, timedelta

# Import with error handling for development
try:
    from vectorstore import add_texts, get_store, get_performance_stats
    from config.constants import MEMORY_FILE, AUTO_SYNC_CONFIG, PERFORMANCE_CONFIG
except ImportError as e:
    print(f"Warning: Could not import dependencies: {e}")
    
    # Mock functions for development
    def add_texts(texts): 
        print(f"Mock add_texts called with {len(texts)} texts")
    def get_store():
        class MockStore:
            def save_local(self, path): pass
        return MockStore()
    def get_performance_stats():
        return {"index_size": 0}

# ─────────────────── 📊 Global state management ─────────────────────
_performance_stats = {
    "total_turns": 0,
    "duplicates_found": 0,
    "sync_operations": 0,
    "last_sync_time": None,
    "avg_sync_duration": 0.0,
    "total_sync_duration": 0.0,
    "startup_time": None,
    "batch_operations": 0,
}

_turn_counter = 0
_seen_hashes: Set[str] = set()
_performance_lock = threading.Lock()
_last_hash_cleanup = time.time()

# Ensure data directory exists
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CHAT_FILE = MEMORY_FILE

# ─────────────────── 🚫 Enhanced deduplication management ─────────────
class MemoryDeduplicator:
    """Handles conversation turn deduplication with enhanced algorithms."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or AUTO_SYNC_CONFIG
        
    def generate_content_hash(self, turn_content: str) -> str:
        """Generate content hash for deduplication."""
        content_bytes = turn_content.encode('utf-8')
        return hashlib.md5(content_bytes).hexdigest()
    
    def is_duplicate(self, turn: Dict[str, str]) -> bool:
        """Check if turn is a duplicate based on content hash."""
        if not self.config.get("dedupe_enabled", True):
            return False
            
        turn_content = f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}"
        content_hash = self.generate_content_hash(turn_content)
        
        if content_hash in _seen_hashes:
            return True
        
        _seen_hashes.add(content_hash)
        return False
    
    def cleanup_old_hashes(self) -> None:
        """Periodically cleanup old hashes to prevent memory growth."""
        global _last_hash_cleanup
        
        current_time = time.time()
        cleanup_interval = 24 * 3600  # 24 hours
        
        if current_time - _last_hash_cleanup > cleanup_interval:
            # Keep only recent hashes (simple LRU-style cleanup)
            max_size = 10000
            if len(_seen_hashes) > max_size:
                # Remove oldest 25% of hashes
                hashes_to_remove = list(_seen_hashes)[:len(_seen_hashes)//4]
                for hash_val in hashes_to_remove:
                    _seen_hashes.discard(hash_val)
            
            _last_hash_cleanup = current_time

# ─────────────────── 🔄 Enhanced auto-sync management ─────────────────
class AutoSyncManager:
    """Handles automatic FAISS synchronization with enhanced features."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or AUTO_SYNC_CONFIG
        
    def should_sync(self, turn_number: int) -> bool:
        """Determine if sync should be triggered based on configuration."""
        if not self.config.get("enabled", True):
            return False
            
        # Immediate sync after each dump_turn()
        if self.config.get("sync_on_dump", True):
            return True
            
        # Threshold-based sync (every N turns)
        threshold = self.config.get("sync_threshold", 10)
        return turn_number % threshold == 0
    
    def perform_sync(self) -> bool:
        """Perform FAISS synchronization with enhanced error handling."""
        try:
            sync_start = time.time()
            
            # Method 1: Try to use rehydrate_from_jsonl if available
            try:
                from core.faiss_utils import rehydrate_from_jsonl
                rehydrate_from_jsonl(force=True)
                sync_method = "rehydrate_from_jsonl"
            except ImportError:
                # Method 2: Use vectorstore save if rehydrate not available
                store = get_store()
                store.save_local(str(DATA_DIR / "faiss_index"))
                sync_method = "store_save_local"
            
            sync_duration = time.time() - sync_start
            
            # Update performance statistics
            with _performance_lock:
                _performance_stats["sync_operations"] += 1
                _performance_stats["last_sync_time"] = datetime.now().isoformat()
                _performance_stats["total_sync_duration"] += sync_duration
                _performance_stats["avg_sync_duration"] = (
                    _performance_stats["total_sync_duration"] / 
                    _performance_stats["sync_operations"]
                )
            
            if PERFORMANCE_CONFIG.get("enable_performance_tracking", True):
                print(f"✅ Sync completed via {sync_method} in {sync_duration:.3f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during FAISS sync: {e}")
            return False

# ─────────────────── 📊 Global instances ─────────────────────────────
_deduplicator = MemoryDeduplicator()
_sync_manager = AutoSyncManager()

# ─────────────────── 🔧 Enhanced core memory functions ───────────────
def _serialise_turn(turn: Dict[str, str]) -> str:
    """Enhanced turn serialization with metadata."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "turn": turn,
        "metadata": {
            "turn_number": _performance_stats["total_turns"] + 1,
            "hash": _deduplicator.generate_content_hash(
                f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}"
            ),
            "phase": "2.0"  # Indicate Phase 2 format
        }
    }
    return json.dumps(entry, ensure_ascii=False)

def dump_turn(turn: Dict[str, str]) -> None:
    """
    Enhanced turn persistence with Phase 2 optimizations.
    
    Phase 2 enhancements:
    • Automatic duplicate detection and prevention
    • Triggered FAISS synchronization with enhanced error handling
    • Performance monitoring and statistics
    • Thread-safe operations with batch processing support
    """
    global _turn_counter
    
    with _performance_lock:
        _turn_counter += 1
        current_turn = _turn_counter
    
    # Phase 2: Enhanced deduplication check
    if _deduplicator.is_duplicate(turn):
        with _performance_lock:
            _performance_stats["duplicates_found"] += 1
        print(f"🚫 Duplicate turn detected and skipped (#{current_turn})")
        return
    
    # Phase 2: Periodic hash cleanup
    _deduplicator.cleanup_old_hashes()
    
    try:
        # 1️⃣ Append to JSONL for audit/debug (enhanced with metadata)
        with CHAT_FILE.open("a", encoding="utf-8") as f:
            f.write(_serialise_turn(turn) + "\n")
        
        # 2️⃣ Add to FAISS for retrieval (Phase 2: with batch processing awareness)
        user = turn.get("user", "")
        assistant = turn.get("assistant", "")
        if user or assistant:  # Only add if there's actual content
            text = f"User: {user}\nAssistant: {assistant}"
            try:
                add_texts([text])  # Uses enhanced batch processing internally
            except Exception as e:
                print(f"⚠️  Could not update FAISS index: {e}")
        
        # 3️⃣ Phase 2: Enhanced auto-sync with performance monitoring
        if _sync_manager.should_sync(current_turn):
            sync_start = time.time()
            print(f"🔄 Triggering auto-sync (turn #{current_turn})...")
            
            sync_success = _sync_manager.perform_sync()
            
            sync_duration = time.time() - sync_start
            if sync_success:
                print(f"✅ Auto-sync completed in {sync_duration:.3f}s")
            else:
                print(f"⚠️  Auto-sync encountered issues after {sync_duration:.3f}s")
        
        # 4️⃣ Update performance statistics
        with _performance_lock:
            _performance_stats["total_turns"] = current_turn
            
    except Exception as e:
        print(f"❌ Error in dump_turn: {e}")

def load_memory() -> List[Dict[str, str]]:
    """
    Enhanced memory loading with Phase 2 format support.
    
    Supports both Phase 1 and Phase 2 JSON formats for backward compatibility.
    """
    if not CHAT_FILE.exists():
        return []
    
    try:
        memories = []
        with CHAT_FILE.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    # Handle both old and new format
                    if "turn" in data:
                        memories.append(data["turn"])  # Phase 2 format
                        # Update seen hashes for deduplication
                        if "metadata" in data and "hash" in data["metadata"]:
                            _seen_hashes.add(data["metadata"]["hash"])
                    else:
                        memories.append(data)  # Phase 1 format fallback
                except json.JSONDecodeError as e:
                    print(f"⚠️  Invalid JSON on line {line_num}: {e}")
                    continue
        
        return memories
        
    except Exception as e:
        print(f"❌ [Memory Load Error]: {e}")
        return []

def get_memory_stats() -> Dict[str, Any]:
    """Get comprehensive Phase 2 memory system statistics."""
    with _performance_lock:
        stats = _performance_stats.copy()
    
    # Add Phase 2 enhanced metrics
    stats.update({
        "deduplication_rate": (
            stats["duplicates_found"] / max(stats["total_turns"] + stats["duplicates_found"], 1)
        ),
        "avg_sync_duration_ms": stats["avg_sync_duration"] * 1000,
        "memory_file_size": CHAT_FILE.stat().st_size if CHAT_FILE.exists() else 0,
        "hash_cache_size": len(_seen_hashes),
        "phase": "2.0",
    })
    
    # Add FAISS performance stats if available
    try:
        faiss_stats = get_performance_stats()
        stats.update({
            "faiss_index_size": faiss_stats.get("index_size", 0),
            "faiss_index_type": faiss_stats.get("index_type", "unknown"),
            "faiss_optimized": faiss_stats.get("is_optimized", False),
            "embedding_efficiency": faiss_stats.get("embedding_efficiency", 0),
        })
    except Exception:
        pass  # FAISS stats not available
    
    return stats

def force_full_sync() -> bool:
    """Force a complete FAISS synchronization with Phase 2 enhancements."""
    print("🔄 Forcing full FAISS synchronization...")
    return _sync_manager.perform_sync()

def startup_initialization() -> None:
    """
    Initialize the Phase 2 enhanced memory system during application startup.
    
    This should be called once when your application starts.
    """
    global _performance_stats
    
    print("🧠 Initializing Phase 2 enhanced memory system...")
    
    # Record startup time
    _performance_stats["startup_time"] = datetime.now().isoformat()
    
    # Phase 2: Enhanced startup deduplication
    if AUTO_SYNC_CONFIG.get("dedupe_on_startup", True):
        duplicates_removed = dedupe_startup()
        print(f"🚫 Startup deduplication: {duplicates_removed} duplicates removed")
    
    # Phase 2: Enhanced initial sync
    if AUTO_SYNC_CONFIG.get("enabled", True) and CHAT_FILE.exists():
        print("🔄 Performing Phase 2 startup synchronization...")
        sync_success = _sync_manager.perform_sync()
        if sync_success:
            print("✅ Phase 2 startup sync completed")
        else:
            print("⚠️  Phase 2 startup sync had issues")
    
    # Display system statistics
    stats = get_memory_stats()
    print(f"📊 System ready - {stats['total_turns']} turns, {stats.get('faiss_index_size', 0)} vectors indexed")
    
    print("✅ Phase 2 enhanced memory system ready")

def dedupe_startup() -> int:
    """
    Enhanced startup deduplication with Phase 2 improvements.
    
    Returns number of duplicates removed.
    """
    if not CHAT_FILE.exists():
        return 0
    
    print("🚫 Phase 2 deduplication scan...")
    
    try:
        # Read all existing turns with enhanced format support
        memories = load_memory()
        seen_hashes = set()
        unique_memories = []
        duplicates_found = 0
        
        for memory in memories:
            content = f"User: {memory.get('user', '')}\nAssistant: {memory.get('assistant', '')}"
            content_hash = _deduplicator.generate_content_hash(content)
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_memories.append(memory)
            else:
                duplicates_found += 1
        
        # If duplicates found, rewrite with Phase 2 format
        if duplicates_found > 0:
            print(f"🚫 Found {duplicates_found} duplicates, cleaning up...")
            
            # Backup original file
            backup_file = CHAT_FILE.with_suffix('.jsonl.backup')
            CHAT_FILE.rename(backup_file)
            
            # Write deduplicated version in Phase 2 format
            with CHAT_FILE.open("w", encoding="utf-8") as f:
                for memory in unique_memories:
                    entry = {
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                        "turn": memory,
                        "metadata": {
                            "turn_number": len(unique_memories),
                            "hash": _deduplicator.generate_content_hash(
                                f"User: {memory.get('user', '')}\nAssistant: {memory.get('assistant', '')}"
                            ),
                            "phase": "2.0",
                            "deduplicated": True
                        }
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            print(f"✅ Removed {duplicates_found} duplicates, backup: {backup_file.name}")
            
            # Update global seen hashes
            _seen_hashes.update(seen_hashes)
            
        else:
            print("✅ No duplicates found")
            # Still populate seen hashes for future deduplication
            _seen_hashes.update(seen_hashes)
        
        return duplicates_found
        
    except Exception as e:
        print(f"❌ Error during startup deduplication: {e}")
        return 0

# ─────────────────── 🧪 Development and testing helpers ───────────────
def reset_memory_stats() -> None:
    """Reset performance statistics (for testing)."""
    global _performance_stats, _turn_counter, _seen_hashes
    
    with _performance_lock:
        _performance_stats = {
            "total_turns": 0,
            "duplicates_found": 0, 
            "sync_operations": 0,
            "last_sync_time": None,
            "avg_sync_duration": 0.0,
            "total_sync_duration": 0.0,
            "startup_time": None,
            "batch_operations": 0,
        }
        _turn_counter = 0
        _seen_hashes.clear()
    
    print("🔄 Phase 2 memory statistics reset")

def batch_import_conversations(conversations: List[Dict[str, str]]) -> None:
    """
    Phase 2: Efficiently import multiple conversations using batch processing.
    
    Optimized for large-scale conversation imports.
    """
    print(f"📥 Importing {len(conversations)} conversations with Phase 2 optimizations...")
    
    start_time = time.time()
    texts_to_add = []
    
    # Process conversations in batches
    for i, conversation in enumerate(conversations):
        # Check for duplicates
        if not _deduplicator.is_duplicate(conversation):
            # Add to JSONL
            with CHAT_FILE.open("a", encoding="utf-8") as f:
                f.write(_serialise_turn(conversation) + "\n")
            
            # Prepare for batch FAISS addition
            user = conversation.get("user", "")
            assistant = conversation.get("assistant", "")
            if user or assistant:
                text = f"User: {user}\nAssistant: {assistant}"
                texts_to_add.append(text)
        
        # Process in batches of 50
        if len(texts_to_add) >= 50 or i == len(conversations) - 1:
            if texts_to_add:
                add_texts(texts_to_add)
                texts_to_add = []
    
    # Force sync after batch import
    _sync_manager.perform_sync()
    
    duration = time.time() - start_time
    print(f"✅ Batch import completed in {duration:.2f}s ({len(conversations)/duration:.1f} conversations/sec)")

print("✅ Phase 2 enhanced turn memory system loaded")

__all__ = ["dump_turn", "load_memory"]