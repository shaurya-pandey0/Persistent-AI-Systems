# vectorstore.py

"""
Thread-safe, persistent FAISS store with Phase 3 automated retraining.

Phase 1: Basic FAISS operations (WORKING) ✅
Phase 2: IndexIVFFlat optimization, batch processing (WORKING) ✅
Phase 3: Automated retraining, distribution drift monitoring (NEW) 🚀
# Implements FAISS search with BGE-M3 embeddings at the top.
"""

from __future__ import annotations

import json
import threading
import time
import statistics
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Any
from datetime import datetime, timedelta
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# Import Phase 3 configuration
try:
    from config.constants import AUTOMATED_RETRAINING_CONFIG
except ImportError:
    # Fallback configuration
    AUTOMATED_RETRAINING_CONFIG = {
        "enable_drift_monitoring": True,
        "drift_check_interval": 100,
        "drift_threshold": 0.15,
        "drift_sample_size": 50,
        "enable_auto_retraining": True,
        "retrain_on_drift": True,
        "retrain_performance_threshold": 0.20,
        "max_retraining_frequency": 24,
        "optimize_on_size_growth": True,
        "size_growth_threshold": 0.30,
        "optimize_on_query_pattern_change": True,
    }

# --------------------------------------------------------------------- #
# Constants and Configuration
# --------------------------------------------------------------------- #

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR = DATA_DIR / "faiss_index"
INDEX_DIR.mkdir(exist_ok=True)

# Phase 2 configuration (maintained for backward compatibility)
FAISS_OPTIMIZATION_CONFIG = {
    "use_ivf_index": True,
    "ncentroids": None,
    "batch_processing": True,
    "batch_size": 32,
    "performance_monitoring": True,
    "auto_optimize_threshold": 1000,
}

# Phase 3: Drift monitoring files
DRIFT_HISTORY_FILE = DATA_DIR / "drift_history.jsonl"
RETRAINING_LOG_FILE = DATA_DIR / "retraining_log.jsonl"

# --------------------------------------------------------------------- #
# Enhanced BGE-M3 Embedding with Phase 3 Monitoring
# --------------------------------------------------------------------- #

_bge_model = None
_model_lock = threading.Lock()
_performance_stats = {
    "embeddings_generated": 0,
    "batch_operations": 0,  
    "total_embedding_time": 0.0,
    "avg_embedding_time": 0.0,
    # Phase 3: Additional monitoring
    "drift_checks_performed": 0,
    "retraining_events": 0,
    "last_drift_check": None,
    "last_retraining": None,
}

def _get_bge_model() -> SentenceTransformer:
    """Get singleton BGE-M3 model with thread safety."""
    global _bge_model
    if _bge_model is None:
        with _model_lock:
            if _bge_model is None:  # double-checked locking
                print("🔄 Loading BGE-M3 model …")
                _bge_model = SentenceTransformer("BAAI/bge-m3")
                print("✅ BGE-M3 model loaded")
    return _bge_model

def _bge_embed_batch(texts: List[str], prefix: str = "passage") -> List[List[float]]:
    """
    Phase 2+3: Enhanced batch embedding with performance monitoring and drift tracking.
    """
    if not texts:
        return []
    
    start_time = time.time()
    
    # Add appropriate prefixes for BGE-M3
    prefixed_texts = [f"{prefix}: {text}" for text in texts]
    
    model = _get_bge_model()
    embeddings = model.encode(prefixed_texts, normalize_embeddings=True, batch_size=FAISS_OPTIMIZATION_CONFIG["batch_size"])
    
    # Update performance statistics
    embedding_time = time.time() - start_time
    with _model_lock:
        _performance_stats["embeddings_generated"] += len(texts)
        _performance_stats["batch_operations"] += 1
        _performance_stats["total_embedding_time"] += embedding_time
        _performance_stats["avg_embedding_time"] = (
            _performance_stats["total_embedding_time"] / _performance_stats["batch_operations"]
        )
    
    if FAISS_OPTIMIZATION_CONFIG["performance_monitoring"] and len(texts) > 1:
        print(f"📊 Batch embedded {len(texts)} texts in {embedding_time:.3f}s")
    
    return embeddings.tolist()

def _bge_embed_single(text: str, prefix: str = "passage") -> List[float]:
    """Single text embedding - uses batch processing internally for consistency."""
    return _bge_embed_batch([text], prefix)[0]

class BGEEmbeddings(Embeddings):
    """
    Phase 2+3: Enhanced LangChain-style embedding wrapper with monitoring.
    """
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents using batch processing."""
        if FAISS_OPTIMIZATION_CONFIG["batch_processing"] and len(texts) > 1:
            return _bge_embed_batch(texts, "passage")
        else:
            return [_bge_embed_single(text, "passage") for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query text."""
        return _bge_embed_single(text, "query")

# --------------------------------------------------------------------- #
# Phase 3: Automated Retraining and Drift Monitoring
# --------------------------------------------------------------------- #

class AutomatedRetrainingManager:
    """
    Phase 3: Manages automated retraining, drift detection,
    and performance monitoring for the FAISS index.
    """
    
    def __init__(self):
        self.config = AUTOMATED_RETRAINING_CONFIG
        self.drift_samples = []
        self.performance_baseline = None
        self.last_index_size = 0
        self.query_pattern_cache = []
        
    def should_check_drift(self, current_size: int) -> bool:
        """Determine if drift check should be performed."""
        if not self.config["enable_drift_monitoring"]:
            return False
        
        # Check based on interval
        size_increase = current_size - self.last_index_size
        return size_increase >= self.config["drift_check_interval"]
    
    def detect_distribution_drift(self, store: FAISS) -> Dict[str, Any]:
        """Phase 3: Detect distribution drift in the vector space."""
        try:
            current_time = datetime.now()
            
            # Sample vectors from current index
            sample_size = min(self.config["drift_sample_size"], store.index.ntotal)
            if sample_size < 10:
                return {"drift_detected": False, "reason": "insufficient_data"}
            
            # Get random sample of vectors
            sample_indices = np.random.choice(store.index.ntotal, sample_size, replace=False)
            current_samples = []
            
            for idx in sample_indices:
                # Extract vector from FAISS index
                vector = store.index.reconstruct(int(idx))
                current_samples.append(vector)
            
            current_samples = np.array(current_samples)
            
            # Compare with historical samples
            if len(self.drift_samples) > 0:
                # Calculate distribution shift
                drift_score = self._calculate_drift_score(current_samples, self.drift_samples[-1])
                
                drift_detected = drift_score > self.config["drift_threshold"]
                
                # Log drift check
                drift_info = {
                    "timestamp": current_time.isoformat(),
                    "drift_score": drift_score,
                    "drift_threshold": self.config["drift_threshold"],
                    "drift_detected": drift_detected,
                    "sample_size": sample_size,
                    "index_size": store.index.ntotal
                }
                
                self._log_drift_check(drift_info)
                
                # Update performance stats
                with _model_lock:
                    _performance_stats["drift_checks_performed"] += 1
                    _performance_stats["last_drift_check"] = current_time.isoformat()
                
                if drift_detected:
                    print(f"🚨 Distribution drift detected! Score: {drift_score:.3f} > {self.config['drift_threshold']:.3f}")
                
                # Store current samples for next comparison
                self.drift_samples.append(current_samples)
                
                # Keep only recent samples
                if len(self.drift_samples) > 10:
                    self.drift_samples = self.drift_samples[-5:]
                
                return drift_info
            else:
                # First sample - establish baseline
                self.drift_samples.append(current_samples)
                print(f"📊 Established drift monitoring baseline with {sample_size} samples")
                
                return {
                    "drift_detected": False,
                    "reason": "baseline_established",
                    "baseline_size": sample_size
                }
                
        except Exception as e:
            print(f"❌ Error detecting drift: {e}")
            return {"drift_detected": False, "error": str(e)}
    
    def _calculate_drift_score(self, current: np.ndarray, previous: np.ndarray) -> float:
        """Calculate drift score between current and previous distributions."""
        try:
            # Calculate mean shift (Euclidean distance between centroids)
            current_centroid = np.mean(current, axis=0)
            previous_centroid = np.mean(previous, axis=0)
            centroid_shift = np.linalg.norm(current_centroid - previous_centroid)
            
            # Calculate variance shift
            current_var = np.var(current, axis=0).mean()
            previous_var = np.var(previous, axis=0).mean()
            variance_shift = abs(current_var - previous_var) / max(previous_var, 1e-6)
            
            # Combine shifts into drift score
            drift_score = centroid_shift * 0.7 + variance_shift * 0.3
            
            return drift_score
            
        except Exception as e:
            print(f"❌ Error calculating drift score: {e}")
            return 0.0
    
    def _log_drift_check(self, drift_info: Dict[str, Any]) -> None:
        """Log drift check results."""
        try:
            with DRIFT_HISTORY_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(drift_info) + "\n")
        except Exception as e:
            print(f"Warning: Could not log drift check: {e}")
    
    def should_trigger_retraining(self, drift_info: Dict[str, Any], performance_degradation: float = 0.0) -> bool:
        """Determine if retraining should be triggered."""
        if not self.config["enable_auto_retraining"]:
            return False
        
        # Check retraining frequency limit
        if not self._can_retrain_now():
            return False
        
        # Trigger on drift detection
        if self.config["retrain_on_drift"] and drift_info.get("drift_detected", False):
            return True
        
        # Trigger on performance degradation
        if performance_degradation > self.config["retrain_performance_threshold"]:
            return True
        
        return False
    
    def _can_retrain_now(self) -> bool:
        """Check if retraining is allowed based on frequency limits."""
        if _performance_stats["last_retraining"] is None:
            return True
        
        try:
            last_retraining = datetime.fromisoformat(_performance_stats["last_retraining"])
            hours_since = (datetime.now() - last_retraining).total_seconds() / 3600
            return hours_since >= self.config["max_retraining_frequency"]
        except:
            return True
    
    def trigger_retraining(self, store: FAISS, reason: str = "drift_detected") -> bool:
        """Phase 3: Trigger automated retraining of the FAISS index."""
        try:
            print(f"🔄 Triggering automated retraining: {reason}")
            start_time = time.time()
            
            # Perform retraining by rebuilding index
            current_size = store.index.ntotal
            
            # Extract all documents from store
            documents = self._extract_all_documents(store)
            
            if not documents:
                print("❌ No documents found for retraining")
                return False
            
            # Create new optimized index
            new_store = _create_optimized_faiss_index(documents)
            
            # Replace global store
            global _store
            _store = new_store
            
            # Save the retrained index
            new_store.save_local(INDEX_DIR.as_posix())
            
            # Log retraining event
            retraining_time = time.time() - start_time
            retraining_info = {
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "retraining_time_seconds": retraining_time,
                "index_size_before": current_size,
                "index_size_after": new_store.index.ntotal,
                "success": True
            }
            
            self._log_retraining_event(retraining_info)
            
            # Update performance stats
            with _model_lock:
                _performance_stats["retraining_events"] += 1
                _performance_stats["last_retraining"] = datetime.now().isoformat()
            
            print(f"✅ Automated retraining completed in {retraining_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"❌ Automated retraining failed: {e}")
            
            # Log failed retraining
            retraining_info = {
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "success": False,
                "error": str(e)
            }
            self._log_retraining_event(retraining_info)
            
            return False
    
    def _extract_all_documents(self, store: FAISS) -> List[Document]:
        """Extract all documents from FAISS store."""
        try:
            documents = []
            docstore = store.docstore
            index_to_docstore_id = store.index_to_docstore_id
            
            for i in range(store.index.ntotal):
                doc_id = index_to_docstore_id.get(i)
                if doc_id and doc_id in docstore._dict:
                    documents.append(docstore._dict[doc_id])
            
            return documents
            
        except Exception as e:
            print(f"❌ Error extracting documents: {e}")
            return []
    
    def _log_retraining_event(self, retraining_info: Dict[str, Any]) -> None:
        """Log retraining event."""
        try:
            with RETRAINING_LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(retraining_info) + "\n")
        except Exception as e:
            print(f"Warning: Could not log retraining event: {e}")

# Create global retraining manager
_retraining_manager = AutomatedRetrainingManager()

# --------------------------------------------------------------------- #
# Enhanced FAISS Store with Phase 3 Features
# --------------------------------------------------------------------- #

_store: Optional[FAISS] = None
_store_lock = threading.Lock()
_embeddings = BGEEmbeddings()

def _calculate_optimal_ncentroids(ntotal: int) -> int:
    """Calculate optimal number of centroids for IndexIVFFlat."""
    if ntotal < 100:
        return min(8, ntotal)
    
    optimal = int(4 * np.sqrt(ntotal))
    return max(8, min(optimal, 65536))

def _create_optimized_faiss_index(documents: List[Document]) -> FAISS:
    """
    Phase 2+3: Create optimized FAISS index with automated retraining support.
    """
    if not documents:
        documents = [Document(page_content="No conversations yet")]
    
    ntotal = len(documents)
    use_ivf = (
        FAISS_OPTIMIZATION_CONFIG["use_ivf_index"] and 
        ntotal >= FAISS_OPTIMIZATION_CONFIG["auto_optimize_threshold"]
    )
    
    if use_ivf:
        ncentroids = (
            FAISS_OPTIMIZATION_CONFIG["ncentroids"] or 
            _calculate_optimal_ncentroids(ntotal)
        )
        
        print(f"🔧 Creating IndexIVFFlat with {ncentroids} centroids for {ntotal} vectors")
        
        # Create IVF index with batch processing
        embeddings_array = np.array(_embeddings.embed_documents([doc.page_content for doc in documents]))
        
        import faiss
        dimension = embeddings_array.shape[1]
        
        # Create and train IVF index
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, ncentroids)
        
        # Train index with embeddings
        print(f"🎯 Training IVF index with {ntotal} vectors...")
        index.train(embeddings_array.astype('float32'))
        
        # Create FAISS store with trained index
        store = FAISS(embedding_function=_embeddings, index=index, docstore={}, index_to_docstore_id={})
        
        # Add documents to store
        store.add_documents(documents)
        
        print(f"✅ IndexIVFFlat created and trained successfully")
        
    else:
        # Use standard IndexFlatL2 for smaller datasets
        print(f"📦 Creating IndexFlatL2 for {ntotal} vectors (under threshold)")
        store = FAISS.from_documents(documents, _embeddings)
    
    return store

def _load_or_create_store() -> FAISS:
    """Phase 2+3: Enhanced store loading with drift monitoring."""
    if INDEX_DIR.joinpath("index.faiss").exists():
        print(f"📖 Loading FAISS index from {INDEX_DIR} …")
        try:
            store = FAISS.load_local(
                INDEX_DIR.as_posix(),
                _embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ FAISS index loaded")
            
            # Phase 3: Initialize baseline size for drift monitoring
            _retraining_manager.last_index_size = store.index.ntotal
            
            # Check if optimization is needed
            current_size = store.index.ntotal
            should_optimize = (
                FAISS_OPTIMIZATION_CONFIG["use_ivf_index"] and
                current_size >= FAISS_OPTIMIZATION_CONFIG["auto_optimize_threshold"] and
                not hasattr(store.index, 'nlist')
            )
            
            if should_optimize:
                print(f"🔄 Auto-optimizing index ({current_size} vectors)")
                store = _upgrade_to_ivf_index(store)
            
            return store
            
        except Exception as e:
            print(f"⚠️  Error loading existing index: {e}")
            print("🆕 Creating new optimized index...")
    
    # Create new optimized store
    print("🆕 No FAISS index on disk – creating new optimized index")
    dummy = Document(page_content="No conversations yet")
    store = _create_optimized_faiss_index([dummy])
    store.save_local(INDEX_DIR.as_posix())
    
    # Initialize baseline for drift monitoring
    _retraining_manager.last_index_size = store.index.ntotal
    
    return store

def _upgrade_to_ivf_index(existing_store: FAISS) -> FAISS:
    """Phase 2+3: Upgrade existing flat index to IndexIVFFlat."""
    print("🔄 Upgrading to IndexIVFFlat for better performance...")
    
    try:
        # Extract documents from existing store
        documents = _retraining_manager._extract_all_documents(existing_store)
        
        # Create new optimized store
        optimized_store = _create_optimized_faiss_index(documents)
        
        # Save optimized store
        optimized_store.save_local(INDEX_DIR.as_posix())
        
        print("✅ Index upgraded to IndexIVFFlat successfully")
        return optimized_store
        
    except Exception as e:
        print(f"⚠️  Index upgrade failed: {e}, keeping existing index")
        return existing_store

def get_store() -> FAISS:
    """Public accessor with Phase 3 drift monitoring."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = _load_or_create_store()
    return _store

# --------------------------------------------------------------------- #
# Enhanced Public API with Phase 3 Features
# --------------------------------------------------------------------- #

def add_texts(texts: List[str]) -> None:
    """
    Phase 2+3: Enhanced batch text addition with automated retraining.
    """
    if not texts:
        return

    start_time = time.time()
    store = get_store()
    
    # Record size before addition
    size_before = store.index.ntotal

    # Use batch processing for multiple texts
    if FAISS_OPTIMIZATION_CONFIG["batch_processing"] and len(texts) > 1:
        store.add_texts(texts)
    else:
        for text in texts:
            store.add_texts([text])

    # Save updated index
    store.save_local(INDEX_DIR.as_posix())
    
    # Phase 3: Check for drift and potential retraining
    size_after = store.index.ntotal
    
    if _retraining_manager.should_check_drift(size_after):
        print(f"🔍 Performing drift check ({size_after} vectors)")
        drift_info = _retraining_manager.detect_distribution_drift(store)
        
        if _retraining_manager.should_trigger_retraining(drift_info):
            print(f"🔄 Triggering retraining due to drift detection")
            _retraining_manager.trigger_retraining(store, "drift_detected")
        
        # Update size tracking  
        _retraining_manager.last_index_size = size_after

    # Performance monitoring
    if FAISS_OPTIMIZATION_CONFIG["performance_monitoring"]:
        duration = time.time() - start_time
        print(f"📈 Added {len(texts)} texts in {duration:.3f}s")

def search(query: str, k: int = 4) -> List[Tuple[str, float]]:
    """
    Phase 2+3: Enhanced search with drift monitoring.
    """
    store = get_store()
    docs_and_scores = store.similarity_search_with_score(query, k=k)
    
    # Phase 3: Track query patterns for drift detection
    _retraining_manager.query_pattern_cache.append({
        "query": query[:50],  # Truncate for privacy
        "timestamp": datetime.now().isoformat(),
        "result_count": len(docs_and_scores)
    })
    
    # Keep cache manageable
    if len(_retraining_manager.query_pattern_cache) > 1000:
        _retraining_manager.query_pattern_cache = _retraining_manager.query_pattern_cache[-500:]
    
    return [(doc.page_content, score) for doc, score in docs_and_scores]

def get_performance_stats() -> dict:
    """Get comprehensive performance statistics with Phase 3 metrics."""
    with _model_lock:
        stats = _performance_stats.copy()
    
    store = get_store()
    stats.update({
        "index_size": store.index.ntotal,
        "index_type": type(store.index).__name__,
        "is_optimized": hasattr(store.index, 'nlist'),
        "embedding_efficiency": stats["embeddings_generated"] / max(stats["batch_operations"], 1),
        # Phase 3: Automated retraining stats
        "drift_monitoring_enabled": AUTOMATED_RETRAINING_CONFIG["enable_drift_monitoring"],
        "auto_retraining_enabled": AUTOMATED_RETRAINING_CONFIG["enable_auto_retraining"],
        "drift_checks_performed": stats["drift_checks_performed"],
        "retraining_events": stats["retraining_events"],
        "last_drift_check": stats["last_drift_check"],
        "last_retraining": stats["last_retraining"],
        "query_patterns_tracked": len(_retraining_manager.query_pattern_cache),
    })
    
    return stats

def get_drift_monitoring_status() -> Dict[str, Any]:
    """Phase 3: Get drift monitoring status and history."""
    try:
        # Load recent drift history
        drift_history = []
        if DRIFT_HISTORY_FILE.exists():
            with DRIFT_HISTORY_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    drift_history.append(json.loads(line))
        
        # Load retraining history
        retraining_history = []
        if RETRAINING_LOG_FILE.exists():
            with RETRAINING_LOG_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    retraining_history.append(json.loads(line))
        
        # Calculate drift trends
        recent_drift_scores = [
            entry["drift_score"] for entry in drift_history[-10:]
            if "drift_score" in entry
        ]
        
        return {
            "drift_monitoring_enabled": AUTOMATED_RETRAINING_CONFIG["enable_drift_monitoring"],
            "drift_threshold": AUTOMATED_RETRAINING_CONFIG["drift_threshold"],
            "total_drift_checks": len(drift_history),
            "total_retraining_events": len(retraining_history),
            "recent_drift_scores": recent_drift_scores,
            "avg_recent_drift": statistics.mean(recent_drift_scores) if recent_drift_scores else 0.0,
            "drift_trend": "increasing" if len(recent_drift_scores) > 3 and recent_drift_scores[-1] > statistics.mean(recent_drift_scores[:-1]) else "stable",
            "last_drift_check": _performance_stats["last_drift_check"],
            "last_retraining": _performance_stats["last_retraining"],
            "can_retrain_now": _retraining_manager._can_retrain_now()
        }
        
    except Exception as e:
        return {"error": str(e), "drift_monitoring_enabled": False}

def force_retraining(reason: str = "manual_trigger") -> bool:
    """Phase 3: Manually force retraining of the FAISS index."""
    store = get_store()
    return _retraining_manager.trigger_retraining(store, reason)

# Legacy compatibility
def optimize_index_if_needed() -> bool:
    """Maintained for backward compatibility."""
    store = get_store()
    current_size = store.index.ntotal
    
    should_optimize = (
        FAISS_OPTIMIZATION_CONFIG["use_ivf_index"] and
        current_size >= FAISS_OPTIMIZATION_CONFIG["auto_optimize_threshold"] and
        not hasattr(store.index, 'nlist')
    )
    
    if should_optimize:
        print(f"🔄 Manual optimization triggered for {current_size} vectors")
        global _store
        _store = _upgrade_to_ivf_index(store)
        return True
    
    print(f"ℹ️  No optimization needed (size: {current_size}, optimized: {hasattr(store.index, 'nlist')})")
    return False

def build_store():
    """Legacy function - now just returns the singleton store."""
    return get_store()

# ─────────────────── 🧪 Phase 3: Testing and debugging ──────────────
def test_phase3_features():
    """Test Phase 3 automated retraining features."""
    print("🧪 Testing Phase 3 Automated Retraining Features")
    print("=" * 55)
    
    # Test drift monitoring status
    drift_status = get_drift_monitoring_status()
    print(f"🔍 Drift monitoring: {'✅ Enabled' if drift_status['drift_monitoring_enabled'] else '❌ Disabled'}")
    print(f"📊 Total drift checks: {drift_status['total_drift_checks']}")
    print(f"🔄 Total retraining events: {drift_status['total_retraining_events']}")
    
    # Test performance stats
    stats = get_performance_stats()
    print(f"📈 Index size: {stats['index_size']} vectors")
    print(f"🔧 Index type: {stats['index_type']}")
    print(f"⚡ Embedding efficiency: {stats['embedding_efficiency']:.2f}")
    
    print("\n✅ Phase 3 automated retraining test completed")

if __name__ == "__main__":
    test_phase3_features()

__all__ = ["get_store"]
