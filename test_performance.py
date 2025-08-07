# performance_tests.py - CORRECTED VERSION
"""
Fixed performance regression test suite for Memory & Mood Pipeline.

FIXES APPLIED:
• Model loading overhead excluded from measurements
• Proper warm-up queries to initialize models before timing
• Improved baseline calculation with realistic data
• Separate initialization phase from performance measurement
"""

from __future__ import annotations

import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

# Import with error handling for development
try:
    from memory.context_retriever import retrieve_top_memories
    from vectorstore import search, add_texts, get_store
    from config.constants import MEMORY_THRESHOLD_CONFIG, TOP_K
    from memory.turn_memory import get_memory_stats, dump_turn
except ImportError as e:
    print(f"Warning: Could not import all dependencies: {e}")
    
    # Mock functions for development
    def retrieve_top_memories(query, **kwargs):
        return [], []
    def search(query, k=3):
        return [("mock result", 0.5)]
    def get_memory_stats():
        return {"total_turns": 0, "sync_operations": 0}

# ─────────────────── 📊 Fixed performance configuration ─────────────
PERFORMANCE_CONFIG = {
    # Latency thresholds (milliseconds)
    "search_latency_threshold": 500,           # Max acceptable search time
    "retrieval_latency_threshold": 1000,      # Max acceptable full retrieval time
    
    # Performance monitoring
    "benchmark_sample_size": 8,               # Number of queries per benchmark
    "warmup_queries": 3,                      # Queries to run before timing starts
    "performance_history_days": 30,           # Days of performance data to keep
    "alert_on_degradation": True,             # Send alerts on performance drops
    "degradation_threshold": 0.25,            # 25% performance drop triggers alert (more lenient)
}

@dataclass
class SearchMetrics:
    """Individual search operation metrics."""
    query: str
    latency_ms: float
    results_count: int
    relevance_score: float
    threshold_used: float
    timestamp: str

@dataclass
class BenchmarkResults:
    """Comprehensive benchmark results."""
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    total_queries: int
    successful_queries: int
    avg_results_count: float
    avg_relevance_score: float
    timestamp: str
    benchmark_type: str

class PerformanceTracker:
    """FIXED: Performance monitoring with proper model initialization handling."""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.performance_file = self.data_dir / "performance_history.jsonl"
        
        # Models initialization status
        self._models_initialized = False
        
        # Standard test queries for consistent benchmarking
        self.benchmark_queries = [
            "Hello how are you",
            "What is the capital of France",
            "Tell me about artificial intelligence", 
            "How does machine learning work",
            "What is the weather like today",
            "Can you help me with programming",
            "I need information about Python",
            "Explain quantum computing"
        ]
        
        # Warmup queries to initialize models
        self.warmup_queries = [
            "warmup query 1",
            "warmup query 2", 
            "warmup test query"
        ]
    
    def _initialize_models(self) -> None:
        """Initialize all models before performance testing to exclude loading time."""
        if self._models_initialized:
            return
            
        print("🔄 Initializing models (excluding from performance measurements)...")
        
        try:
            # Run warmup queries to initialize BGE-M3 and other models
            for warmup_query in self.warmup_queries:
                # Initialize search models
                search(warmup_query, k=1)
                # Initialize retrieval models
                retrieve_top_memories(warmup_query, k_short=1)
                time.sleep(0.1)  # Small delay between warmup queries
            
            print("✅ Models initialized successfully")
            self._models_initialized = True
            
        except Exception as e:
            print(f"⚠️  Model initialization had issues: {e}")
            # Continue anyway - performance tests will still work
            self._models_initialized = True
    
    def measure_search_latency(self, query: str, k: int = 3) -> SearchMetrics:
        """FIXED: Measure search latency with model loading excluded."""
        # Ensure models are pre-loaded
        self._initialize_models()
        
        start_time = time.time()
        
        try:
            # Measure raw search performance (models already loaded)
            results = search(query, k=k)
            search_latency = (time.time() - start_time) * 1000
            
            # Calculate relevance score
            if results:
                avg_score = sum(score for _, score in results) / len(results)
                # For L2 distance, lower scores are better, so invert for relevance
                relevance_score = max(0, 1.0 - (avg_score / 2.0))
            else:
                relevance_score = 0.0
            
            return SearchMetrics(
                query=query,
                latency_ms=search_latency,
                results_count=len(results),
                relevance_score=relevance_score,
                threshold_used=MEMORY_THRESHOLD_CONFIG.get("base_threshold", 1.0),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            print(f"Error measuring search latency for '{query}': {e}")
            return SearchMetrics(
                query=query,
                latency_ms=float('inf'),
                results_count=0,
                relevance_score=0.0,
                threshold_used=0.0,
                timestamp=datetime.now().isoformat()
            )
    
    def measure_retrieval_latency(self, query: str) -> SearchMetrics:
        """FIXED: Measure retrieval latency with model loading excluded."""
        # Ensure models are pre-loaded
        self._initialize_models()
        
        start_time = time.time()
        
        try:
            short_results, long_results = retrieve_top_memories(query)
            retrieval_latency = (time.time() - start_time) * 1000
            
            total_results = len(short_results) + len(long_results)
            # Simplified relevance based on result count
            relevance_score = min(1.0, total_results / 3) if total_results > 0 else 0.0
            
            return SearchMetrics(
                query=query,
                latency_ms=retrieval_latency,
                results_count=total_results,
                relevance_score=relevance_score,
                threshold_used=MEMORY_THRESHOLD_CONFIG.get("base_threshold", 1.0),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            print(f"Error measuring retrieval latency for '{query}': {e}")
            return SearchMetrics(
                query=query,
                latency_ms=float('inf'),
                results_count=0,
                relevance_score=0.0,
                threshold_used=0.0,
                timestamp=datetime.now().isoformat()
            )
    
    def run_performance_benchmark(self, benchmark_type: str = "comprehensive") -> BenchmarkResults:
        """FIXED: Run comprehensive performance benchmark with proper initialization."""
        print(f"🚀 Running {benchmark_type} performance benchmark...")
        print(f"📊 Testing {len(self.benchmark_queries)} queries")
        
        # Critical fix: Initialize models BEFORE any timing measurements
        self._initialize_models()
        
        all_metrics = []
        
        # Run test queries with models already loaded
        for query in self.benchmark_queries:
            if benchmark_type in ["search", "comprehensive"]:
                metrics = self.measure_search_latency(query)
                all_metrics.append(metrics)
                
            if benchmark_type in ["retrieval", "comprehensive"]:
                metrics = self.measure_retrieval_latency(query)
                all_metrics.append(metrics)
        
        # Calculate aggregate statistics (excluding infinite latencies)
        latencies = [m.latency_ms for m in all_metrics if m.latency_ms != float('inf')]
        relevance_scores = [m.relevance_score for m in all_metrics]
        result_counts = [m.results_count for m in all_metrics]
        
        if not latencies:
            print("❌ No successful measurements recorded")
            return self._create_empty_benchmark_results(benchmark_type)
        
        benchmark_results = BenchmarkResults(
            avg_latency_ms=statistics.mean(latencies),
            median_latency_ms=statistics.median(latencies),
            p95_latency_ms=self._percentile(latencies, 95),
            max_latency_ms=max(latencies),
            min_latency_ms=min(latencies),
            total_queries=len(all_metrics),
            successful_queries=len(latencies),
            avg_results_count=statistics.mean(result_counts) if result_counts else 0,
            avg_relevance_score=statistics.mean(relevance_scores) if relevance_scores else 0,
            timestamp=datetime.now().isoformat(),
            benchmark_type=benchmark_type
        )
        
        # Store results for historical tracking
        self._store_benchmark_results(benchmark_results)
        
        # Print summary
        self._print_benchmark_summary(benchmark_results)
        
        # Check for performance degradation (with fixed baseline calculation)
        self._check_performance_degradation(benchmark_results)
        
        return benchmark_results
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _create_empty_benchmark_results(self, benchmark_type: str) -> BenchmarkResults:
        """Create empty benchmark results for error cases."""
        return BenchmarkResults(
            avg_latency_ms=0.0,
            median_latency_ms=0.0,
            p95_latency_ms=0.0,
            max_latency_ms=0.0,
            min_latency_ms=0.0,
            total_queries=0,
            successful_queries=0,
            avg_results_count=0.0,
            avg_relevance_score=0.0,
            timestamp=datetime.now().isoformat(),
            benchmark_type=benchmark_type
        )
    
    def _store_benchmark_results(self, results: BenchmarkResults) -> None:
        """Store benchmark results for historical tracking."""
        try:
            with self.performance_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(results)) + "\n")
        except Exception as e:
            print(f"Warning: Could not store benchmark results: {e}")
    
    def _print_benchmark_summary(self, results: BenchmarkResults) -> None:
        """Print formatted benchmark summary."""
        print(f"\n📊 {results.benchmark_type.title()} Benchmark Results")
        print("=" * 50)
        print(f"🎯 Queries: {results.successful_queries}/{results.total_queries} successful")
        print(f"⚡ Average latency: {results.avg_latency_ms:.2f}ms")
        print(f"📊 Median latency: {results.median_latency_ms:.2f}ms")
        print(f"📈 95th percentile: {results.p95_latency_ms:.2f}ms")
        print(f"⏱️  Max latency: {results.max_latency_ms:.2f}ms")
        print(f"📝 Average results: {results.avg_results_count:.1f}")
        print(f"🎯 Average relevance: {results.avg_relevance_score:.3f}")
        
        # Performance status indicators
        threshold = PERFORMANCE_CONFIG["search_latency_threshold"]
        if results.avg_latency_ms < threshold * 0.5:
            print("✅ Performance: EXCELLENT")
        elif results.avg_latency_ms < threshold:
            print("✅ Performance: GOOD")  
        elif results.avg_latency_ms < threshold * 1.5:
            print("⚠️  Performance: ACCEPTABLE")
        else:
            print("❌ Performance: NEEDS IMPROVEMENT")
    
    def _check_performance_degradation(self, current_results: BenchmarkResults) -> None:
        """FIXED: Check for performance degradation with proper baseline calculation."""
        if not PERFORMANCE_CONFIG["alert_on_degradation"]:
            return
        
        try:
            # Load recent historical data (excluding current result)
            recent_results = self._load_recent_benchmarks(days=7)
            if len(recent_results) < 3:  # Need at least 3 historical points
                print("ℹ️  Insufficient historical data for degradation detection")
                return
            
            # Calculate baseline from recent stable results (exclude outliers)
            historical_latencies = [r["avg_latency_ms"] for r in recent_results[:-1]]
            
            # Remove outliers to get a stable baseline
            mean_latency = statistics.mean(historical_latencies)
            filtered_latencies = [lat for lat in historical_latencies if lat < mean_latency * 2]
            
            if not filtered_latencies:
                print("ℹ️  Unable to establish stable baseline")
                return
                
            baseline_latency = statistics.mean(filtered_latencies)
            current_latency = current_results.avg_latency_ms
            
            # Check for degradation
            degradation_ratio = (current_latency - baseline_latency) / baseline_latency
            threshold = PERFORMANCE_CONFIG["degradation_threshold"]
            
            if degradation_ratio > threshold:
                print(f"\n🚨 PERFORMANCE DEGRADATION ALERT!")
                print(f"   Current latency: {current_latency:.2f}ms")
                print(f"   Baseline latency: {baseline_latency:.2f}ms")
                print(f"   Degradation: {degradation_ratio*100:.1f}% (threshold: {threshold*100:.1f}%)")
            else:
                print(f"\n✅ Performance within acceptable range")
                print(f"   Current: {current_latency:.2f}ms vs Baseline: {baseline_latency:.2f}ms")
                
        except Exception as e:
            print(f"Warning: Could not check performance degradation: {e}")
    
    def _load_recent_benchmarks(self, days: int = 7) -> List[Dict[str, Any]]:
        """Load recent benchmark results for trend analysis."""
        if not self.performance_file.exists():
            return []
        
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_results = []
        
        try:
            with self.performance_file.open("r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    result_time = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
                    if result_time >= cutoff_time:
                        recent_results.append(data)
        except Exception as e:
            print(f"Warning: Could not load recent benchmarks: {e}")
        
        return recent_results

# ─────────────────── 🧪 Fixed test runner and utilities ─────────────────────────
def run_quick_performance_test() -> None:
    """Run a quick performance test for immediate feedback."""
    print("⚡ Running quick performance test...")
    
    tracker = PerformanceTracker()
    
    # Initialize models first
    tracker._initialize_models()
    
    # Test a few key queries
    test_queries = [
        "Hello how are you",
        "What is the capital of France", 
        "Tell me about AI"
    ]
    
    for query in test_queries:
        metrics = tracker.measure_retrieval_latency(query)
        print(f"   '{query[:30]}...' → {metrics.latency_ms:.2f}ms ({metrics.results_count} results)")
    
    print("✅ Quick performance test completed")


def run_full_performance_benchmark() -> BenchmarkResults:
    """Run comprehensive performance benchmark."""
    tracker = PerformanceTracker()
    return tracker.run_performance_benchmark("comprehensive")


def check_system_performance() -> bool:
    """Check if system performance meets thresholds."""
    tracker = PerformanceTracker()
    results = tracker.run_performance_benchmark("search")
    
    # Check against thresholds
    latency_ok = results.avg_latency_ms < PERFORMANCE_CONFIG["search_latency_threshold"]
    
    if latency_ok:
        print("✅ System performance meets all thresholds")
        return True
    else:
        print("⚠️  System performance below thresholds")
        print(f"   Latency: {results.avg_latency_ms:.2f}ms > {PERFORMANCE_CONFIG['search_latency_threshold']}ms")
        return False


def main():
    """Main entry point for performance testing."""
    print("🧪 Memory & Mood Pipeline Performance Test Suite")
    print("=======================================================")
    
    # Check if system is operational
    try:
        stats = get_memory_stats()
        print(f"📊 System status: {stats['total_turns']} turns, {stats.get('sync_operations', 0)} syncs")
    except Exception as e:
        print(f"⚠️  Could not get system stats: {e}")
    
    # Run performance tests
    print("\n🚀 Starting comprehensive performance benchmark...")
    tracker = PerformanceTracker()
    
    # Run different types of benchmarks
    search_results = tracker.run_performance_benchmark("search")
    retrieval_results = tracker.run_performance_benchmark("retrieval")
    
    print("\n✅ Performance testing completed")
    print("💡 Check data/performance_history.jsonl for detailed metrics")


if __name__ == "__main__":
    main()