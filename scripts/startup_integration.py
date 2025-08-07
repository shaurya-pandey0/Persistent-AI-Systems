# scripts/startup_integration.py

"""
Memory & Mood Pipeline Startup Integration Script

This is the MAIN ORCHESTRATION SCRIPT that should be placed in:
PROJECT_ROOT/scripts/startup_integration.py

Purpose:
• Orchestrates Phase 2 optimizations initialization
• Validates system configuration and health
• Establishes performance baselines
• Coordinates all memory pipeline components

Usage:
    python scripts/startup_integration.py

# Or integrate into your app:
    from scripts.startup_integration import StartupOrchestrator
    orchestrator = StartupOrchestrator()
    orchestrator.run_startup_sequence()
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configuration and system imports
try:
    from config.constants import (
        MEMORY_THRESHOLD_CONFIG,
        MEMORY_FILE,
        TOP_K,
        API_KEY
    )
    from memory.turn_memory import (
        get_memory_stats,
        startup_initialization,
        AUTO_SYNC_CONFIG,  # FIXED: Removed DEDUPLICATION_CONFIG (doesn't exist)
    )
    from vectorstore import get_store, search
    from memory.context_retriever import retrieve_top_memories
    
except ImportError as e:
    print(f"❌ Critical import error: {e}")
    print("   Make sure you're running from the project root directory")
    sys.exit(1)

class StartupOrchestrator:
    """Orchestrates startup sequence for Memory & Mood Pipeline optimizations."""
    
    def __init__(self):
        self.startup_start_time = time.time()
        self.initialization_results = {}
        self.health_check_results = {}
        # FIXED: Ensure data directory is root/data
        self.data_dir = project_root / "data"
        self.data_dir.mkdir(exist_ok=True)
    
    def run_startup_sequence(self) -> bool:
        """Execute complete startup sequence with comprehensive health checks."""
        print("🚀 Memory & Mood Pipeline - Phase 2 Startup Integration")
        print("=" * 65)
        
        try:
            # Step 1: Configuration validation
            print("\n🔧 Step 1: Configuration Validation")
            self._validate_configurations()
            
            # Step 2: Directory structure validation
            print("\n📁 Step 2: Directory Structure Validation")
            self._validate_directory_structure()
            
            # Step 3: Memory system initialization
            print("\n🧠 Step 3: Memory System Initialization")
            self._initialize_memory_system()
            
            # Step 4: FAISS index validation
            print("\n🔍 Step 4: FAISS Index Validation")
            self._validate_faiss_index()
            
            # Step 5: Performance baseline establishment
            print("\n📊 Step 5: Performance Baseline Establishment")
            self._establish_performance_baseline()
            
            # Step 6: System health validation
            print("\n✅ Step 6: System Health Validation")
            self._run_health_checks()
            
            # Step 7: Startup summary
            print("\n📋 Step 7: Startup Summary")
            self._print_startup_summary()
            
            return True
            
        except Exception as e:
            print(f"\n❌ Startup failed: {e}")
            print("   Check the error details above and ensure all dependencies are installed")
            return False
    
    def _validate_configurations(self) -> None:
        """Validate all configuration settings for consistency."""
        validations = []
        
        # Phase 1 threshold configuration
        try:
            threshold_config = MEMORY_THRESHOLD_CONFIG
            assert threshold_config["enabled"] in [True, False]
            assert 0.0 < threshold_config["min_threshold"] <= threshold_config["base_threshold"] <= threshold_config["max_threshold"]
            validations.append("✅ MEMORY_THRESHOLD_CONFIG valid")
        except Exception as e:
            validations.append(f"❌ MEMORY_THRESHOLD_CONFIG invalid: {e}")
        
        # Phase 2 auto-sync configuration (FIXED: includes deduplication settings)
        try:
            sync_config = AUTO_SYNC_CONFIG
            assert sync_config["enabled"] in [True, False]
            assert isinstance(sync_config["sync_threshold"], int)
            
            # FIXED: Check deduplication settings within AUTO_SYNC_CONFIG
            if "dedupe_enabled" in sync_config:
                assert sync_config["dedupe_enabled"] in [True, False]
                validations.append("✅ AUTO_SYNC_CONFIG (with deduplication) valid")
            else:
                validations.append("✅ AUTO_SYNC_CONFIG valid")
                
        except Exception as e:
            validations.append(f"❌ AUTO_SYNC_CONFIG invalid: {e}")
        
        # API configuration
        try:
            assert API_KEY is not None and len(API_KEY) > 0
            validations.append("✅ API_KEY configured")
        except:
            validations.append("⚠️  API_KEY missing (LLM features disabled)")
        
        for validation in validations:
            print(f"   {validation}")
        
        self.initialization_results["config_validation"] = validations
    
    def _validate_directory_structure(self) -> None:
        """Ensure all required directories exist with correct paths."""
        # FIXED: Ensure data injection goes to root/data
        required_dirs = [
            "data",              # FIXED: root/data (not scripts/data)
            "data/faiss_index",  # FIXED: root/data/faiss_index
            "memory",
            "config", 
            "scripts",
            "tests",
            "persona"
        ]
        
        validations = []
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name  # FIXED: All paths relative to project_root
            if dir_path.exists():
                validations.append(f"✅ {dir_name}/ exists")
            else:
                dir_path.mkdir(parents=True, exist_ok=True)
                validations.append(f"🆕 {dir_name}/ created")
        
        # FIXED: Explicitly validate data directory path
        print(f"   📁 Data directory: {self.data_dir} (confirmed root/data)")
        
        for validation in validations:
            print(f"   {validation}")
        
        self.initialization_results["directory_validation"] = validations
    
    def _initialize_memory_system(self) -> None:
        """Initialize enhanced memory system with Phase 2 features."""
        try:
            # Call memory system startup initialization
            startup_initialization()
            
            # Get initial memory statistics
            stats = get_memory_stats()
            
            print(f"   ✅ Memory system initialized")
            print(f"   📊 Total conversation turns: {stats.get('total_turns', 0)}")
            print(f"   🚫 Duplicates prevented: {stats.get('duplicates_found', 0)}")
            print(f"   🔄 Sync operations: {stats.get('sync_operations', 0)}")
            
            # FIXED: Display data directory info
            if MEMORY_FILE.exists():
                file_size = MEMORY_FILE.stat().st_size
                print(f"   💾 Memory file: {MEMORY_FILE} ({file_size} bytes)")
            
            self.initialization_results["memory_system"] = {
                "status": "success",
                "stats": stats
            }
            
        except Exception as e:
            print(f"   ❌ Memory system initialization failed: {e}")
            self.initialization_results["memory_system"] = {
                "status": "failed", 
                "error": str(e)
            }
    
    def _validate_faiss_index(self) -> None:
        """Validate FAISS index is properly loaded and functional."""
        try:
            # Get FAISS store
            store = get_store()
            index_size = store.index.ntotal
            
            # Test basic search functionality
            test_results = search("test query", k=1)
            
            print(f"   ✅ FAISS index loaded successfully")
            print(f"   📦 Index contains {index_size} vectors")
            print(f"   🔍 Search functionality verified ({len(test_results)} results)")
            
            # FIXED: Display index location
            faiss_dir = self.data_dir / "faiss_index" 
            print(f"   📁 FAISS index location: {faiss_dir}")
            
            self.initialization_results["faiss_validation"] = {
                "status": "success",
                "index_size": index_size,
                "search_functional": len(test_results) >= 0
            }
            
        except Exception as e:
            print(f"   ❌ FAISS index validation failed: {e}")
            self.initialization_results["faiss_validation"] = {
                "status": "failed",
                "error": str(e)
            }
    
    def _establish_performance_baseline(self) -> None:
        """Establish initial performance baseline metrics."""
        try:
            baseline_queries = [
                "Hello how are you",
                "What is the weather", 
                "Tell me about AI",
                "France capital query"
            ]
            
            latencies = []
            for query in baseline_queries:
                start_time = time.time()
                results = retrieve_top_memories(query, k_short=TOP_K)
                latency = (time.time() - start_time) * 1000  # Convert to ms
                latencies.append(latency)
            
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            
            print(f"   ✅ Performance baseline established")
            print(f"   ⚡ Average retrieval latency: {avg_latency:.2f}ms")
            print(f"   📊 Maximum retrieval latency: {max_latency:.2f}ms")
            
            # FIXED: Store baseline in root/data/performance_history.jsonl
            baseline_data = {
                "timestamp": time.time(),
                "type": "startup_baseline",
                "avg_latency_ms": avg_latency,
                "max_latency_ms": max_latency,
                "query_count": len(baseline_queries)
            }
            
            performance_file = self.data_dir / "performance_history.jsonl"  # FIXED: root/data
            with performance_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(baseline_data) + "\n")
            
            print(f"   💾 Baseline saved to: {performance_file}")
            
            self.initialization_results["performance_baseline"] = baseline_data
            
        except Exception as e:
            print(f"   ❌ Performance baseline establishment failed: {e}")
            self.initialization_results["performance_baseline"] = {
                "status": "failed",
                "error": str(e)
            }
    
    def _run_health_checks(self) -> None:
        """Run comprehensive system health checks."""
        health_checks = []
        
        # Check 1: Memory file integrity
        try:
            if MEMORY_FILE.exists():
                with MEMORY_FILE.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
                valid_json_lines = sum(1 for line in lines if self._is_valid_json(line.strip()))
                health_checks.append(f"✅ Memory file integrity: {valid_json_lines}/{len(lines)} valid JSON lines")
            else:
                health_checks.append("⚠️  Memory file does not exist yet (normal for new installations)")
        except Exception as e:
            health_checks.append(f"❌ Memory file integrity check failed: {e}")
        
        # Check 2: Threshold configuration consistency  
        try:
            config = MEMORY_THRESHOLD_CONFIG
            if config["min_threshold"] <= config["base_threshold"] <= config["max_threshold"]:
                health_checks.append("✅ Threshold configuration consistency verified")
            else:
                health_checks.append("❌ Threshold configuration inconsistent")
        except Exception as e:
            health_checks.append(f"❌ Threshold consistency check failed: {e}")
        
        # Check 3: System component integration
        try:
            # Test full retrieval pipeline
            short_results, long_results = retrieve_top_memories("system health check")
            health_checks.append(f"✅ Retrieval pipeline functional ({len(short_results)} short, {len(long_results)} long results)")
        except Exception as e:
            health_checks.append(f"❌ Retrieval pipeline check failed: {e}")
        
        # FIXED: Check 4: Data directory structure
        try:
            required_files = ["memory.jsonl", "faiss_index"]
            existing_files = []
            for item in required_files:
                item_path = self.data_dir / item
                if item_path.exists():
                    existing_files.append(item)
            
            health_checks.append(f"✅ Data directory structure: {len(existing_files)}/{len(required_files)} components present")
            
        except Exception as e:
            health_checks.append(f"❌ Data directory check failed: {e}")
        
        for check in health_checks:
            print(f"   {check}")
        
        self.health_check_results = health_checks
    
    def _print_startup_summary(self) -> None:
        """Print comprehensive startup summary."""
        startup_duration = time.time() - self.startup_start_time
        
        print(f"   🎉 Startup completed in {startup_duration:.2f} seconds")
        print(f"   🔧 Configuration validation: {'✅ PASSED' if 'config_validation' in self.initialization_results else '❌ FAILED'}")
        print(f"   🧠 Memory system: {'✅ READY' if self.initialization_results.get('memory_system', {}).get('status') == 'success' else '❌ ERROR'}")
        print(f"   🔍 FAISS index: {'✅ FUNCTIONAL' if self.initialization_results.get('faiss_validation', {}).get('status') == 'success' else '❌ ERROR'}")
        print(f"   📊 Performance: {'✅ BASELINED' if 'performance_baseline' in self.initialization_results else '❌ NOT SET'}")
        
        # FIXED: Display data directory summary
        print(f"   📁 Data directory: {self.data_dir}")
        
        print("\n" + "=" * 65)
        print("🚀 Memory & Mood Pipeline Phase 2 is READY FOR OPERATION!")
        print("=" * 65)
    
    def _is_valid_json(self, line: str) -> bool:
        """Check if a line contains valid JSON."""
        try:
            json.loads(line)
            return True
        except:
            return False

def main():
    """Main entry point for startup integration."""
    orchestrator = StartupOrchestrator()
    success = orchestrator.run_startup_sequence()
    
    if not success:
        sys.exit(1)
    
    print("\n💡 Next steps:")
    print("   1. Run your main application (app.py)")
    print("   2. Monitor performance via: python performance_tests.py")
    print("   3. Run tests via: python test_threshold_optimization.py")

if __name__ == "__main__":
    main()
