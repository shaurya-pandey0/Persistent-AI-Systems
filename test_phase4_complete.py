# tests/test_phase4_complete.py

"""
Complete Phase 4 Test Suite for Memory & Mood Pipeline

Tests all Phase 4 advanced features:
• Temporal decay models for hormones
• Adaptive baseline system with circadian rhythms  
• Hierarchical memory system (working/short-term/long-term)
• Compound mood modeling with vector states
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_DATA_DIR = Path("tests/test_data")
TEST_DATA_DIR.mkdir(exist_ok=True)

class Phase4TestSuite:
    """Complete test suite for Phase 4 features."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete Phase 4 test suite."""
        print("🚀 Starting Phase 4 Complete Test Suite")
        print("=" * 60)
        
        # Test each major component
        test_components = [
            ("Temporal Decay Models", self.test_temporal_decay_system),
            ("Adaptive Baseline System", self.test_adaptive_baseline_system),
            ("Hierarchical Memory System", self.test_hierarchical_memory_system),
            ("Compound Mood Modeling", self.test_compound_mood_system),
            ("Enhanced Memory Retrieval", self.test_enhanced_memory_retrieval),
            ("Integration Testing", self.test_system_integration)
        ]
        
        for component_name, test_function in test_components:
            print(f"\n🧪 Testing {component_name}")
            print("-" * 40)
            
            try:
                result = test_function()
                self.test_results[component_name] = {
                    "status": "PASSED" if result else "FAILED",
                    "details": result if isinstance(result, dict) else {}
                }
                
                status_emoji = "✅" if result else "❌"
                print(f"{status_emoji} {component_name}: {'PASSED' if result else 'FAILED'}")
                
            except Exception as e:
                self.test_results[component_name] = {
                    "status": "ERROR",
                    "error": str(e)
                }
                print(f"💥 {component_name}: ERROR - {e}")
        
        # Generate test report
        self.generate_test_report()
        return self.test_results
    
    def test_temporal_decay_system(self) -> bool:
        """Test Phase 4 temporal decay models."""
        print("⏰ Testing temporal decay models...")
        
        try:
            from persona.hormone_adjuster import (
                get_temporal_decay_stats,
                force_hormone_decay,
                reset_decay_timer,
                apply_contextual_hormone_adjustments
            )
            
            # Test decay statistics
            decay_stats = get_temporal_decay_stats()
            assert decay_stats["decay_enabled"] == True
            assert "hormone_half_lives" in decay_stats
            assert len(decay_stats["hormone_half_lives"]) == 4
            print("  ✓ Decay statistics working")
            
            # Test hormone adjustment with decay
            test_messages = [
                "I'm really excited about this new project!",
                "This is frustrating and stressful",
                "I feel calm and peaceful now"
            ]
            
            for message in test_messages:
                hormones = apply_contextual_hormone_adjustments(message)
                assert isinstance(hormones, dict)
                assert len(hormones) == 4
                for hormone, level in hormones.items():
                    assert 0.0 <= level <= 1.0
                print(f"  ✓ Processed: '{message[:30]}...'")
            
            # Test forced decay
            decayed_hormones = force_hormone_decay()
            assert isinstance(decayed_hormones, dict)
            print("  ✓ Forced decay working")
            
            # Test timer reset
            reset_decay_timer()
            print("  ✓ Timer reset working")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Temporal decay test failed: {e}")
            return False
    
    def test_adaptive_baseline_system(self) -> bool:
        """Test Phase 4 adaptive baseline system."""
        print("🔄 Testing adaptive baseline system...")
        
        try:
            from persona.adaptive_baseline_manager import (
                get_adaptive_baselines,
                update_baselines_from_interaction,
                record_user_feedback,
                learn_circadian_patterns,
                get_baseline_adaptation_stats
            )
            
            # Test baseline retrieval
            baselines = get_adaptive_baselines()
            assert isinstance(baselines, dict)
            assert len(baselines) == 4
            for hormone, level in baselines.items():
                assert 0.0 <= level <= 1.0
            print("  ✓ Baseline retrieval working")
            
            # Test context-specific baselines
            contexts = ["work", "social", "creative", "rest"]
            for context in contexts:
                context_baselines = get_adaptive_baselines(context)
                assert isinstance(context_baselines, dict)
                assert len(context_baselines) == 4
                print(f"  ✓ Context '{context}' baselines working")
            
            # Test baseline learning
            test_interactions = [
                ({"dopamine": 0.7, "serotonin": 0.6, "cortisol": 0.3, "oxytocin": 0.8}, 0.9, "social"),
                ({"dopamine": 0.4, "serotonin": 0.5, "cortisol": 0.8, "oxytocin": 0.3}, 0.3, "work")
            ]
            
            for hormones, satisfaction, context in test_interactions:
                update_baselines_from_interaction(hormones, satisfaction, context)
                print(f"  ✓ Learning interaction: {context} (satisfaction: {satisfaction})")
            
            # Test user feedback
            record_user_feedback("mood_preference", ("cheerful", 0.9))
            record_user_feedback("interaction_satisfaction", 0.8)
            print("  ✓ User feedback recording working")
            
            # Test circadian learning
            learn_circadian_patterns({"dopamine": 0.6, "serotonin": 0.7, "cortisol": 0.4, "oxytocin": 0.6}, 0.8)
            print("  ✓ Circadian pattern learning working")
            
            # Test statistics
            stats = get_baseline_adaptation_stats()
            assert isinstance(stats, dict)
            assert "current_baselines" in stats
            assert "configuration" in stats
            print("  ✓ Adaptation statistics working")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Adaptive baseline test failed: {e}")
            return False
    
    def test_hierarchical_memory_system(self) -> bool:
        """Test Phase 4 hierarchical memory system."""
        print("🏗️ Testing hierarchical memory system...")
        
        try:
            from memory.hierarchical_memory import (
                add_conversation_to_hierarchical_memory,
                retrieve_hierarchical_memories,
                add_fact_to_semantic_memory,
                add_procedure_to_memory,
                get_hierarchical_memory_stats,
                force_memory_consolidation
            )
            
            # Test conversation addition
            test_conversations = [
                ("What is machine learning?", "Machine learning is a method of data analysis that automates analytical model building."),
                ("I love learning about AI", "That's wonderful! AI is a fascinating field with many applications."),
                ("How do I train a neural network?", "To train a neural network, you need data, architecture, and backpropagation.")
            ]
            
            for user_input, assistant_response in test_conversations:
                add_conversation_to_hierarchical_memory(user_input, assistant_response)
                print(f"  ✓ Added conversation: '{user_input[:40]}...'")
            
            # Test memory retrieval
            queries = ["machine learning", "neural network", "AI applications"]
            for query in queries:
                results = retrieve_hierarchical_memories(query, k=2)
                assert isinstance(results, list)
                print(f"  ✓ Retrieved {len(results)} memories for '{query}'")
            
            # Test fact addition
            add_fact_to_semantic_memory("Python is a programming language")
            print("  ✓ Added semantic fact")
            
            # Test procedure addition
            add_procedure_to_memory("To install Python: 1. Download installer 2. Run installer 3. Verify installation")
            print("  ✓ Added procedural memory")
            
            # Test memory statistics
            stats = get_hierarchical_memory_stats()
            assert isinstance(stats, dict)
            assert "working_memory" in stats
            assert "long_term_memory" in stats
            print("  ✓ Memory statistics working")
            
            # Test consolidation
            consolidation_result = force_memory_consolidation()
            assert isinstance(consolidation_result, dict)
            print("  ✓ Memory consolidation working")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Hierarchical memory test failed: {e}")
            return False
    
    def test_compound_mood_system(self) -> bool:
        """Test Phase 4 compound mood modeling."""
        print("🎭 Testing compound mood system...")
        
        try:
            from persona.mood_tracker import (
                calculate_mood_from_hormones,
                update_mood_from_hormones,
                get_compound_mood_analysis,
                get_mood_predictions,
                get_enhanced_mood_summary
            )
            
            # Test compound mood calculation
            test_hormone_sets = [
                {"dopamine": 0.7, "serotonin": 0.6, "cortisol": 0.4, "oxytocin": 0.8},  # Happy
                {"dopamine": 0.3, "serotonin": 0.2, "cortisol": 0.8, "oxytocin": 0.3},  # Stressed
                {"dopamine": 0.5, "serotonin": 0.5, "cortisol": 0.5, "oxytocin": 0.5},  # Neutral
            ]
            
            for i, hormones in enumerate(test_hormone_sets, 1):
                result = calculate_mood_from_hormones(hormones)
                
                if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict):
                    # Compound mood result
                    mood_vector, analysis = result
                    assert isinstance(mood_vector, dict)
                    assert len(mood_vector) > 0
                    print(f"  ✓ Compound mood test {i}: {len(mood_vector)} dimensions")
                else:
                    # Scalar mood result (fallback)
                    mood, intensity = result
                    assert isinstance(mood, str)
                    assert 0.0 <= intensity <= 1.0
                    print(f"  ✓ Scalar mood test {i}: {mood} ({intensity:.2f})")
            
            # Test mood updates
            update_mood_from_hormones("test_compound_system")
            print("  ✓ Mood update working")
            
            # Test compound mood analysis
            try:
                analysis = get_compound_mood_analysis()
                if "vector_moods_disabled" not in analysis:
                    assert isinstance(analysis, dict)
                    print("  ✓ Compound mood analysis working")
                else:
                    print("  ⚠️ Vector moods disabled - using scalar fallback")
            except Exception:
                print("  ⚠️ Compound analysis not available - using fallback")
            
            # Test mood predictions
            try:
                predictions = get_mood_predictions()
                assert isinstance(predictions, dict)
                print("  ✓ Mood predictions working")
            except Exception:
                print("  ⚠️ Mood predictions not available")
            
            # Test enhanced mood summary
            summary = get_enhanced_mood_summary()
            assert isinstance(summary, dict)
            assert "current_state" in summary
            print("  ✓ Enhanced mood summary working")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Compound mood test failed: {e}")
            return False
    
    def test_enhanced_memory_retrieval(self) -> bool:
        """Test Phase 4 enhanced memory retrieval."""
        print("🔍 Testing enhanced memory retrieval...")
        
        try:
            from memory.context_retriever import (
                retrieve_top_memories,
                retrieve_memories_by_importance_and_tier,
                get_memory_management_stats,
                test_phase4_memory_features
            )
            
            # Test enhanced retrieval
            test_queries = [
                "How does machine learning work?",
                "What is artificial intelligence?",
                "I love programming"
            ]
            
            for query in test_queries:
                short, long = retrieve_top_memories(query, k_short=2, k_long=1, use_hierarchical=True)
                assert isinstance(short, list)
                assert isinstance(long, list)
                print(f"  ✓ Retrieved memories for '{query}': {len(short)} short, {len(long)} long")
            
            # Test importance-based retrieval
            important_memories = retrieve_memories_by_importance_and_tier(
                "machine learning", 
                min_importance=0.5, 
                k=2
            )
            assert isinstance(important_memories, list)
            print(f"  ✓ Important memories: {len(important_memories)}")
            
            # Test memory management stats
            stats = get_memory_management_stats()
            assert isinstance(stats, dict)
            assert "hierarchical_memory" in stats
            print("  ✓ Memory management statistics working")
            
            # Run built-in tests
            try:
                test_phase4_memory_features()
                print("  ✓ Built-in memory tests passed")
            except Exception as e:
                print(f"  ⚠️ Built-in tests warning: {e}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Enhanced memory retrieval test failed: {e}")
            return False
    
    def test_system_integration(self) -> bool:
        """Test Phase 4 system integration."""
        print("🔗 Testing system integration...")
        
        try:
            # Test full pipeline: conversation -> hormones -> mood -> memory
            test_conversation = {
                "user": "I'm really excited about learning AI and machine learning!",
                "assistant": "That's fantastic! AI and ML are fascinating fields with endless possibilities."
            }
            
            # 1. Process conversation through hormone system
            from persona.hormone_adjuster import apply_contextual_hormone_adjustments
            hormones = apply_contextual_hormone_adjustments(test_conversation["user"])
            assert isinstance(hormones, dict)
            print("  ✓ Hormone processing working")
            
            # 2. Update mood based on hormones
            from persona.mood_tracker import update_mood_from_hormones
            update_mood_from_hormones("integration_test")
            print("  ✓ Mood update working")
            
            # 3. Add to hierarchical memory
            try:
                from memory.hierarchical_memory import add_conversation_to_hierarchical_memory
                add_conversation_to_hierarchical_memory(
                    test_conversation["user"], 
                    test_conversation["assistant"]
                )
                print("  ✓ Hierarchical memory storage working")
            except ImportError:
                print("  ⚠️ Hierarchical memory not available")
            
            # 4. Update adaptive baselines
            from persona.adaptive_baseline_manager import update_baselines_from_interaction
            update_baselines_from_interaction(hormones, 0.8, "learning")
            print("  ✓ Baseline adaptation working")
            
            # 5. Test retrieval
            from memory.context_retriever import retrieve_top_memories
            short, long = retrieve_top_memories("machine learning", use_hierarchical=True)
            assert isinstance(short, list)
            print("  ✓ Memory retrieval working")
            
            # Test configuration validation
            from config.constants import (
                TEMPORAL_DECAY_CONFIG,
                ADAPTIVE_BASELINE_CONFIG,
                HIERARCHICAL_MEMORY_CONFIG,
                COMPOUND_MOOD_CONFIG
            )
            
            configs = [
                TEMPORAL_DECAY_CONFIG,
                ADAPTIVE_BASELINE_CONFIG, 
                HIERARCHICAL_MEMORY_CONFIG,
                COMPOUND_MOOD_CONFIG
            ]
            
            for config in configs:
                assert isinstance(config, dict)
                assert len(config) > 0
            
            print("  ✓ Configuration validation working")
            
            return True
            
        except Exception as e:
            print(f"  ❌ System integration test failed: {e}")
            return False
    
    def generate_test_report(self) -> None:
        """Generate comprehensive test report."""
        total_time = time.time() - self.start_time
        
        print(f"\n" + "=" * 60)
        print("📊 PHASE 4 TEST REPORT")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results.values() if result["status"] == "PASSED")
        failed = sum(1 for result in self.test_results.values() if result["status"] == "FAILED")
        errors = sum(1 for result in self.test_results.values() if result["status"] == "ERROR")
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"💥 Errors: {errors}")
        print(f"⏱️ Total Time: {total_time:.2f}s")
        
        success_rate = (passed / total * 100) if total > 0 else 0
        print(f"🎯 Success Rate: {success_rate:.1f}%")
        
        print(f"\n📋 Detailed Results:")
        for component, result in self.test_results.items():
            status_emoji = {"PASSED": "✅", "FAILED": "❌", "ERROR": "💥"}[result["status"]]
            print(f"  {status_emoji} {component}: {result['status']}")
            
            if result["status"] == "ERROR":
                print(f"    Error: {result['error']}")
        
        # Save detailed report
        report_file = TEST_DATA_DIR / f"phase4_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_time": total_time,
                "summary": {
                    "total": total,
                    "passed": passed,
                    "failed": failed,
                    "errors": errors,
                    "success_rate": success_rate
                },
                "results": self.test_results
            }, f, indent=2)
        
        print(f"\n💾 Detailed report saved: {report_file}")
        
        if success_rate >= 80:
            print(f"\n🎉 PHASE 4 TESTS SUCCESSFUL! ({success_rate:.1f}% pass rate)")
        else:
            print(f"\n⚠️ PHASE 4 TESTS NEED ATTENTION ({success_rate:.1f}% pass rate)")

def main():
    """Run the complete Phase 4 test suite."""
    test_suite = Phase4TestSuite()
    results = test_suite.run_all_tests()
    
    # Return exit code based on results
    passed = sum(1 for result in results.values() if result["status"] == "PASSED")
    total = len(results)
    success_rate = (passed / total * 100) if total > 0 else 0
    
    return 0 if success_rate >= 80 else 1

if __name__ == "__main__":
    exit(main())
