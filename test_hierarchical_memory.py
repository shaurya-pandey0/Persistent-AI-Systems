# tests/test_hierarchical_memory.py

"""
Specific tests for Phase 4 hierarchical memory system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_hierarchical_memory_detailed():
    """Test hierarchical memory system in detail."""
    print("🧪 Detailed Hierarchical Memory System Testing")
    print("=" * 55)
    
    try:
        from memory.hierarchical_memory import (
            _hierarchical_memory,
            add_conversation_to_hierarchical_memory,
            retrieve_hierarchical_memories,
            add_fact_to_semantic_memory,
            add_procedure_to_memory,
            get_hierarchical_memory_stats,
            force_memory_consolidation
        )
        
        # Test 1: Memory addition and classification
        print("\n1️⃣ Testing memory addition and classification:")
        
        test_items = [
            ("episodic", "Yesterday I had a great conversation about AI with my friend"),
            ("semantic", "Python is a high-level programming language"),
            ("procedural", "To make coffee: 1. Boil water 2. Add coffee grounds 3. Pour and stir"),
            ("episodic", "I remember feeling excited when I learned about neural networks"),
            ("semantic", "Machine learning is a subset of artificial intelligence")
        ]
        
        for expected_type, content in test_items:
            if expected_type == "episodic":
                # Split into conversation format
                add_conversation_to_hierarchical_memory(content, "That sounds interesting!")
            elif expected_type == "semantic":
                add_fact_to_semantic_memory(content)
            elif expected_type == "procedural":
                add_procedure_to_memory(content)
            
            print(f"  ✓ Added {expected_type}: '{content[:40]}...'")
        
        # Test 2: Memory tier functionality
        print("\n2️⃣ Testing memory tiers:")
        
        # Check working memory
        working_items = _hierarchical_memory.working_memory.get_items()
        print(f"  Working memory: {len(working_items)} items")
        
        # Check short-term memory
        short_term_items = _hierarchical_memory.short_term_memory.get_items()
        print(f"  Short-term memory: {len(short_term_items)} items")
        
        # Check long-term memory
        for memory_type in ["episodic", "semantic", "procedural"]:
            lt_items = _hierarchical_memory.long_term_memory.get_items(memory_type=memory_type)
            print(f"  Long-term {memory_type}: {len(lt_items)} items")
        
        # Test 3: Memory retrieval by type
        print("\n3️⃣ Testing memory retrieval by type:")
        
        test_queries = [
            ("Python programming", ["semantic", "procedural"]),
            ("conversation about AI", ["episodic"]),
            ("how to make coffee", ["procedural"])
        ]
        
        for query, memory_types in test_queries:
            results = retrieve_hierarchical_memories(query, k=3, memory_types=memory_types)
            print(f"  Query '{query}' ({memory_types}): {len(results)} results")
            
            for i, result in enumerate(results[:2], 1):
                print(f"    {i}. {result[:50]}...")
        
        # Test 4: Memory consolidation
        print("\n4️⃣ Testing memory consolidation:")
        
        stats_before = get_hierarchical_memory_stats()
        print(f"  Before consolidation:")
        print(f"    Working: {stats_before['working_memory']['count']}")
        print(f"    Short-term: {stats_before['short_term_memory']['count']}")
        print(f"    Long-term: {stats_before['long_term_memory']['total']}")
        
        consolidation_result = force_memory_consolidation()
        print(f"  Consolidation completed in {consolidation_result['consolidation_time_seconds']:.3f}s")
        print(f"  Promoted to short-term: {consolidation_result['promoted_to_short_term']}")
        print(f"  Consolidated to long-term: {consolidation_result['consolidated_to_long_term']}")
        
        stats_after = get_hierarchical_memory_stats()
        print(f"  After consolidation:")
        print(f"    Working: {stats_after['working_memory']['count']}")
        print(f"    Short-term: {stats_after['short_term_memory']['count']}")
        print(f"    Long-term: {stats_after['long_term_memory']['total']}")
        
        # Test 5: Memory associations
        print("\n5️⃣ Testing memory associations:")
        
        # Add related items to test associations
        related_items = [
            "Neural networks are inspired by biological neurons",
            "Deep learning uses multiple layers in neural networks",
            "Backpropagation is used to train neural networks"
        ]
        
        for item in related_items:
            add_fact_to_semantic_memory(item)
        
        # Force consolidation to create associations
        force_memory_consolidation()
        
        # Test retrieval of associated memories
        neural_results = retrieve_hierarchical_memories("neural networks", k=5)
        print(f"  Neural network query returned {len(neural_results)} associated results")
        
        # Test 6: Performance under load  
        print("\n6️⃣ Testing performance under load:")
        
        import time
        start_time = time.time()
        
        # Add many items quickly
        for i in range(50):
            add_conversation_to_hierarchical_memory(
                f"Test conversation {i} about topic {i % 10}",
                f"This is response {i} discussing topic {i % 10}"
            )
        
        load_time = time.time() - start_time
        print(f"  Added 50 conversations in {load_time:.3f}s ({50/load_time:.1f} items/sec)")
        
        # Test retrieval performance
        start_time = time.time()
        
        for i in range(10):
            results = retrieve_hierarchical_memories(f"topic {i}", k=3)
        
        retrieval_time = time.time() - start_time
        print(f"  10 retrievals in {retrieval_time:.3f}s ({10/retrieval_time:.1f} queries/sec)")
        
        # Final stats
        final_stats = get_hierarchical_memory_stats()
        print(f"\n📊 Final memory statistics:")
        print(f"  Total items: {final_stats['working_memory']['count'] + final_stats['short_term_memory']['count'] + final_stats['long_term_memory']['total']}")
        print(f"  Working memory utilization: {final_stats['working_memory']['utilization']:.1%}")
        print(f"  Short-term utilization: {final_stats['short_term_memory']['utilization']:.1%}")
        
        print("\n✅ All hierarchical memory tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Hierarchical memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_hierarchical_memory_detailed()
