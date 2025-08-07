# tests/test_threshold_optimization.py
"""
Regression test for Phase-1 threshold calibration (CORRECTED).

FIXES:
• Corrected threshold logic for L2 distance (score <= threshold)
• Updated test expectations based on actual FAISS scoring behavior

Run:  python tests/test_threshold_optimization.py
"""

from memory.context_retriever import retrieve_top_memories
from vectorstore import add_texts, search


_FAKE_TURNS = [
    "hi",
    "hello!",
    "What is the capital of France? Assistant: The capital of France is Paris.",
    "Tell me a joke. Assistant: Why don't scientists trust atoms? Because they make up everything!",
]

def _seed_store_once():
    """Add fake conversation turns to the FAISS store."""
    add_texts(_FAKE_TURNS)

def test_threshold_filters_noise():
    """Test that threshold filtering works correctly with L2 distance scores."""
    print("🧪 Testing threshold filtering...")
    
    _seed_store_once()
    
    # Search directly to see actual scores
    raw_results = search("France capital", k=4)
    print("\n📊 Raw search results:")
    for text, score in raw_results:
        print(f"  Score: {score:.4f} | {text[:50]}...")
    
    # Test the retrieve function
    short, _ = retrieve_top_memories("France capital")
    print(f"\n🔍 Filtered results: {len(short)} matches")
    for result in short:
        print(f"  ✓ {result[:50]}...")
    
    # Assertions
    assert any("Paris" in s for s in short), f"Relevant fact vanished! Got: {short}"
    assert not any(s.lower().strip() in ["hi", "hello!"] for s in short), f"Noise still present! Got: {short}"
    
    print("✅ Threshold filtering test passed!")

def test_score_distribution():
    """Verify our understanding of the scoring mechanism."""
    print("\n🔬 Testing score distribution...")
    
    _seed_store_once()
    results = search("France capital", k=4)
    
    print("Score analysis:")
    for i, (text, score) in enumerate(results):
        relevance = "RELEVANT" if "Paris" in text else "NOISE" if text.lower().strip() in ["hi", "hello!"] else "OTHER"
        print(f"  {i+1}. {score:.4f} ({relevance}): {text[:40]}...")
    
    # Verify that relevant results have lower scores than noise
    relevant_scores = [score for text, score in results if "Paris" in text]
    noise_scores = [score for text, score in results if text.lower().strip() in ["hi", "hello!"]]
    
    if relevant_scores and noise_scores:
        avg_relevant = sum(relevant_scores) / len(relevant_scores)
        avg_noise = sum(noise_scores) / len(noise_scores)
        print(f"  📈 Avg relevant score: {avg_relevant:.4f}")
        print(f"  📉 Avg noise score: {avg_noise:.4f}")
        
        assert avg_relevant < avg_noise, "L2 distance assumption violated: relevant should have lower scores"
        print("✅ L2 distance scoring confirmed!")


if __name__ == "__main__":
    test_threshold_filters_noise()
    test_score_distribution()
    print("\n🎉 All Phase-1 threshold optimization tests passed!")