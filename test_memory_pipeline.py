# test_memory_pipeline.py

from memory.turn_memory import dump_turn
from memory.vectorstore import search
from memory.context_retriever import retrieve_top_memories
from memory.tools import rehydrate_from_jsonl


def test_inject_fake_chat():
    print("\n[🧠] Injecting fake conversation turns...")

    turns = [
        {"user": "hi", "assistant": "Hello! How can I help you today?"},
        {"user": "What is the capital of France?", "assistant": "The capital of France is Paris."},
        {"user": "Who wrote 1984?", "assistant": "George Orwell."},
        {"user": "Tell me a joke", "assistant": "Why don’t scientists trust atoms? Because they make up everything!"},
    ]
    for turn in turns:
        dump_turn(turn)

    print("[✅] Fake chat injected.")


def test_rehydrate():
    print("\n♻️ Rehydrating FAISS index from memory.jsonl ...")
    rehydrate_from_jsonl(force=True)
    print("[✅] FAISS rehydrated.")


def test_faiss_search():
    print("\n🔍 Searching FAISS for 'France capital'...")
    results = search("France capital", k=5)
    for i, (text, score) in enumerate(results):
        oneline = text.replace("\n", " ")
        print(f"[{i+1}] Score: {score:.4f} | {oneline[:100]}...")
    assert any("Paris" in text for text, _ in results), "❌ FAISS failed to return expected memory"
    print("[✅] test_faiss_search passed")


def test_retrieve_top_memories():
    print("\n🔎 Context Retrieval Test: 'Who wrote the novel 1984?'")
    short, long = retrieve_top_memories("Who wrote the novel 1984?")
    print("Short-term Matches:")
    for s in short:
        print("•", s[:100].replace("\n", " "))
    print("Long-term Matches:")
    for l in long:
        print("•", l[:100].replace("\n", " "))

    assert any("George Orwell" in s for s in (short + long)), \
        "Context retrieval failed to find 'George Orwell'"
    print("[✅] test_retrieve_top_memories passed")


if __name__ == "__main__":
    # Fresh, deterministic flow
    test_inject_fake_chat()       # write via chatbot's real path (dump_turn)
    test_rehydrate()              # rebuild FAISS from jsonl (fully cleared)
    test_faiss_search()           # verify FAISS recall
    test_retrieve_top_memories()  # verify context_retriever pipeline

    print("\n🎉 All memory pipeline tests passed (fresh run)")
