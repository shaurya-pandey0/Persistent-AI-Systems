# memory/tools.py
from vectorstore import add_texts, get_store
from memory.turn_memory import load_memory

def rehydrate_from_jsonl(force: bool = False):
    """
    Reloads all messages from memory.jsonl and updates FAISS.
    If force=True, completely clears the FAISS index, docstore and id mapping
    before re-adding everything. This removes the 'No conversations yet' dummy.
    """
    store = get_store()

    if force:
        print("♻️ Clearing FAISS index before reload (force=True)...")
        # Wipe vector index
        store.index.reset()
        # Wipe docstore and id mapping (LangChain FAISS internals)
        try:
            store.docstore._dict.clear()
        except Exception:
            pass
        store.index_to_docstore_id = {}
        store.save_local("data/faiss_index")

    # Rebuild from JSONL
    texts = []
    for turn in load_memory():
        user, assistant = turn.get("user", ""), turn.get("assistant", "")
        if user or assistant:
            texts.append(f"User: {user}\nAssistant: {assistant}")

    print(f"📦 Re-adding {len(texts)} texts to FAISS...")
    if texts:
        add_texts(texts)
