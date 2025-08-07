# scripts/show_faiss_contents.py

from vectorstore import get_store
import argparse
import textwrap

def main():
    parser = argparse.ArgumentParser(
        description="📦 Inspect current FAISS memory contents."
    )
    parser.add_argument(
        "-k", "--top", type=int, default=10,
        help="Number of top entries to print (default: 10)"
    )
    args = parser.parse_args()

    store = get_store()
    index_size = store.index.ntotal

    print(f"\n🧠 FAISS Index Size: {index_size}")
    print("=" * 50)

    if index_size == 0:
        print("⚠️  FAISS index is empty.")
        return

    ids = list(store.index_to_docstore_id.items())[: args.top]
    if not ids:
        print("⚠️  No documents found in FAISS index.")
        return

    for i, doc_id in ids:
        doc = store.docstore._dict.get(doc_id)
        if doc is None:
            print(f"[{i}] ⚠️ Missing doc for ID {doc_id}")
            continue
        snippet = textwrap.shorten(doc.page_content.replace("\n", " "), width=120)
        print(f"[{i}] {snippet}")

if __name__ == "__main__":
    main()
