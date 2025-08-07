"""
core/faiss_utils.py
Backward-compat façade so legacy imports keep working.
Prefer `vectorstore.get_store()` for all new code.
"""
from typing import List

from vectorstore import BGEEmbeddings, get_store


def load_faiss_index(_: List[str]):
    """Legacy shim that just returns the live singleton."""
    class _Wrapper:                   # matches the old interface
        @property
        def index(self):
            return get_store().index
        
        texts: List[str] = []         # no longer used

    return _Wrapper()
