"""Постоянное векторное хранилище на ChromaDB."""

import chromadb
from config import CHROMA_PATH, COLLECTION_NAME

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def get_collection():
    return _collection


def reset_collection():
    global _collection
    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def add_chunks(ids, documents, embeddings, metadatas):
    _collection.add(
        ids=ids, documents=documents,
        embeddings=embeddings, metadatas=metadatas,
    )


def query_chunks(query_embedding, n_results=5):
    return _collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )


def count():
    return _collection.count()


def get_all_documents():
    """Возвращает все чанки — для построения BM25-индекса."""
    result = _collection.get(include=["documents", "metadatas"])
    return result["ids"], result["documents"], result["metadatas"]
