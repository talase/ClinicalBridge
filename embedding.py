# build the ChromaDB vector store from the patient records

import os
import chromadb
from sentence_transformers import SentenceTransformer

from chunking import build_documents

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
COLLECTION_NAME = "clinical_records"

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts):
    vectors = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def get_client():
    return chromadb.PersistentClient(path=VECTOR_STORE_DIR)


def build_index():
    docs = build_documents()
    print("Built", len(docs), "chunks")

    client = get_client()

    # delete the old collection first so we don't store the same records twice
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    metadatas = [{"patient_id": d["patient_id"], "source": d["source"]} for d in docs]
    embeddings = embed_texts(texts)

    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    print("Stored", collection.count(), "vectors")
    return collection


if __name__ == "__main__":
    col = build_index()

    # quick test query
    q = embed_texts(["low oxygen breathing problem"])
    res = col.query(query_embeddings=q, n_results=3)
    print()
    print("Test query results:")
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        print(" ", meta["source"], meta["patient_id"], round(dist, 3))
