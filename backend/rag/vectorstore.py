import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

_client = None
_collection = None

def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=os.environ["CHROMA_PATH"])
        _collection = _client.get_collection(
            name=os.environ["COLLECTION_NAME"]
        )
    return _collection

def search(query_embedding, top_k=4):
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    documents = []
    for i in range(len(results["documents"][0])):
        distance = results["distances"][0][i]
        similarity = 1 - distance
        documents.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": round(similarity, 3),
            "distance": round(distance, 3)
        })
    return documents

def get_collection_stats():
    collection = get_collection()
    return {
        "total_documents": collection.count(),
        "collection_name": os.environ["COLLECTION_NAME"]
    }