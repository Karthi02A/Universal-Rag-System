import chromadb
import uuid
import numpy as np
import time

_client = None

def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path="./chroma_db"
        )
    return _client

def get_collection(collection_name):
    client = get_client()
    return client.get_or_create_collection(
        name=collection_name
    )

def store_chunks(
    collection_name,
    chunks,
    embeddings,
    metadata_list
):
    collection = get_collection(collection_name)
    ids = [
        str(uuid.uuid4())
        for _ in range(len(chunks))
    ]

    # Convert embeddings to list safely
    if hasattr(embeddings, "tolist"):
        embeddings_list = embeddings.tolist()
    elif isinstance(embeddings, np.ndarray):
        embeddings_list = embeddings.tolist()
    else:
        embeddings_list = [list(e) for e in embeddings]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings_list,
        metadatas=metadata_list
    )

    return True

def delete_collection(collection_name):
    try:
        client = get_client()
        client.delete_collection(name=collection_name)
        return True
    except Exception:
        return False

def get_collection_size(collection_name):
    try:
        collection = get_collection(collection_name)
        return collection.count()
    except Exception:
        return 0

def get_db_stats(collection_name):
    try:
        client = get_client()
        collection = client.get_collection(name=collection_name)
        count = collection.count()
        return {
            "chunks": count,
            "pdfs": 0,
            "websites": 0
        }
    except Exception as e:
        return {"chunks": 0, "pdfs": 0, "websites": 0}

def cleanup_expired_collections(max_age_seconds=7200):
    """
    Scans all collection names starting with 'rag_' and deletes if creation timestamp
    extracted from the name is older than max_age_seconds.
    """
    try:
        current_time = int(time.time())
        client = get_client()
        collections = client.list_collections()
        for coll in collections:
            name = coll.name
            if name.startswith("rag_"):
                try:
                    parts = name.split("_")
                    if len(parts) >= 3:
                        creation_ts = int(parts[1])
                        if current_time - creation_ts > max_age_seconds:
                            client.delete_collection(name=name)
                except (ValueError, IndexError):
                    pass
    except Exception as e:
        print(f"Error during expired collection cleanup: {e}")

