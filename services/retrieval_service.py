from services.vector_service import get_collection
from services.embedding_service import generate_embeddings
import re
import numpy as np

import streamlit as st

@st.cache_resource
def get_reranker():
    from sentence_transformers import CrossEncoder
    import os
    
    local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "local_reranker")
    if os.path.exists(local_path):
        model_name_or_path = local_path
    else:
        model_name_or_path = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
    return CrossEncoder(model_name_or_path)

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def resolve_parents_and_deduplicate(candidates, top_k):
    seen_parents = set()
    unique_candidates = []
    for cand in candidates:
        meta = cand.get("metadata", {}) or {}
        parent_text = meta.get("parent_text", cand["text"])
        if parent_text not in seen_parents:
            seen_parents.add(parent_text)
            cand["text"] = parent_text
            unique_candidates.append(cand)
            if len(unique_candidates) >= top_k:
                break
    return unique_candidates

@st.cache_resource
def get_bm25_index(collection_name):
    collection = get_collection(collection_name)
    all_chunks = collection.get(include=["documents", "metadatas"])
    documents = all_chunks["documents"]
    metadatas = all_chunks["metadatas"]
    ids = all_chunks["ids"]
    
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, ids, documents, metadatas

def search_chunks(collection_name, query, top_k=4, use_hybrid=True, use_reranker=False):
    collection = get_collection(collection_name)
    
    # 1. Fetch total document count to check if database is empty
    count = collection.count()
    if count == 0:
        return []

    # If hybrid search is disabled, run pure vector search
    if not use_hybrid:
        # Generate embedding for the query
        query_embeddings = generate_embeddings([query])
        query_embedding = query_embeddings[0]
        
        vector_results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(30, count)
        )
        
        candidates = []
        if vector_results and "documents" in vector_results and len(vector_results["documents"]) > 0:
            ids = vector_results["ids"][0]
            docs = vector_results["documents"][0]
            metas = vector_results["metadatas"][0]
            distances = vector_results["distances"][0] if "distances" in vector_results else [1.0]*len(ids)
            for i in range(len(ids)):
                candidates.append({
                    "id": ids[i],
                    "text": docs[i],
                    "metadata": metas[i],
                    "distance": distances[i],
                    "search_type": "vector"
                })
        
        if use_reranker:
            candidates = rerank_results(query, candidates)
            
        return resolve_parents_and_deduplicate(candidates, top_k)

    # 2. Run Hybrid Search (Vector + BM25 using RRF)
    # Fetch cached BM25 index and corpus details
    bm25, ids, documents, metadatas = get_bm25_index(collection_name)
    
    # BM25 Keyword Search
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    
    ranked_indices = np.argsort(bm25_scores)[::-1]
    
    bm25_candidates = []
    for rank_idx, idx in enumerate(ranked_indices):
        if bm25_scores[idx] <= 0:
            break
        bm25_candidates.append({
            "id": ids[idx],
            "text": documents[idx],
            "metadata": metadatas[idx],
            "bm25_score": bm25_scores[idx]
        })
        if len(bm25_candidates) >= 30:
            break

    # Vector Database Search
    query_embeddings = generate_embeddings([query])
    query_embedding = query_embeddings[0]
    
    # Retrieve top 30 candidates to merge
    vector_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(30, count)
    )
    
    vector_candidates = []
    if vector_results and "documents" in vector_results and len(vector_results["documents"]) > 0:
        v_ids = vector_results["ids"][0]
        v_docs = vector_results["documents"][0]
        v_metas = vector_results["metadatas"][0]
        v_distances = vector_results["distances"][0] if "distances" in vector_results else [1.0]*len(v_ids)
        for i in range(len(v_ids)):
            vector_candidates.append({
                "id": v_ids[i],
                "text": v_docs[i],
                "metadata": v_metas[i],
                "distance": v_distances[i]
            })

    # Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    doc_lookup = {}
    
    # Process vector candidates (higher rank is better)
    for rank, cand in enumerate(vector_candidates, start=1):
        doc_id = cand["id"]
        doc_lookup[doc_id] = cand
        doc_lookup[doc_id]["search_type"] = "vector"
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60.0 + rank))
        
    # Process BM25 candidates
    for rank, cand in enumerate(bm25_candidates, start=1):
        doc_id = cand["id"]
        if doc_id not in doc_lookup:
            doc_lookup[doc_id] = cand
            doc_lookup[doc_id]["search_type"] = "keyword"
        else:
            doc_lookup[doc_id]["search_type"] = "hybrid"
            
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60.0 + rank))

    # Sort candidates by RRF score
    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    candidates = []
    for doc_id, rrf_score in sorted_ids:
        cand = doc_lookup[doc_id]
        cand["rrf_score"] = rrf_score
        candidates.append(cand)
        
    # If reranking, we take more candidates (up to 15) and select the top_k
    if use_reranker:
        rerank_pool = candidates[:15]
        candidates = rerank_results(query, rerank_pool)
        
    return resolve_parents_and_deduplicate(candidates, top_k)

def rerank_results(query, candidates):
    if not candidates:
        return []
    try:
        reranker = get_reranker()
        pairs = [[query, cand["text"]] for cand in candidates]
        scores = reranker.predict(pairs)
        
        for idx, score in enumerate(scores):
            candidates[idx]["rerank_score"] = float(score)
            
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates
    except Exception as e:
        print(f"Reranking error: {e}. Returning initial search order.")
        return candidates