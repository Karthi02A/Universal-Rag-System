import streamlit as st

@st.cache_resource
def get_embedding_model():
    # Lazy import to avoid startup lag
    from sentence_transformers import SentenceTransformer
    import torch
    import os
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Path to local model committed in the repository
    local_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "local_model")
    if os.path.exists(local_model_path):
        model_name_or_path = local_model_path
    else:
        # Fallback to downloading if local model is not found
        model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
        
    return SentenceTransformer(model_name_or_path, device=device)

def generate_embeddings(chunks, progress_callback=None):
    model = get_embedding_model()
    if not progress_callback:
        embeddings = model.encode(
            chunks,
            show_progress_bar=True
        )
        return embeddings

    import numpy as np
    all_embeddings = []
    total = len(chunks)
    batch_size = 32

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        batch_emb = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(batch_emb)
        completed = min(i + batch_size, total)
        progress_callback(completed, total)

    return np.vstack(all_embeddings) if all_embeddings else np.array([])