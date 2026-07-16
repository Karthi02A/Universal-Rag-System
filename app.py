import streamlit as st
import os
import uuid
import numpy as np
import time
import gc
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from services.document_service import extract_document_pages
from services.web_service import extract_website_text
from services.chunk_service import create_parent_child_chunks
from services.embedding_service import generate_embeddings
from services.vector_service import store_chunks, delete_collection, get_db_stats, cleanup_expired_collections
from services.retrieval_service import search_chunks
import services.llm_service
try:
    from services.llm_service import generate_answer, is_conversational_query
except ImportError:
    import importlib
    importlib.reload(services.llm_service)
    from services.llm_service import generate_answer, is_conversational_query

# Background Preloader for PyTorch Models (avoids the 70s first-use block during document indexing)
@st.cache_resource
def start_model_preloading():
    import threading
    
    # Using mutable dictionary to share status safely across threads
    status = {"embedding": "loading", "reranker": "loading"}
    
    def preload_worker():
        try:
            # Load embedding model
            from services.embedding_service import get_embedding_model
            get_embedding_model()
            status["embedding"] = "ready"
        except Exception as e:
            status["embedding"] = f"error: {str(e)}"
            
        try:
            # Load reranker model
            from services.retrieval_service import get_reranker
            get_reranker()
            status["reranker"] = "ready"
        except Exception as e:
            status["reranker"] = f"error: {str(e)}"
            
    thread = threading.Thread(target=preload_worker, daemon=True)
    thread.start()
    return status

# Initialize preloading asynchronously (Streamlit runs this once per server startup lifecycle)
preload_status = start_model_preloading()

# Page Configuration
st.set_page_config(
    page_title="Conversational RAG System",
    layout="centered"
)

# Custom minimalist styling (ChatGPT inspired)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif !important;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #f1f5f9;
    text-align: center;
    margin-top: 1.5rem;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    font-size: 1rem;
    color: #94a3b8;
    text-align: center;
    margin-bottom: 2rem;
}

/* Sidebar styling overrides */
section[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid #334155;
}

.sidebar-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #f8fafc;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid #334155;
    padding-bottom: 0.5rem;
}

/* Clean status indicator */
.status-box {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    color: #e2e8f0;
    font-size: 0.9rem;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
}

/* Primary Button overrides */
.stButton>button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: background-color 0.15s ease-in-out !important;
}

.stButton>button:hover {
    background-color: #1d4ed8 !important;
    box-shadow: none !important;
    transform: none !important;
}

/* Mobile-friendly responsiveness overrides */
@media (max-width: 768px) {
    /* Reduce block-container padding to maximize screen width on mobile */
    [data-testid="stAppViewBlockContainer"], .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1.5rem !important;
        padding-bottom: 3.5rem !important;
    }
    
    .hero-title {
        font-size: 1.6rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.2rem !important;
    }
    
    .hero-subtitle {
        font-size: 0.85rem !important;
        margin-bottom: 1rem !important;
    }
    
    .status-box {
        padding: 0.6rem 0.8rem !important;
        font-size: 0.8rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Compact tabs padding for small screen width */
    .stTabs [data-baseweb="tab"] {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        font-size: 0.85rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Read Groq API Key silently from .env
api_key = os.getenv("GROQ_API_KEY", "")

# Initialize session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize collection name (rag_<timestamp>_<session_id_suffix>)
if "collection_name" not in st.session_state:
    st.session_state.collection_name = f"rag_{int(time.time())}_{st.session_state.session_id[:8]}"
    # Auto-cleanup expired collections (>2 hours old) on spin-up
    cleanup_expired_collections(max_age_seconds=7200)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_source" not in st.session_state:
    st.session_state.indexed_source = "None"
if "source_name" not in st.session_state:
    st.session_state.source_name = ""
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "db_stats" not in st.session_state:
    st.session_state.db_stats = get_db_stats(st.session_state.collection_name)
if "advanced_expanded" not in st.session_state:
    st.session_state.advanced_expanded = False

# Fetch current database statistics from session state
db_stats = st.session_state.db_stats

# ----------------------------------------------------
# SIDEBAR: CONFIGURATIONS
# ----------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>Control Panel</div>", unsafe_allow_html=True)
    
    # Show preloading status in the Control Panel
    emb_state = preload_status["embedding"]
    rerank_state = preload_status["reranker"]
    
    if emb_state == "ready" and rerank_state == "ready":
        st.success("✓ AI models loaded and active")
    elif "error" in emb_state or "error" in rerank_state:
        st.error(f"⚠️ Model load failed: Emb: {emb_state}, Reranker: {rerank_state}")
    else:
        st.info("⏳ AI models initializing in background... (Est: ~60s on first startup)")
        
    # default to llama-3.1-8b-instant for fast generations
    model_choice = st.selectbox(
        "Model Selection",
        options=[
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ],
        index=1,
        help="Select the model for response generation."
    )
    
    # Group settings inside collapsible expander to avoid sidebar clutter
    with st.expander("Advanced Settings", expanded=st.session_state.advanced_expanded):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05
        )
        
        timeout_val = st.slider(
            "Max Processing Timeout (s)",
            min_value=30,
            max_value=600,
            value=120,
            step=10,
            help="Maximum time allowed for PDF parsing before warning."
        )
        
        max_chunk_val = st.slider(
            "Max Chunk Limit",
            min_value=100,
            max_value=20000,
            value=3000,
            step=100,
            help="Maximum number of text chunks allowed per ingested document."
        )
        
        use_hybrid = st.checkbox("Use Hybrid Search", value=True)
        use_reranker = st.checkbox("Use Cross-Encoder Re-ranking", value=False)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    if st.button("Reset Session Context"):
        # Purge current session collection and data states
        delete_collection(st.session_state.collection_name)
        st.session_state.messages = []
        st.session_state.indexed_source = "None"
        st.session_state.source_name = ""
        st.session_state.chunk_count = 0
        st.session_state.db_stats = {"chunks": 0, "pdfs": 0, "websites": 0}
        st.session_state.advanced_expanded = False
        st.session_state.collection_name = f"rag_{int(time.time())}_{st.session_state.session_id[:8]}"
        gc.collect()
        st.success("Session context reset complete.")
        st.rerun()

# ----------------------------------------------------
# MAIN CONTENT PANEL
# ----------------------------------------------------

# Professional Minimal Title
st.markdown("<div class='hero-title'>Conversational RAG System</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Ask questions about your uploaded documents or web content.</div>", unsafe_allow_html=True)

if not api_key or not api_key.strip():
    st.error("GROQ_API_KEY is missing from environment. Please verify your .env configuration.")

# COLLAPSIBLE INGESTION SECTION (Expands by default if database is empty)
is_db_empty = (db_stats['chunks'] == 0)
with st.expander("Ingest Document or Web Page", expanded=is_db_empty):
    tab_pdf, tab_web = st.tabs(["Document File", "Website URL"])
    
    # Document Tab
    with tab_pdf:
        uploaded_file = st.file_uploader(
            "Upload File",
            type=["pdf", "txt", "docx", "csv", "xlsx", "xls", "json"],
            accept_multiple_files=False,
            label_visibility="collapsed"
        )
        
        # Reset database immediately if document was removed by user
        if st.session_state.indexed_source == "Document" and not uploaded_file:
            delete_collection(st.session_state.collection_name)
            st.session_state.messages = []
            st.session_state.indexed_source = "None"
            st.session_state.source_name = ""
            st.session_state.chunk_count = 0
            st.session_state.db_stats = {"chunks": 0, "pdfs": 0, "websites": 0}
            st.session_state.advanced_expanded = False
            st.session_state.collection_name = f"rag_{int(time.time())}_{st.session_state.session_id[:8]}"
            gc.collect()
            st.rerun()
            
        if uploaded_file:
            # File Size validation check (Max 30 MB)
            MAX_FILE_SIZE = 30 * 1024 * 1024
            if uploaded_file.size > MAX_FILE_SIZE:
                st.error("File size exceeds the 30 MB limit. Please upload a smaller file.")
            else:
                is_ready = (preload_status["embedding"] == "ready")
                btn_pdf_label = "Index Document" if is_ready else "Indexing (Waiting for AI models to preload...)"
                if st.button(btn_pdf_label, key="btn_pdf", disabled=not is_ready):
                    try:
                        t_start_processing = time.time()
                        
                        # Purge old session collection before replacing
                        delete_collection(st.session_state.collection_name)
                        st.session_state.messages = []
                        st.session_state.collection_name = f"rag_{int(time.time())}_{st.session_state.session_id[:8]}"
                        cleanup_expired_collections(max_age_seconds=7200) # Maintenance purge
                        gc.collect()
                        
                        progress_placeholder = st.empty()
                        
                        # Set initial stages progress parameters (without OCR)
                        step_states = {
                            "extract": {"name": "Extracting text", "status": "pending", "pct": 0, "detail": ""},
                            "chunk": {"name": "Chunking", "status": "pending", "pct": 0, "detail": ""},
                            "embed": {"name": "Generating embeddings", "status": "pending", "pct": 0, "detail": ""},
                            "index": {"name": "Indexing vectors", "status": "pending", "pct": 0, "detail": ""}
                        }
                        
                        def render_progress_text(states):
                            lines = ["**✓ File uploaded**\n"]
                            for key, step in states.items():
                                name = step["name"]
                                status = step["status"]
                                pct = step["pct"]
                                detail = step["detail"]
                                
                                if status == "pending":
                                    lines.append(f"⏳ {name}...")
                                elif status == "running":
                                    filled = int(pct / 10)
                                    bar = "█" * filled + "░" * (10 - filled)
                                    lines.append(f"🔄 **{name}...**\n`{bar} {pct}%` {detail}")
                                elif status == "done":
                                    lines.append(f"✓ {name} {detail}")
                                elif status == "skipped":
                                    lines.append(f"✓ {name} (Skipped)")
                            return "\n\n".join(lines)
                        
                        progress_placeholder.markdown(render_progress_text(step_states))
                        
                        # Run document extractor
                        step_states["extract"]["status"] = "running"
                        step_states["extract"]["pct"] = 50
                        progress_placeholder.markdown(render_progress_text(step_states))
                        
                        pages = extract_document_pages(
                            uploaded_file,
                            filename=uploaded_file.name,
                            timeout_seconds=timeout_val
                        )
                        
                        # Finalize extraction phase statuses
                        step_states["extract"]["status"] = "done"
                        step_states["extract"]["pct"] = 100
                        step_states["extract"]["detail"] = ""
                        progress_placeholder.markdown(render_progress_text(step_states))
                        
                        # Stage 3: Chunking
                        step_states["chunk"]["status"] = "running"
                        step_states["chunk"]["pct"] = 50
                        step_states["chunk"]["detail"] = ""
                        progress_placeholder.markdown(render_progress_text(step_states))
                        
                        total_extracted_pages = len(pages)
                        chunks_list, metadata_list = create_parent_child_chunks(
                            pages=pages,
                            source_type="pdf", # Treated as pdf internally for status logs compatibility
                            filename=uploaded_file.name
                        )
                        
                        # Guard against empty chunk generation
                        if len(chunks_list) == 0:
                            delete_collection(st.session_state.collection_name)
                            st.error("No extractable digital text layers found in this document. Since OCR is disabled, purely scanned images or empty documents cannot be processed. Please upload a document containing digital text.")
                            st.stop()
                            
                        # Max chunk limit threshold verification
                        if len(chunks_list) > max_chunk_val:
                            delete_collection(st.session_state.collection_name)
                            st.session_state.advanced_expanded = True
                            st.error(
                                f"⚠️ The uploaded document yields {len(chunks_list)} chunks, which exceeds your current limit of {max_chunk_val} chunks.\n\n"
                                f"We have automatically opened the **Advanced Settings** expander in the left sidebar's Control Panel for you. "
                                f"Please adjust the **Max Chunk Limit** to at least **{len(chunks_list)}** chunks and rebuild the index."
                            )
                            st.stop()
                            
                        step_states["chunk"]["status"] = "done"
                        step_states["chunk"]["pct"] = 100
                        step_states["chunk"]["detail"] = f"({len(chunks_list)} chunks)"
                        progress_placeholder.markdown(render_progress_text(step_states))
                        
                        # Stage 4: Embeddings
                        step_states["embed"]["status"] = "running"
                        step_states["embed"]["pct"] = 0
                        step_states["embed"]["detail"] = ""
                        progress_placeholder.markdown(render_progress_text(step_states))
                        
                        def embed_progress(completed, total):
                            pct_embed = int((completed / total) * 100)
                            step_states["embed"]["pct"] = pct_embed
                            step_states["embed"]["detail"] = f"({completed}/{total} generated)"
                            progress_placeholder.markdown(render_progress_text(step_states))
                            
                        embeddings = generate_embeddings(chunks_list, progress_callback=embed_progress)
                        
                        step_states["embed"]["status"] = "done"
                        step_states["embed"]["pct"] = 100
                        step_states["embed"]["detail"] = ""
                        progress_placeholder.markdown(render_progress_text(step_states))
                        
                        # Stage 5: Indexing Vectors
                        step_states["index"]["status"] = "running"
                        step_states["index"]["pct"] = 0
                        step_states["index"]["detail"] = ""
                        progress_placeholder.markdown(render_progress_text(step_states))
                        
                        store_chunks(st.session_state.collection_name, chunks_list, embeddings, metadata_list)
                        
                        step_states["index"]["status"] = "done"
                        step_states["index"]["pct"] = 100
                        step_states["index"]["detail"] = ""
                        progress_placeholder.markdown(render_progress_text(step_states))
                        
                        time.sleep(0.5)
                        progress_placeholder.empty()
                        
                        # Save session active context profiles
                        st.session_state.indexed_source = "Document"
                        st.session_state.source_name = uploaded_file.name
                        st.session_state.chunk_count = len(chunks_list)
                        st.session_state.db_stats = {"chunks": len(chunks_list), "pdfs": 1, "websites": 0}
                        st.session_state.advanced_expanded = False
                        
                        processing_time = time.time() - t_start_processing
                        st.markdown(f"""
                        <div style="background-color:#1e293b; color:#e2e8f0; border-radius:8px; border:1px solid #334155; padding:1.2rem; margin-bottom:1.5rem;">
                            <span style="color:#22c55e; font-weight:bold; font-size:1.05rem;">✓ Document processed successfully</span><br><br>
                            Pages: <b>{total_extracted_pages}</b><br>
                            Chunks: <b>{len(chunks_list)}</b><br>
                            Processing time: <b>{processing_time:.1f} s</b><br><br>
                            <span style="color:#3b82f6; font-weight:600;">Ready for questions.</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        time.sleep(1.5)
                        st.rerun()
                        
                    except TimeoutError:
                        progress_placeholder.empty()
                        delete_collection(st.session_state.collection_name)
                        st.error("This document is too complex to process efficiently. Please upload a smaller or cleaner PDF.")
                    except Exception as e:
                        progress_placeholder.empty()
                        delete_collection(st.session_state.collection_name)
                        st.error(f"Failed to process PDF: {str(e)}")
                        
    # Website Tab
    with tab_web:
        web_url = st.text_input("Web URL", placeholder="https://example.com/article", label_visibility="collapsed")
        
        is_ready = (preload_status["embedding"] == "ready")
        btn_web_label = "Index Web Page" if is_ready else "Indexing (Waiting for AI models to preload...)"
        if st.button(btn_web_label, key="btn_web", disabled=not is_ready):
            if web_url.strip():
                try:
                    t_start_processing = time.time()
                    
                    # Delete collection metadata before replacing
                    delete_collection(st.session_state.collection_name)
                    st.session_state.messages = []
                    st.session_state.collection_name = f"rag_{int(time.time())}_{st.session_state.session_id[:8]}"
                    cleanup_expired_collections(max_age_seconds=7200) # Maintenance purge
                    gc.collect()
                    
                    progress_placeholder = st.empty()
                    
                    # Web visual status tracker
                    step_states = {
                        "extract": {"name": "Scraping page contents", "status": "pending", "pct": 0, "detail": ""},
                        "chunk": {"name": "Chunking", "status": "pending", "pct": 0, "detail": ""},
                        "embed": {"name": "Generating embeddings", "status": "pending", "pct": 0, "detail": ""},
                        "index": {"name": "Indexing vectors", "status": "pending", "pct": 0, "detail": ""}
                    }
                    
                    def render_progress_text(states):
                        lines = ["**✓ URL provided**\n"]
                        for key, step in states.items():
                            name = step["name"]
                            status = step["status"]
                            pct = step["pct"]
                            detail = step["detail"]
                            
                            if status == "pending":
                                lines.append(f"⏳ {name}...")
                            elif status == "running":
                                filled = int(pct / 10)
                                bar = "█" * filled + "░" * (10 - filled)
                                lines.append(f"🔄 **{name}...**\n`{bar} {pct}%` {detail}")
                            elif status == "done":
                                lines.append(f"✓ {name} {detail}")
                        return "\n\n".join(lines)
                        
                    step_states["extract"]["status"] = "running"
                    step_states["extract"]["pct"] = 50
                    progress_placeholder.markdown(render_progress_text(step_states))
                    
                    web_text = extract_website_text(web_url)
                    if web_text and not web_text.startswith("Error:"):
                        web_text_clean = web_text.strip()
                        
                        # Guard against zero-character extractable HTML parsed text
                        if not web_text_clean:
                            progress_placeholder.empty()
                            delete_collection(st.session_state.collection_name)
                            st.error("Scraped website contains no extractable text content (cookie wall, redirect protection, or empty page).")
                            st.stop()
                            
                        step_states["extract"]["status"] = "done"
                        step_states["extract"]["pct"] = 100
                        
                        # Stage 2: Chunking
                        step_states["chunk"]["status"] = "running"
                        step_states["chunk"]["pct"] = 50
                        progress_placeholder.markdown(render_progress_text(step_states))
                        
                        # Use unified parent-child chunking for web pages
                        web_pages = [{"page": 1, "text": web_text_clean}]
                        web_chunks, metadata = create_parent_child_chunks(
                            pages=web_pages,
                            source_type="website",
                            filename=web_url
                        )
                        
                        # Guard against empty chunk generation
                        if not web_chunks:
                            progress_placeholder.empty()
                            delete_collection(st.session_state.collection_name)
                            st.error("Scraped website content resulted in 0 text chunks.")
                            st.stop()
                            
                        # Validate chunks limit
                        if len(web_chunks) > max_chunk_val:
                            delete_collection(st.session_state.collection_name)
                            st.session_state.advanced_expanded = True
                            st.error(
                                f"⚠️ The scraped website yields {len(web_chunks)} chunks, which exceeds your current limit of {max_chunk_val} chunks.\n\n"
                                f"We have automatically opened the **Advanced Settings** expander in the left sidebar's Control Panel for you. "
                                f"Please adjust the **Max Chunk Limit** to at least **{len(web_chunks)}** chunks and rebuild the index."
                            )
                            st.stop()
                            
                        step_states["chunk"]["status"] = "done"
                        step_states["chunk"]["pct"] = 100
                        step_states["chunk"]["detail"] = f"({len(web_chunks)} chunks)"
                        
                        # Stage 3: Embeddings
                        step_states["embed"]["status"] = "running"
                        step_states["embed"]["pct"] = 0
                        step_states["embed"]["detail"] = ""
                        progress_placeholder.markdown(render_progress_text(step_states))
                        
                        def embed_progress(completed, total):
                            pct_embed = int((completed / total) * 100)
                            step_states["embed"]["pct"] = pct_embed
                            step_states["embed"]["detail"] = f"({completed}/{total} generated)"
                            progress_placeholder.markdown(render_progress_text(step_states))
                            
                        embeddings = generate_embeddings(web_chunks, progress_callback=embed_progress)
                        
                        step_states["embed"]["status"] = "done"
                        step_states["embed"]["pct"] = 100
                        progress_placeholder.markdown(render_progress_text(step_states))
                        
                        # Stage 4: Indexing Vectors
                        step_states["index"]["status"] = "running"
                        step_states["index"]["pct"] = 0
                        progress_placeholder.markdown(render_progress_text(step_states))
                        
                        store_chunks(st.session_state.collection_name, web_chunks, embeddings, metadata)
                        
                        step_states["index"]["status"] = "done"
                        step_states["index"]["pct"] = 100
                        progress_placeholder.markdown(render_progress_text(step_states))
                        
                        time.sleep(0.5)
                        progress_placeholder.empty()
                        
                        st.session_state.indexed_source = "Website"
                        st.session_state.source_name = web_url
                        st.session_state.chunk_count = len(web_chunks)
                        st.session_state.db_stats = {"chunks": len(web_chunks), "pdfs": 0, "websites": 1}
                        st.session_state.advanced_expanded = False
                        
                        processing_time = time.time() - t_start_processing
                        st.markdown(f"""
                        <div style="background-color:#1e293b; color:#e2e8f0; border-radius:8px; border:1px solid #334155; padding:1.2rem; margin-bottom:1.5rem;">
                            <span style="color:#22c55e; font-weight:bold; font-size:1.05rem;">✓ Website indexed successfully</span><br><br>
                            Chunks: <b>{len(web_chunks)}</b><br>
                            Processing time: <b>{processing_time:.1f} s</b><br><br>
                            <span style="color:#3b82f6; font-weight:600;">Ready for questions.</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        progress_placeholder.empty()
                        delete_collection(st.session_state.collection_name)
                        st.error(f"Failed to index website: {web_text}")
                except Exception as e:
                    progress_placeholder.empty()
                    delete_collection(st.session_state.collection_name)
                    st.error(f"Failed to process website: {str(e)}")

# Active status indicator
if db_stats['chunks'] > 0:
    source_type = "Document" if st.session_state.indexed_source == "Document" else "Website"
    if st.session_state.indexed_source == "None":
        source_type = "Previous Session"
        st.session_state.source_name = "Loaded session vectors"
        st.session_state.chunk_count = db_stats['chunks']
        
    st.markdown(f"""
    <div class='status-box'>
        Active Context: <b>{st.session_state.source_name}</b> ({source_type})<br>
        Chunks Indexed: <b>{st.session_state.chunk_count}</b>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class='status-box' style='border-color: #b45309; background-color: rgba(180, 83, 9, 0.05); color: #d97706;'>
        No active context found. Ingest contents to start.
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------
# CHAT LIST & INPUT
# ----------------------------------------------------
chat_container = st.container()

with chat_container:
    # Print clean chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

# Lock prompt if database contains no documents or a query is processing
prompt = st.chat_input("Ask a question...", disabled=is_db_empty or st.session_state.get("processing", False))

if prompt:
    if not api_key:
        st.error("GROQ_API_KEY is not configured.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.current_prompt = prompt
        st.session_state.processing = True
        st.rerun()

# Call LLM with a simple thinking spinner (no emojis or intermediate statuses displayed)
if st.session_state.get("processing", False) and "current_prompt" in st.session_state:
    current_prompt = st.session_state.current_prompt
    with chat_container:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    # Filter out basic greetings, acknowledgments, or politeness turns to bypass retrieval and optimize TPM
                    if is_conversational_query(current_prompt):
                        retrieved = []
                    else:
                        retrieved = search_chunks(
                            st.session_state.collection_name, 
                            current_prompt, 
                            top_k=3, # optimized top_k from 4 to 3 to conserve token limit quota
                            use_hybrid=use_hybrid, 
                            use_reranker=use_reranker
                        )
                    
                    import inspect
                    sig = inspect.signature(generate_answer)
                    if "chat_history" in sig.parameters:
                        response_stream = generate_answer(
                            query=current_prompt,
                            context_chunks=retrieved,
                            api_key=api_key,
                            chat_history=st.session_state.messages[:-1],
                            model=model_choice,
                            temperature=temperature
                        )
                    else:
                        response_stream = generate_answer(
                            query=current_prompt,
                            context_chunks=retrieved,
                            api_key=api_key,
                            model=model_choice,
                            temperature=temperature
                        )
                    
                    placeholder = st.empty()
                    full_response = ""
                    for chunk in response_stream:
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    err = f"Error occurred: {str(e)}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
                finally:
                    st.session_state.processing = False
                    if "current_prompt" in st.session_state:
                        del st.session_state.current_prompt
                    st.rerun()