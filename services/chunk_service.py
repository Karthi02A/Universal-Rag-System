import uuid

_child_splitter = None
_parent_splitter = None

def get_child_splitter():
    global _child_splitter
    if _child_splitter is None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        _child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100
        )
    return _child_splitter

def get_parent_splitter():
    global _parent_splitter
    if _parent_splitter is None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        _parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=500
        )
    return _parent_splitter

def create_chunks(text):
    """
    Backwards compatible standard chunker.
    """
    splitter = get_child_splitter()
    return splitter.split_text(text)

def create_parent_child_chunks(pages, source_type="pdf", filename=""):
    """
    Splits text page-by-page. For short pages, the page itself serves as the parent block.
    For longer pages (or parsed CSVs, Excel, JSON or long txt files), they are split
    into parent blocks first, then child blocks. Each child chunk metadata contains its
    parent block's raw text under the 'parent_text' key.
    """
    child_splitter = get_child_splitter()
    parent_splitter = get_parent_splitter()
    
    chunks_list = []
    metadata_list = []
    
    for page_data in pages:
        p_num = page_data["page"]
        p_text = page_data["text"]
        
        # If the page text is small, treat the whole page as the parent context.
        if len(p_text) <= 3500:
            p_chunks = child_splitter.split_text(p_text)
            for i, chunk in enumerate(p_chunks):
                chunks_list.append(chunk)
                metadata_list.append({
                    "source": source_type,
                    "filename": filename,
                    "page": p_num,
                    "chunk_id": f"{source_type}_{p_num}_{i}_{uuid.uuid4().hex[:6]}",
                    "parent_text": p_text
                })
        else:
            # If the page text is very large, split it into 3,000 char parent chunks.
            parent_chunks = parent_splitter.split_text(p_text)
            for p_idx, parent_text in enumerate(parent_chunks):
                children = child_splitter.split_text(parent_text)
                for c_idx, child_text in enumerate(children):
                    chunks_list.append(child_text)
                    metadata_list.append({
                        "source": source_type,
                        "filename": filename,
                        "page": p_num,
                        "chunk_id": f"{source_type}_{p_num}_{p_idx}_{c_idx}_{uuid.uuid4().hex[:6]}",
                        "parent_text": parent_text
                    })
                    
    return chunks_list, metadata_list