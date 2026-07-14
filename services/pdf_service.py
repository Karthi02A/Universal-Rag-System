import fitz
import re
import time

def extract_pdf_pages(pdf_file, progress_callback=None, timeout_seconds=120):
    """
    Extracts text page-by-page. Automatically detects whether a page contains
    extractable text. Only applies OCR to pages without text (scanned pages)
    entirely in-memory.
    
    timeout_seconds: maximum time allowed for processing before raising TimeoutError.
    progress_callback: a function that takes (stage, current, total, status_text)
    """
    t_start = time.time()
    
    # Reset file pointer just in case
    pdf_file.seek(0)
    
    pdf_document = fitz.open(
        stream=pdf_file.read(),
        filetype="pdf"
    )
    
    total_pages = len(pdf_document)
    pages_data = []
    
    scanned_pages = []
    text_by_page = {}
    
    # Phase 1: Fast text extraction pass
    for page_idx in range(total_pages):
        # Timeout check
        if time.time() - t_start > timeout_seconds:
            pdf_document.close()
            raise TimeoutError("Processing timeout exceeded")
            
        page_num = page_idx + 1
        page = pdf_document[page_idx]
        text = page.get_text()
        clean_text = text.strip()
        
        has_text = bool(re.search(r'[a-zA-Z0-9]', clean_text))
        if has_text:
            text_by_page[page_num] = clean_text
        else:
            scanned_pages.append(page_num)
            
        if progress_callback:
            progress_callback(
                "extracting",
                page_num,
                total_pages,
                f"Page {page_num}/{total_pages}"
            )
            
    pdf_document.close()
    
    # Reassemble in correct order
    for page_num in sorted(text_by_page.keys()):
        pages_data.append({
            "page": page_num,
            "text": text_by_page[page_num]
        })
        
    return pages_data

def extract_pdf_text(pdf_file, timeout_seconds=120):
    """
    Backwards compatible helper returning all text as a single string.
    """
    pages = extract_pdf_pages(pdf_file, timeout_seconds=timeout_seconds)
    return "\n\n".join([p["text"] for p in pages])
