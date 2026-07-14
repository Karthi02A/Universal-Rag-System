import streamlit as st

@st.cache_resource
def get_ocr_reader():
    """
    Lazy initialization of EasyOCR reader to save startup time and memory.
    Downloads the model files on the first execution.
    """
    import easyocr
    import logging
    # Configure logger to suppress verbose output
    logging.getLogger('easyocr').setLevel(logging.WARNING)
    
    # Initialize for English
    return easyocr.Reader(['en'], gpu=False)

def ocr_image_bytes(image_bytes):
    """
    Run OCR on image bytes and return the extracted text.
    """
    try:
        reader = get_ocr_reader()
        # Perform OCR
        results = reader.readtext(image_bytes, detail=0)
        return " ".join(results)
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""
