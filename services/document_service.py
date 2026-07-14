import os
import json

def extract_document_pages(file_obj, filename, timeout_seconds=120):
    """
    Extracts text from various file formats (pdf, txt, json, docx, csv, xlsx, xls)
    and returns a list of page objects: [{'page': int, 'text': str}]
    """
    ext = os.path.splitext(filename.lower())[1]
    
    if ext == ".pdf":
        from services.pdf_service import extract_pdf_pages
        return extract_pdf_pages(file_obj, timeout_seconds=timeout_seconds)
        
    elif ext in [".txt", ".md"]:
        file_obj.seek(0)
        content = file_obj.read()
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="ignore")
        else:
            text = str(content)
        return [{"page": 1, "text": text.strip()}]
        
    elif ext == ".json":
        file_obj.seek(0)
        content = file_obj.read()
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="ignore")
        else:
            text = str(content)
        try:
            data = json.loads(text)
            formatted_text = json.dumps(data, indent=2)
        except Exception:
            formatted_text = text
        return [{"page": 1, "text": formatted_text.strip()}]
        
    elif ext == ".docx":
        import docx
        file_obj.seek(0)
        doc = docx.Document(file_obj)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        text_content = "\n".join(full_text)
        return [{"page": 1, "text": text_content.strip()}]
        
    elif ext == ".csv":
        import pandas as pd
        file_obj.seek(0)
        df = pd.read_csv(file_obj)
        # Drop completely empty rows and columns
        df = df.dropna(how="all").dropna(axis=1, how="all")
        rows_text = []
        for idx, row in df.iterrows():
            # Skip empty/nan cells
            parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val) and str(val).strip().lower() != "nan"]
            if parts:
                rows_text.append(f"Row {idx + 1}: {', '.join(parts)}")
        text_content = "\n".join(rows_text)
        return [{"page": 1, "text": text_content.strip()}]
        
    elif ext in [".xlsx", ".xls"]:
        import pandas as pd
        file_obj.seek(0)
        xls = pd.ExcelFile(file_obj)
        sheets_text = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            # Drop completely empty rows and columns
            df = df.dropna(how="all").dropna(axis=1, how="all")
            rows_text = [f"--- Sheet: {sheet_name} ---"]
            for idx, row in df.iterrows():
                parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val) and str(val).strip().lower() != "nan"]
                if parts:
                    rows_text.append(f"Row {idx + 1}: {', '.join(parts)}")
            if len(rows_text) > 1: # Only append if sheet has row data
                sheets_text.append("\n".join(rows_text))
        text_content = "\n\n".join(sheets_text)
        return [{"page": 1, "text": text_content.strip()}]
        
    else:
        raise ValueError(f"Unsupported file format: {ext}")
