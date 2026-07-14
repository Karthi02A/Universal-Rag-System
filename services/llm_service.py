import requests

def is_conversational_query(text):
    """
    Identifies if a prompt is simply a greeting, acknowledgment, or basic politeness turn
    to bypass context document retrieval and save token budgets.
    Supports clean token-level checks and common spelling variations or phrases.
    """
    text_clean = text.lower().strip().strip("?!.,\"'")
    words = text_clean.split()
    if not words:
        return True
        
    conversational_words = {
        "hi", "hello", "hey", "hii", "helloo", "yoo", "yo", "greetings", "there",
        "thanks", "thank", "thankyou", "thanksyou", "thx", "ty", "you", "a", "lot", "very", "much",
        "yes", "no", "ok", "okay", "yep", "yup", "sure", "cool", 
        "bye", "goodbye", "help", "info", "test", "testing"
    }
    
    # Check if all words in the input are in our conversational word set
    if all(w in conversational_words for w in words):
        return True
        
    # Check for common phrases
    phrases = ["thank you", "thanks a lot", "no thanks", "no thank you", "many thanks"]
    if any(p in text_clean for p in phrases) and len(text_clean) < 30:
        return True
        
    return False

def generate_answer(
    query,
    context_chunks,
    api_key,
    chat_history=None,
    model="llama-3.3-70b-versatile",
    temperature=0.2
):
    """
    Calls the Groq completions API to generate an answer based only on context chunks.
    The system minimizes hallucinations by restricting the LLM to retrieved document context.
    When no supporting evidence exists, it responds that the information could not be found in the provided source.
    """
    if not api_key or not api_key.strip():
        raise ValueError("Groq API Key is not set. Please provide it in the sidebar.")

    # Check for empty context
    if not context_chunks:
        if is_conversational_query(query):
            # Polite response guiding user to ask about documents
            system_prompt = (
                "You are a helpful assistant for the Universal RAG System.\n"
                "The user is engaging in general conversation or greeting. Respond politely and guide them "
                "to ask questions about their uploaded documents or web content.\n"
                "Do not reference any specific document details or ratings unless the user specifically asks you about them. "
                "Be concise. Do not use emojis."
            )
            user_prompt = query
        else:
            return "I cannot find the answer in the provided documents."
    else:
        # Format the context chunks for the prompt with index labels
        formatted_chunks = []
        for idx, chunk in enumerate(context_chunks):
            meta = chunk.get("metadata", {}) or {}
            source_info = "Unknown Source"
            if meta.get("source") == "pdf":
                source_info = f"PDF: {meta.get('filename', 'Document')}, Page {meta.get('page', 1)}"
            elif meta.get("source") == "website":
                source_info = f"Website URL: {meta.get('url', 'URL')}"
                
            formatted_chunks.append(
                f"--- Source [{idx+1}]: {source_info} ---\n{chunk['text']}"
            )
            
        context_str = "\n\n".join(formatted_chunks)
        
        system_prompt = (
            "You are a helpful, professional AI assistant for the Universal RAG System.\n"
            "Your task is to answer the user's question using ONLY the provided Sources. "
            "Strictly adhere to the following rules:\n"
            "1. Base your answer solely on the facts presented in the Sources.\n"
            "2. If the answer cannot be found in the Sources, say: 'I cannot find the answer in the provided documents.' Do not invent facts or draw unsupported conclusions. If the user asks for a specific row number (e.g., Row 322), verify that the exact row number is displayed in the Sources. If the requested row number is not present in the Sources, state: 'Row X is not present in my source context.' Do not list the content of other rows.\n"
            "3. Format your answer neatly and professionally: use bullet points, numbered lists, or short structured paragraphs when explaining concepts. Clear, bulleted information is preferred over long walls of text.\n"
            "4. Do not include inline citations (such as [Source 1]) or raw prompt headers (such as --- Source [1] ---) within the sentences of your response. Instead, at the very end of your response under a 'Sources:' heading, list only the unique filenames and their referenced page numbers (or URLs for websites) as a clean bulleted list (e.g., '- PDF: filename.pdf (Page X)' or '- Website: URL'). Do not output raw prompt markers or headers.\n"
            "5. Be concise, direct, factually accurate, and professional. Do not use emojis in your response.\n"
            "6. ADAPT RESPONSE LENGTH DYNAMICALLY:\n"
            "   - For quick, direct, or factual queries (e.g., specific metrics, yes/no, GPAs, names, dates, scores, ratings), answer immediately and concisely in 1 or 2 sentences. Do not add background analysis or extra details.\n"
            "   - For analytical, concept-exploring, or detail-oriented queries (e.g., summarize, explain why or how, contrast ideas), provide a comprehensive, structured response.\n"
            "   - Strictly follow explicit formatting requirements specified by the user (e.g., if asked 'rate my resume out of 10 with no explanation', output ONLY the rating number/score)."
        )
        
        user_prompt = f"Sources:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
    
    # Construct message trail starting with system rules
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Inject active session chat history logs (exclude current prompt, limit size to 4 messages for TPM conservation)
    if chat_history:
        for msg in chat_history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
    # Include current user query carrying parsed document context
    messages.append({"role": "user", "content": user_prompt})
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024,
        "stream": True
    }
    
    import time
    max_retries = 3
    retry_delay = 1.0  # start with 1 second delay
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
                stream=True
            )
            
            if response.status_code == 200:
                import json
                for line in response.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data:"):
                        data_val = line_str[5:].strip()
                        if data_val == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data_val)
                            delta = chunk_data["choices"][0]["delta"]
                            if "content" in delta:
                                yield delta["content"]
                        except Exception:
                            pass
                return
            elif response.status_code == 429: # Rate Limit
                # Extract wait time if offered in headers (e.g. Retry-After)
                retry_after_str = response.headers.get("Retry-After")
                if retry_after_str:
                    try:
                        wait_sec = float(retry_after_str) + 0.1
                    except ValueError:
                        wait_sec = retry_delay
                else:
                    wait_sec = retry_delay
                
                if attempt < max_retries - 1:
                    time.sleep(wait_sec)
                    retry_delay *= 2  # double backoff time
                    continue
                else:
                    # Final attempt failed
                    raise Exception("Groq API Rate Limit reached (Status 429). Please try again in a moment.")
            else:
                try:
                    import json
                    err_data = response.json()
                    err_msg = err_data.get("error", {}).get("message", response.text)
                except Exception:
                    err_msg = response.text
                raise Exception(f"Groq API Error (Status {response.status_code}): {err_msg}")
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            raise Exception("Request to Groq API timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network request to Groq API failed: {str(e)}")

