import requests

def extract_website_text(url):
    try:
        from bs4 import BeautifulSoup
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Decompose scripts and styles to clean HTML text extraction
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        return text

    except Exception as e:
        return f"Error: {e}"