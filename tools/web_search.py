from ddgs import DDGS
import time

def web_search(query: str, max_retries: int = 3) -> str:
    """
    DuckDuckGo Search Wrapper with retry logic.
    
    Args:
        query: Search query string
        max_retries: Maximum number of retry attempts (default: 3)
    
    Returns:
        Search result string or error message
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            ddgs = DDGS()
            results = ddgs.text(query, max_results=1)
            if results:
                return f"Search Result: {results[0]['body']}"
            return "No results found."
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 10)  # Exponential backoff, max 10s
                time.sleep(wait_time)
            else:
                return f"Search Error after {max_retries} attempts: {str(e)}"
    
    return f"Search Error: {str(last_exception) if last_exception else 'Unknown error'}"

