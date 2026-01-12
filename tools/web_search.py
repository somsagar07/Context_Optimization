from ddgs import DDGS
import time

def web_search(query: str, max_retries: int = 3, max_results: int = 5) -> str:
    """
    Enhanced Search Wrapper with retry logic (using Google backend).
    Returns multiple results with titles, URLs, and snippets for comprehensive information.
    
    Args:
        query: Search query string
        max_retries: Maximum number of retry attempts (default: 3)
        max_results: Maximum number of results to return (default: 5, was 1)
    
    Returns:
        Formatted search results string with multiple entries, or error message
    """
    # Clean and validate query
    query = query.strip()
    if not query:
        return "Error: Empty search query provided."
    
    # Limit query length to avoid API issues
    if len(query) > 200:
        query = query[:200]
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            ddgs = DDGS()
            # Using Google backend instead of DuckDuckGo
            results = ddgs.text(query, backend="google", max_results=max_results)
            
            if not results or len(results) == 0:
                return "No results found for your query."
            
            # Format results in a clear, structured way
            formatted_results = []
            for idx, result in enumerate(results, 1):
                # Extract fields (handle missing fields gracefully)
                title = result.get('title', 'No title')
                body = result.get('body', 'No description')
                href = result.get('href', '')
                
                # Format each result
                result_str = f"[Result {idx}]\n"
                result_str += f"Title: {title}\n"
                if href:
                    result_str += f"URL: {href}\n"
                result_str += f"Content: {body}\n"
                formatted_results.append(result_str)
            
            # Combine all results
            output = f"Found {len(results)} result(s) for: \"{query}\"\n\n"
            output += "\n---\n\n".join(formatted_results)
            
            return output
            
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 10)  # Exponential backoff, max 10s
                time.sleep(wait_time)
            else:
                return f"Search Error after {max_retries} attempts: {str(e)}"
    
    return f"Search Error: {str(last_exception) if last_exception else 'Unknown error'}"

