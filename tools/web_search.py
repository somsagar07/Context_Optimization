from ddgs import DDGS

def web_search(query: str) -> str:
    """DuckDuckGo Search Wrapper."""
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=1)
        if results:
            return f"Search Result: {results[0]['body']}"
        return "No results found."
    except Exception as e:
        return f"Search Error: {str(e)}"

