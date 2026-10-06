"""Tools available to the ReAct agent."""
from ddgs import DDGS
from langchain_core.tools import tool

MAX_RESULTS = 5


@tool
def web_search(query: str) -> str:
    """Search the web for current or factual information such as news, recent events, prices,
    sports results, or documentation. Input is a concise search query."""
    try:
        results = DDGS().text(query, max_results=MAX_RESULTS)
    except Exception as e:
        return f"Search failed ({e}). Tell the user search is unavailable and answer from your own knowledge."
    if not results:
        return "No results found. Try rephrasing the query."
    return "\n\n".join(
        f"[{i}] {r.get('title', '')}\nURL: {r.get('href', '')}\n{r.get('body', '')}"
        for i, r in enumerate(results, 1)
    )


TOOLS = [web_search]
