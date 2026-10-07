"""Tools available to the ReAct agent."""
from ddgs import DDGS
from langchain_core.tools import tool

MAX_RESULTS = 5
# Cloud hosts (e.g. Streamlit Community Cloud) are often rate-limited by a single search engine,
# so try the default metasearch first and then individual engines one by one.
BACKENDS = ["auto", "brave", "yahoo", "mojeek", "startpage", "duckduckgo", "wikipedia"]


def _search(query: str) -> tuple[list[dict], list[str]]:
    errors = []
    for backend in BACKENDS:
        try:
            results = DDGS().text(query, max_results=MAX_RESULTS, backend=backend)
        except Exception as e:
            errors.append(f"{backend}: {e}")
            continue
        if results:
            return results, errors
        errors.append(f"{backend}: no results")
    return [], errors


@tool
def web_search(query: str) -> str:
    """Search the web for current or factual information such as news, recent events, prices,
    sports results, or documentation. Input is a concise search query."""
    results, errors = _search(query)
    if not results:
        return (
            f"Search failed on every engine ({'; '.join(errors)[:300]}). "
            "Tell the user search is unavailable and answer from your own knowledge."
        )
    return "\n\n".join(
        f"[{i}] {r.get('title', '')}\nURL: {r.get('href', '')}\n{r.get('body', '')}"
        for i, r in enumerate(results, 1)
    )


TOOLS = [web_search]
