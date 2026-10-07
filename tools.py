"""Tools available to the ReAct agent."""
import os

import requests
from ddgs import DDGS
from langchain_core.tools import tool

MAX_RESULTS = 5
TAVILY_URL = "https://api.tavily.com/search"
REQUEST_TIMEOUT = 15
# Free scraping backends are often blocked on cloud hosts (e.g. Streamlit Community Cloud); they remain a
# best-effort fallback for local use. Set TAVILY_API_KEY for reliable search anywhere.
BACKENDS = ["auto", "brave", "yahoo", "mojeek", "startpage", "duckduckgo", "wikipedia"]


def _tavily_search(query: str, api_key: str) -> list[dict]:
    response = requests.post(
        TAVILY_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"query": query, "max_results": MAX_RESULTS, "search_depth": "basic"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return [
        {"title": r.get("title", ""), "href": r.get("url", ""), "body": r.get("content", "")}
        for r in response.json().get("results", [])
    ]


def _ddgs_search(query: str) -> tuple[list[dict], list[str]]:
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


def _search(query: str) -> tuple[list[dict], list[str]]:
    errors = []
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key:
        try:
            results = _tavily_search(query, api_key)
            if results:
                return results, errors
            errors.append("tavily: no results")
        except Exception as e:
            errors.append(f"tavily: {type(e).__name__}")  # never include the request (it carries the key)
    results, ddgs_errors = _ddgs_search(query)
    return results, errors + ddgs_errors


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
