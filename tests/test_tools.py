import pytest

import tools


class FakeDDGS:
    """Fake search client: `plan` maps backend -> results list, or an Exception to raise."""

    calls: list[str] = []

    def __init__(self, plan):
        self._plan = plan

    def text(self, query, max_results, backend):
        FakeDDGS.calls.append(backend)
        outcome = self._plan.get(backend, [])
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def no_tavily_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)


def use_plan(monkeypatch, plan):
    FakeDDGS.calls = []
    monkeypatch.setattr(tools, "DDGS", lambda: FakeDDGS(plan))


def test_web_search_formats_results(monkeypatch):
    results = [{"title": "Python 3.14", "href": "https://python.org", "body": "Latest release."}]
    use_plan(monkeypatch, {"auto": results})

    out = tools.web_search.invoke({"query": "python release"})

    assert "[1] Python 3.14" in out
    assert "https://python.org" in out
    assert FakeDDGS.calls == ["auto"]


def test_web_search_falls_back_to_next_backend(monkeypatch):
    results = [{"title": "From Brave", "href": "https://example.com", "body": "ok"}]
    use_plan(monkeypatch, {"auto": RuntimeError("No results found."), "brave": results})

    out = tools.web_search.invoke({"query": "anything"})

    assert "From Brave" in out
    assert FakeDDGS.calls == ["auto", "brave"]


def test_web_search_reports_failure_when_every_backend_fails(monkeypatch):
    use_plan(monkeypatch, {b: RuntimeError("rate limited") for b in tools.BACKENDS})

    out = tools.web_search.invoke({"query": "anything"})

    assert "Search failed" in out
    assert FakeDDGS.calls == tools.BACKENDS


class FakeResponse:
    def __init__(self, payload, status_ok=True):
        self._payload, self._ok = payload, status_ok

    def raise_for_status(self):
        if not self._ok:
            raise RuntimeError("401 Unauthorized")

    def json(self):
        return self._payload


def test_tavily_is_used_when_key_is_set(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    use_plan(monkeypatch, {})
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, auth=headers["Authorization"], query=json["query"])
        return FakeResponse({"results": [{"title": "T", "url": "https://t.dev", "content": "from tavily"}]})

    monkeypatch.setattr(tools.requests, "post", fake_post)

    out = tools.web_search.invoke({"query": "streamlit version"})

    assert "from tavily" in out and "https://t.dev" in out
    assert seen == {"url": tools.TAVILY_URL, "auth": "Bearer tvly-test", "query": "streamlit version"}
    assert FakeDDGS.calls == []  # no fallback needed


def test_tavily_failure_falls_back_without_leaking_the_key(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret")
    use_plan(monkeypatch, {b: RuntimeError("blocked") for b in tools.BACKENDS})
    monkeypatch.setattr(tools.requests, "post", lambda *a, **k: FakeResponse({}, status_ok=False))

    out = tools.web_search.invoke({"query": "anything"})

    assert "tavily: RuntimeError" in out
    assert "tvly-secret" not in out
    assert FakeDDGS.calls == tools.BACKENDS
