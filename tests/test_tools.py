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
