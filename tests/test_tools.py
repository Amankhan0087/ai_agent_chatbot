import tools


class FakeDDGS:
    def __init__(self, results=None, error=None):
        self._results, self._error = results, error

    def text(self, query, max_results):
        if self._error:
            raise self._error
        return self._results


def test_web_search_formats_results(monkeypatch):
    results = [{"title": "Python 3.14", "href": "https://python.org", "body": "Latest release."}]
    monkeypatch.setattr(tools, "DDGS", lambda: FakeDDGS(results))

    out = tools.web_search.invoke({"query": "python release"})

    assert "[1] Python 3.14" in out
    assert "https://python.org" in out


def test_web_search_handles_no_results(monkeypatch):
    monkeypatch.setattr(tools, "DDGS", lambda: FakeDDGS([]))
    assert "No results" in tools.web_search.invoke({"query": "zzzz"})


def test_web_search_reports_failures_instead_of_raising(monkeypatch):
    monkeypatch.setattr(tools, "DDGS", lambda: FakeDDGS(error=RuntimeError("rate limited")))
    assert "Search failed" in tools.web_search.invoke({"query": "anything"})
