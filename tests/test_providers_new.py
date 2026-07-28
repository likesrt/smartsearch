import json

import httpx
import pytest

from smart_search.providers.context7 import Context7Provider
from smart_search.providers.exa import ExaSearchProvider
from smart_search.providers.zhipu import ZhipuWebSearchProvider


@pytest.mark.asyncio
async def test_zhipu_provider_normalizes_search_results(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, timeout, follow_redirects=True):
            self.timeout = timeout
            self.follow_redirects = follow_redirects

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, endpoint, headers, json):
            return httpx.Response(
                200,
                json={
                    "request_id": "r1",
                    "search_result": [
                        {
                            "title": "Title",
                            "content": "Snippet",
                            "link": "https://example.com",
                            "media": "Example",
                            "publish_date": "2026-05-12",
                        }
                    ],
                },
                request=httpx.Request("POST", endpoint),
            )

    monkeypatch.setattr("smart_search.providers.zhipu.httpx.AsyncClient", FakeAsyncClient)
    provider = ZhipuWebSearchProvider("https://open.bigmodel.cn/api", "key")

    data = json.loads(await provider.search("hello"))

    assert data["ok"] is True
    assert data["results"][0]["url"] == "https://example.com"
    assert data["results"][0]["provider"] == "zhipu"


@pytest.mark.asyncio
async def test_zhipu_provider_uses_configured_engine_and_call_override(monkeypatch):
    payloads = []

    class FakeAsyncClient:
        def __init__(self, timeout, follow_redirects=True):
            self.timeout = timeout
            self.follow_redirects = follow_redirects

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, endpoint, headers, json):
            payloads.append(json.copy())
            return httpx.Response(
                200,
                json={"request_id": "r1", "search_result": []},
                request=httpx.Request("POST", endpoint),
            )

    monkeypatch.setattr("smart_search.providers.zhipu.httpx.AsyncClient", FakeAsyncClient)
    provider = ZhipuWebSearchProvider("https://open.bigmodel.cn/api", "key", search_engine="search_pro")

    data = json.loads(await provider.search("hello"))
    override_data = json.loads(await provider.search("hello", search_engine="search_pro_quark"))

    assert data["search_engine"] == "search_pro"
    assert override_data["search_engine"] == "search_pro_quark"
    assert payloads[0]["search_engine"] == "search_pro"
    assert payloads[1]["search_engine"] == "search_pro_quark"


@pytest.mark.asyncio
async def test_zhipu_provider_reports_rate_limit_without_retry(monkeypatch):
    calls = []

    class FakeAsyncClient:
        def __init__(self, timeout, follow_redirects=True):
            self.timeout = timeout
            self.follow_redirects = follow_redirects

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, endpoint, headers, json):
            calls.append(endpoint)
            return httpx.Response(
                429,
                json={"error": "rate limited"},
                request=httpx.Request("POST", endpoint),
            )

    monkeypatch.setattr("smart_search.providers.zhipu.httpx.AsyncClient", FakeAsyncClient)
    provider = ZhipuWebSearchProvider("https://open.bigmodel.cn/api", "key")

    data = json.loads(await provider.search("test"))

    assert data["ok"] is False
    assert data["error_type"] == "rate_limited"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_context7_provider_normalizes_library_results(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, timeout, follow_redirects=True):
            self.timeout = timeout
            self.follow_redirects = follow_redirects

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, endpoint, headers):
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "/reactjs/react.dev",
                        "title": "React",
                        "description": "Official React documentation",
                        "trustScore": 10,
                        "benchmarkScore": 88.52,
                        "score": 250,
                        "verified": True,
                        "vip": True,
                    },
                    {
                        "id": "/devopshq/artifactory-cleanup",
                        "title": "Artifactory Cleanup",
                        "description": "Cleanup jobs",
                        "trustScore": 9.2,
                        "benchmarkScore": 48,
                        "score": 452,
                        "verified": True,
                    },
                ],
                headers={"content-type": "application/json"},
                request=httpx.Request("GET", endpoint),
            )

    monkeypatch.setattr("smart_search.providers.context7.httpx.AsyncClient", FakeAsyncClient)
    provider = Context7Provider("https://context7.com", "key")

    data = json.loads(await provider.library("react", "hooks"))

    assert data["ok"] is True
    assert data["results"][0]["id"] == "/reactjs/react.dev"
    assert data["results"][0]["verified"] is True
    assert data["results"][0]["vip"] is True
    assert data["results"][0]["score"] == 250
    assert data["results"][0]["provider"] == "context7"


@pytest.mark.asyncio
async def test_context7_docs_normalizes_content_only_response(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, timeout, follow_redirects=True):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, endpoint, headers):
            return httpx.Response(
                200,
                json={"content": "### Corrected Effect\nReturn a cleanup function."},
                headers={"content-type": "application/json"},
                request=httpx.Request("GET", endpoint),
            )

    monkeypatch.setattr("smart_search.providers.context7.httpx.AsyncClient", FakeAsyncClient)
    provider = Context7Provider("https://context7.com", "key")

    data = json.loads(await provider.docs("/reactjs/react.dev", "useEffect cleanup"))

    assert data["ok"] is True
    assert data["content"].startswith("### Corrected Effect")
    assert data["total"] == 1
    assert data["results"][0]["evidence_type"] == "documentation"
    assert data["results"][0]["content"] == data["content"]
    assert data["raw_response"]["content"] == data["content"]


@pytest.mark.asyncio
async def test_context7_docs_unwraps_json_string_content(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, timeout, follow_redirects=True):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, endpoint, headers):
            return httpx.Response(
                200,
                json={"content": json.dumps({"content": "React cleanup docs"})},
                headers={"content-type": "application/json"},
                request=httpx.Request("GET", endpoint),
            )

    monkeypatch.setattr("smart_search.providers.context7.httpx.AsyncClient", FakeAsyncClient)
    provider = Context7Provider("https://context7.com", "key")

    data = json.loads(await provider.docs("/reactjs/react.dev", "cleanup"))

    assert data["content"] == "React cleanup docs"
    assert data["total"] == 1


@pytest.mark.asyncio
async def test_context7_docs_reports_library_redirect(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, timeout, follow_redirects=True):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, endpoint, headers):
            return httpx.Response(
                301,
                json={
                    "error": "library_redirected",
                    "message": "Library /facebook/react has been redirected",
                    "redirectUrl": "/react/react",
                },
                headers={"content-type": "application/json"},
                request=httpx.Request("GET", endpoint),
            )

    monkeypatch.setattr("smart_search.providers.context7.httpx.AsyncClient", FakeAsyncClient)
    provider = Context7Provider("https://context7.com", "key")

    data = json.loads(await provider.docs("/facebook/react", "hooks"))

    assert data["ok"] is False
    assert data["error_type"] == "library_redirected"
    assert data["redirected_to"] == "/react/react"


@pytest.mark.asyncio
async def test_exa_provider_reports_bad_request_as_parameter_error(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, endpoint, headers, json):
            return httpx.Response(
                400,
                json={"error": "invalid includeDomains"},
                request=httpx.Request("POST", endpoint),
            )

    monkeypatch.setattr("smart_search.providers.exa.httpx.AsyncClient", FakeAsyncClient)
    provider = ExaSearchProvider("https://api.exa.ai", "key")

    data = json.loads(await provider.search("test", include_domains=["github.com freertos.org"]))

    assert data["ok"] is False
    assert data["error_type"] == "parameter_error"
    assert "HTTP 400" in data["error"]
    assert "invalid includeDomains" in data["error"]
