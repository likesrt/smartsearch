import json
import re
import time
from typing import Any
from urllib.parse import quote

import httpx
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from .base import BaseSearchProvider
from ..config import config
from ..logger import log_info


RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _is_retryable_exception(exc) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return False


def _normalize_library(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id") or "",
        "title": item.get("title") or "",
        "description": item.get("description") or "",
        "trust_score": item.get("trustScore"),
        "benchmark_score": item.get("benchmarkScore"),
        "total_snippets": item.get("totalSnippets"),
        "total_tokens": item.get("totalTokens"),
        "stars": item.get("stars"),
        "score": item.get("score"),
        "verified": bool(item.get("verified")),
        "vip": bool(item.get("vip")),
        "state": item.get("state") or "",
        "versions": item.get("versions") or [],
        "provider": "context7",
    }


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _library_match_score(item: dict[str, Any], query: str) -> float:
    query_text = query.casefold()
    query_tokens = set(re.findall(r"[\w.+-]+", query_text))
    title = str(item.get("title") or "").casefold().strip()
    library_id = str(item.get("id") or "").casefold()
    title_tokens = set(re.findall(r"[\w.+-]+", title))
    overlap = len(query_tokens & title_tokens) / max(len(title_tokens), 1)
    score = overlap * 40
    if title and title in query_tokens:
        score += 100
    if title and title == query_text.strip():
        score += 150
    if item.get("verified"):
        score += 25
    if item.get("vip"):
        score += 20
    if "official" in query_tokens and (item.get("verified") or item.get("vip")):
        score += 25
    if any(token and token in library_id for token in query_tokens):
        score += 10
    score += min(_number(item.get("trust_score")), 10) * 2
    score += min(_number(item.get("benchmark_score")), 100) * 0.2
    score += min(_number(item.get("score")), 500) * 0.01
    return score


def rank_library_candidates(results: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Return Context7 candidates in a stable, intent-aware order."""
    indexed = list(enumerate(results))
    indexed.sort(key=lambda pair: (-_library_match_score(pair[1], query), pair[0]))
    return [item for _, item in indexed]


def _docs_content(data: Any) -> str:
    if isinstance(data, str):
        text = data.strip()
        if text.startswith(("{", "[")):
            try:
                return _docs_content(json.loads(text))
            except json.JSONDecodeError:
                pass
        return text
    if isinstance(data, dict):
        content = data.get("content")
        if content:
            return _docs_content(content)
        parts: list[str] = []
        for key in ("codeSnippets", "infoSnippets"):
            for item in data.get(key) or []:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    value = item.get("content") or item.get("description") or item.get("text")
                    if value:
                        parts.append(str(value))
        return "\n\n".join(parts).strip()
    if isinstance(data, list):
        return "\n\n".join(filter(None, (_docs_content(item) for item in data))).strip()
    return "" if data is None else str(data).strip()


def _context7_error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        try:
            body = response.json()
        except (ValueError, json.JSONDecodeError):
            body = {}
        if response.status_code in {301, 302, 307, 308} and isinstance(body, dict):
            redirected_to = body.get("redirectUrl") or response.headers.get("location") or ""
            return {
                "error_type": "library_redirected",
                "error": body.get("message") or str(exc),
                "redirected_to": redirected_to,
            }
        if response.status_code in {401, 403}:
            error_type = "auth_error"
        elif response.status_code == 429:
            error_type = "rate_limited"
        elif response.status_code in {400, 404, 422}:
            error_type = "parameter_error"
        else:
            error_type = "network_error"
        message = body.get("message") or body.get("error") if isinstance(body, dict) else ""
        return {"error_type": error_type, "error": str(message or exc)}
    if isinstance(exc, httpx.TimeoutException):
        return {"error_type": "timeout", "error": "request timed out"}
    if isinstance(exc, httpx.RequestError):
        return {"error_type": "network_error", "error": str(exc)}
    return {"error_type": "runtime_error", "error": str(exc)}


class Context7Provider(BaseSearchProvider):
    def __init__(self, api_url: str, api_key: str, timeout: float = 30.0):
        super().__init__(api_url.rstrip("/"), api_key)
        self.timeout = timeout

    def get_provider_name(self) -> str:
        return "Context7"

    async def search(self, query: str, max_results: int = 5) -> str:
        return await self.library(query)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/plain",
            "X-Context7-Source": "smart-search",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def library(self, name: str, query: str = "", ctx=None) -> str:
        request_query = f"{name} {query}".strip()
        endpoint = f"{self.api_url}/api/v2/search?query={quote(request_query)}"
        await log_info(ctx, f"Context7 library: {request_query}", config.debug_enabled)
        start_time = time.time()
        try:
            data = await self._get_with_retry(endpoint)
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            raw_results = data if isinstance(data, list) else data.get("results", [])
            results = [_normalize_library(item) for item in raw_results or []]
            results = rank_library_candidates(results, request_query)
            output = {
                "ok": True,
                "query": request_query,
                "provider": "context7",
                "results": results,
                "total": len(results),
                "elapsed_ms": elapsed_ms,
            }
        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            error = _context7_error(e)
            output = {
                "ok": False,
                "query": request_query,
                "provider": "context7",
                **error,
                "elapsed_ms": elapsed_ms,
            }
        return json.dumps(output, ensure_ascii=False, indent=2)

    async def docs(self, library_id: str, query: str, ctx=None) -> str:
        endpoint = f"{self.api_url}/api/v2/context?libraryId={quote(library_id, safe='')}&query={quote(query)}"
        await log_info(ctx, f"Context7 docs: {library_id} {query}", config.debug_enabled)
        start_time = time.time()
        try:
            data = await self._get_with_retry(endpoint)
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            snippets = data.get("codeSnippets", []) if isinstance(data, dict) else []
            info = data.get("infoSnippets", []) if isinstance(data, dict) else []
            content = _docs_content(data)
            results = snippets + info
            if content and not results:
                results = [
                    {
                        "title": "Context7 documentation",
                        "description": content[:500],
                        "content": content,
                        "evidence_type": "documentation",
                        "provider": "context7",
                    }
                ]
            output = {
                "ok": True,
                "library_id": library_id,
                "query": query,
                "provider": "context7",
                "code_snippets": snippets,
                "info_snippets": info,
                "results": results,
                "total": len(results),
                "content": content,
                "raw_response": data,
                "elapsed_ms": elapsed_ms,
            }
        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            error = _context7_error(e)
            output = {
                "ok": False,
                "library_id": library_id,
                "query": query,
                "provider": "context7",
                **error,
                "elapsed_ms": elapsed_ms,
            }
        return json.dumps(output, ensure_ascii=False, indent=2)

    async def _get_with_retry(self, endpoint: str) -> Any:
        timeout = httpx.Timeout(connect=6.0, read=self.timeout, write=10.0, pool=None)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(config.retry_max_attempts + 1),
                wait=wait_random_exponential(multiplier=config.retry_multiplier, max=config.retry_max_wait),
                retry=retry_if_exception(_is_retryable_exception),
                reraise=True,
            ):
                with attempt:
                    response = await client.get(endpoint, headers=self._headers())
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "")
                    if "application/json" in content_type:
                        return response.json()
                    text = response.text
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return {"content": text, "results": []}
        return {}
