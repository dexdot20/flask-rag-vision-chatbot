from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import random
import re
import socket
import unicodedata
import warnings
import xml.etree.ElementTree as ET
from io import BytesIO
from urllib.parse import quote as url_quote
from urllib.parse import urlparse

import requests as http_requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from pypdf import PdfReader
from urllib3.exceptions import InsecureRequestWarning

from config import (
    CONTENT_MAX_CHARS,
    FETCH_MAX_REDIRECTS,
    FETCH_MAX_SIZE,
    FETCH_TIMEOUT,
    PRIVATE_NETWORKS,
    PROXIES_PATH,
    SEARCH_MAX_RESULTS,
)
from db import cache_get, cache_set

_BROWSER_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.82 Mobile Safari/537.36",
]
_CHROME_SEC_UA = [
    '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    '"Chromium";v="123", "Google Chrome";v="123", "Not-A.Brand";v="99"',
    '"Microsoft Edge";v="124", "Chromium";v="124", "Not-A.Brand";v="99"',
]
_ACCEPT_LANGS = [
    "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    "en-US,en;q=0.9,tr;q=0.8",
    "en-GB,en;q=0.9,en-US;q=0.8",
    "tr,en;q=0.8",
]
_GN_LANG = {
    "tr": {"hl": "tr", "gl": "TR", "ceid": "TR:tr"},
    "en": {"hl": "en", "gl": "US", "ceid": "US:en"},
}
_GN_WHEN = {
    "d": "when:1d",
    "w": "when:7d",
    "m": "when:30d",
    "y": "when:1y",
}
_DDGS_TIMELIMIT = {"d": "d", "w": "w", "m": "m", "y": "y"}
_DDGS_REGION = {"tr": "tr-tr", "en": "us-en"}
_proxy_index = 0
_proxy_cache: list[str] | None = None
_proxy_cache_mtime: float | None = None
_FETCH_RETRYABLE_STATUS_CODES = {401, 403, 408, 425, 429, 500, 502, 503, 504}
_THIN_CONTENT_MIN_CHARS = 80
_HTML_NOISE_TAGS = (
    "script",
    "style",
    "nav",
    "footer",
    "aside",
    "iframe",
    "form",
    "button",
    "input",
    "select",
    "textarea",
    "svg",
    "canvas",
    "header",
)
_ZERO_WIDTH_TRANSLATION = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)


def _build_browser_headers(
    accept: str = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    url: str | None = None,
    relaxed: bool = False,
) -> dict:
    ua = random.choice(_BROWSER_UAS)
    is_firefox = "Firefox" in ua
    headers: dict = {
        "User-Agent": ua,
        "Accept": accept,
        "Accept-Language": random.choice(_ACCEPT_LANGS),
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
        "DNT": "1",
    }
    if url:
        parsed = urlparse(url)
        if parsed.scheme and parsed.netloc:
            headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"
    if not is_firefox:
        headers["Sec-CH-UA"] = random.choice(_CHROME_SEC_UA)
        headers["Sec-CH-UA-Mobile"] = "?1" if "Mobile" in ua else "?0"
        headers["Sec-CH-UA-Platform"] = (
            '"Android"'
            if "Android" in ua
            else '"macOS"'
            if "Macintosh" in ua
            else '"Linux"'
            if "Linux" in ua
            else '"Windows"'
        )
        headers["Sec-Fetch-Dest"] = "document"
        headers["Sec-Fetch-Mode"] = "navigate"
        headers["Sec-Fetch-Site"] = "none"
        headers["Sec-Fetch-User"] = "?1"
    if relaxed:
        headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        headers["Cache-Control"] = "no-cache"
        headers.pop("Upgrade-Insecure-Requests", None)
    return headers


def _iter_fetch_header_variants(url: str):
    yield _build_browser_headers(url=url)
    yield _build_browser_headers(url=url, relaxed=True)


def load_proxies() -> list[str]:
    global _proxy_cache, _proxy_cache_mtime

    try:
        current_mtime = os.path.getmtime(PROXIES_PATH)
    except OSError:
        _proxy_cache = []
        _proxy_cache_mtime = None
        return []

    if _proxy_cache is not None and _proxy_cache_mtime == current_mtime:
        return list(_proxy_cache)

    proxies = []
    with open(PROXIES_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            proxy = line.strip()
            if not proxy or proxy.startswith("#"):
                continue
            parsed = urlparse(proxy)
            if parsed.scheme in {"http", "https", "socks5", "socks5h"} and parsed.hostname and parsed.port:
                proxies.append(proxy)

    _proxy_cache = proxies
    _proxy_cache_mtime = current_mtime
    return list(proxies)


def get_proxy_candidates(include_direct_fallback: bool = False) -> list[str | None]:
    global _proxy_index
    proxies = load_proxies()
    if not proxies:
        return [None]

    start = _proxy_index % len(proxies)
    _proxy_index = (_proxy_index + 1) % len(proxies)
    ordered = proxies[start:] + proxies[:start]
    if include_direct_fallback:
        ordered.append(None)
    return ordered


def _requests_proxy_dict(proxy: str | None):
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


def _is_safe_url(url: str) -> tuple[bool, str]:
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL"
    if parsed.scheme not in ("http", "https"):
        return False, "Only http and https are supported"
    hostname = parsed.hostname or ""
    if not hostname:
        return False, "Hostname not found"
    if hostname.lower() in ("localhost", "localhost."):
        return False, "Local addresses are prohibited"
    try:
        for info in socket.getaddrinfo(hostname, None):
            addr = info[4][0]
            ip = ipaddress.ip_address(addr)
            for net in PRIVATE_NETWORKS:
                if ip in net:
                    return False, f"Private/local network address prohibited: {addr}"
    except socket.gaierror:
        return False, f"DNS resolution failed: {hostname}"
    return True, ""


def _clean_extracted_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", str(text or ""))
    cleaned = cleaned.translate(_ZERO_WIDTH_TRANSLATION)
    cleaned = cleaned.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    normalized_lines = []
    for line in cleaned.split("\n"):
        stripped = re.sub(r"\s+", " ", line).strip()
        if stripped and re.fullmatch(r"[-_=|~•·*.]{3,}", stripped):
            continue
        normalized_lines.append(stripped)

    cleaned = "\n".join(normalized_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _truncate_content(text: str) -> str:
    if len(text) <= CONTENT_MAX_CHARS:
        return text
    return text[:CONTENT_MAX_CHARS].rstrip() + "\n[Content truncated]"


def _extract_meta_content(soup: BeautifulSoup, *selectors: tuple[str, str]) -> str:
    for attr, value in selectors:
        tag = soup.find("meta", attrs={attr: value})
        content = (tag.get("content") or "").strip() if tag else ""
        if content:
            return content
    return ""


def _collect_structured_text(value, parts: list[str], limit: int = 6):
    if len(parts) >= limit or value is None:
        return
    if isinstance(value, str):
        cleaned = _clean_extracted_text(value)
        if cleaned and cleaned not in parts:
            parts.append(cleaned)
        return
    if isinstance(value, list):
        for item in value:
            if len(parts) >= limit:
                break
            _collect_structured_text(item, parts, limit=limit)
        return
    if not isinstance(value, dict):
        return

    for key in ("headline", "name", "description", "articleBody", "text"):
        if len(parts) >= limit:
            break
        _collect_structured_text(value.get(key), parts, limit=limit)

    if len(parts) < limit:
        main_entity = value.get("mainEntity") or value.get("mainEntityOfPage")
        _collect_structured_text(main_entity, parts, limit=limit)


def _extract_json_ld_text(soup: BeautifulSoup) -> str:
    parts: list[str] = []
    for tag in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        raw = (tag.string or tag.get_text() or "").strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        _collect_structured_text(parsed, parts)
        if len(parts) >= 6:
            break
    return "\n\n".join(parts[:6])


def _combine_distinct_text_blocks(blocks: list[str]) -> str:
    combined: list[str] = []
    seen = set()
    for block in blocks:
        cleaned = _clean_extracted_text(block)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        combined.append(cleaned)
    return "\n\n".join(combined)


def _extract_html(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("title") or ""
    title = title.get_text(strip=True) if title else ""
    if not title:
        title = _extract_meta_content(soup, ("property", "og:title"), ("name", "twitter:title"))

    noscript_text = _combine_distinct_text_blocks([tag.get_text(separator="\n") for tag in soup.find_all("noscript")])
    meta_description = _extract_meta_content(
        soup,
        ("name", "description"),
        ("property", "og:description"),
        ("name", "twitter:description"),
    )
    structured_text = _extract_json_ld_text(soup)

    for tag in soup(_HTML_NOISE_TAGS):
        tag.decompose()

    content_root = soup.find("main") or soup.find("article") or soup.body or soup
    primary_text = _clean_extracted_text(content_root.get_text(separator="\n"))
    text = primary_text
    if len(primary_text) < _THIN_CONTENT_MIN_CHARS:
        text = _combine_distinct_text_blocks([primary_text, noscript_text, meta_description, structured_text])
    if not text:
        text = _combine_distinct_text_blocks([meta_description, structured_text, noscript_text])
    return {"url": url, "title": title, "content": _truncate_content(text), "content_format": "html"}


def _extract_pdf(data: bytes, url: str) -> dict:
    try:
        reader = PdfReader(BytesIO(data))
        pages = []
        for index, page in enumerate(reader.pages):
            if index >= 50:
                pages.append("[More pages available, truncated]")
                break
            pages.append(page.extract_text() or "")
        text = _clean_extracted_text("\n\n".join(page for page in pages if page.strip()))
        return {
            "url": url,
            "title": f"PDF: {url.rstrip('/').split('/')[-1]}",
            "content": _truncate_content(text),
            "content_format": "pdf",
        }
    except Exception as exc:
        return {"url": url, "title": "", "content": "", "error": f"Could not read PDF: {exc}"}


def _extract_json_text(resp, raw: bytes, url: str) -> dict:
    try:
        parsed = resp.json()
        text = json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        text = raw.decode("utf-8", errors="replace")
    return {"url": url, "title": "", "content": _truncate_content(_clean_extracted_text(text)), "content_format": "json"}


def _extract_xml_text(raw: bytes, url: str) -> dict:
    decoded = raw.decode("utf-8", errors="replace")
    try:
        root = ET.fromstring(decoded)
    except Exception:
        text = decoded
    else:
        text_fragments = []
        for element in root.iter():
            value = (element.text or "").strip()
            if not value:
                continue
            label = re.sub(r"\s+", " ", str(element.tag or "")).strip()
            text_fragments.append(f"{label}: {value}" if label else value)
        text = "\n".join(text_fragments) or decoded
    return {"url": url, "title": "", "content": _truncate_content(_clean_extracted_text(text)), "content_format": "xml"}


def _extract_plain_text(raw: bytes, url: str, content_format: str = "text") -> dict:
    text = raw.decode("utf-8", errors="replace")
    return {
        "url": url,
        "title": "",
        "content": _truncate_content(_clean_extracted_text(text)),
        "content_format": content_format,
    }


def _build_fetch_result_from_response(resp, raw: bytes, url: str, partial_error: str | None = None) -> dict:
    ct = resp.headers.get("Content-Type", "").lower()
    final_url = resp.url

    if "pdf" in ct or url.lower().endswith(".pdf"):
        result = _extract_pdf(raw, final_url)
    elif "json" in ct:
        result = _extract_json_text(resp, raw, final_url)
    elif "xml" in ct and "html" not in ct:
        result = _extract_xml_text(raw, final_url)
    elif "text/plain" in ct:
        result = _extract_plain_text(raw, final_url)
    else:
        enc = resp.encoding or "utf-8"
        result = _extract_html(raw.decode(enc, errors="replace"), final_url)

    result["cleanup_applied"] = True
    result["status"] = resp.status_code
    if partial_error:
        result["fetch_warning"] = partial_error
        result["partial_content"] = True
    return result


def _has_useful_fetch_content(result: dict) -> bool:
    return len(_clean_extracted_text(result.get("content") or "")) >= _THIN_CONTENT_MIN_CHARS


def _append_fetch_warning(result: dict, warning: str):
    existing = (result.get("fetch_warning") or "").strip()
    if existing:
        if warning not in existing:
            result["fetch_warning"] = f"{existing}; {warning}"
        return
    result["fetch_warning"] = warning


def _should_retry_without_ssl_verification(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return "certificate verify failed" in text or "sslcertverificationerror" in text


def fetch_url_tool(url: str) -> dict:
    safe, reason = _is_safe_url(url)
    if not safe:
        return {"url": url, "error": reason, "content": ""}

    cache_key = f"fetch:{hashlib.md5(url.encode()).hexdigest()}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    last_error = None
    best_result = None
    header_variants = list(_iter_fetch_header_variants(url))
    for proxy in get_proxy_candidates(include_direct_fallback=True):
        for index, headers in enumerate(header_variants):
            session = None
            try:
                session = http_requests.Session()
                session.max_redirects = FETCH_MAX_REDIRECTS
                session.trust_env = False
                proxy_map = _requests_proxy_dict(proxy)
                if proxy_map:
                    session.proxies.update(proxy_map)
                bypassed_ssl_verification = False
                try:
                    resp = session.get(
                        url,
                        timeout=FETCH_TIMEOUT,
                        headers=headers,
                        stream=True,
                        allow_redirects=True,
                    )
                except http_requests.exceptions.SSLError as exc:
                    if not _should_retry_without_ssl_verification(exc):
                        raise
                    bypassed_ssl_verification = True
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", InsecureRequestWarning)
                        resp = session.get(
                            url,
                            timeout=FETCH_TIMEOUT,
                            headers=headers,
                            stream=True,
                            allow_redirects=True,
                            verify=False,
                        )
                raw = b""
                partial_error = None
                try:
                    for chunk in resp.iter_content(chunk_size=8192):
                        raw += chunk
                        if len(raw) >= FETCH_MAX_SIZE:
                            raw = raw[:FETCH_MAX_SIZE]
                            break
                except (http_requests.exceptions.ChunkedEncodingError, http_requests.exceptions.ConnectionError) as exc:
                    if raw:
                        partial_error = f"Connection ended early; partial page content was recovered ({exc})"
                    else:
                        raise

                result = _build_fetch_result_from_response(resp, raw, url, partial_error=partial_error)
                if bypassed_ssl_verification:
                    result["ssl_verification_bypassed"] = True
                    _append_fetch_warning(
                        result,
                        "SSL certificate verification failed; retried without certificate verification",
                    )
                if resp.status_code >= 400:
                    _append_fetch_warning(result, f"HTTP {resp.status_code} returned by origin")
                    if resp.status_code in _FETCH_RETRYABLE_STATUS_CODES and not _has_useful_fetch_content(result):
                        best_result = best_result or result
                        last_error = f"HTTP {resp.status_code}"
                        if index + 1 < len(header_variants):
                            continue
                        break

                if not result.get("content"):
                    best_result = best_result or result
                    last_error = last_error or "Fetched page returned no extractable content"
                    if index + 1 < len(header_variants):
                        continue
                    break

                if result.get("partial_content"):
                    cache_set(cache_key, result)
                    return result

                if not _has_useful_fetch_content(result):
                    _append_fetch_warning(result, "Only limited extractable content was found on the page")
                    best_result = result
                    if result.get("content_format") == "html" and index + 1 < len(header_variants):
                        continue
                    cache_set(cache_key, result)
                    return result

                cache_set(cache_key, result)
                return result
            except http_requests.exceptions.TooManyRedirects:
                last_error = "Too many redirects"
                break
            except http_requests.exceptions.Timeout:
                last_error = "Request timed out (20s)"
                break
            except Exception as exc:
                last_error = str(exc)
                break
            finally:
                if session is not None:
                    session.close()

    if best_result is not None and best_result.get("content"):
        return best_result

    return {"url": url, "error": last_error or "Could not fetch URL", "content": ""}


def search_web_tool(queries: list) -> list:
    if not queries:
        return []

    results = []
    seen_urls = set()

    for raw_query in queries[:5]:
        query = str(raw_query).strip()
        if not query:
            continue

        cache_key = f"search:{hashlib.md5(query.lower().encode()).hexdigest()}"
        cached = cache_get(cache_key)
        if cached is not None:
            for row in cached:
                if row.get("url") not in seen_urls:
                    seen_urls.add(row.get("url"))
                    results.append(row)
            continue

        try:
            hits = None
            last_error = None
            for proxy in get_proxy_candidates(include_direct_fallback=True):
                try:
                    with DDGS(proxy=proxy) as ddgs:
                        hits = list(ddgs.text(query, max_results=SEARCH_MAX_RESULTS))
                    break
                except Exception as exc:
                    last_error = exc
            if hits is None:
                raise last_error or RuntimeError("Search failed")
            normalized = [
                {
                    "title": hit.get("title", ""),
                    "url": hit.get("href", ""),
                    "snippet": hit.get("body", ""),
                }
                for hit in hits
            ]
            cache_set(cache_key, normalized)
            for row in normalized:
                if row["url"] not in seen_urls:
                    seen_urls.add(row["url"])
                    results.append(row)
        except Exception as exc:
            results.append({"error": str(exc), "query": query})

    return results


def search_news_ddgs_tool(queries: list, lang: str = "tr", when: str | None = None) -> list:
    if not queries:
        return []

    region = _DDGS_REGION.get(lang, "tr-tr")
    timelimit = _DDGS_TIMELIMIT.get(when) if when else None
    results = []
    seen_urls = set()

    for raw_query in queries[:5]:
        query = str(raw_query).strip()
        if not query:
            continue

        cache_key = f"news_ddgs:{hashlib.md5((query + lang + (when or '')).lower().encode()).hexdigest()}"
        cached = cache_get(cache_key)
        if cached is not None:
            for row in cached:
                if row.get("link") not in seen_urls:
                    seen_urls.add(row["link"])
                    results.append(row)
            continue

        try:
            hits = None
            last_error = None
            for proxy in get_proxy_candidates(include_direct_fallback=True):
                try:
                    with DDGS(proxy=proxy) as ddgs:
                        hits = list(
                            ddgs.news(
                                query,
                                region=region,
                                safesearch="off",
                                timelimit=timelimit,
                                max_results=SEARCH_MAX_RESULTS,
                            )
                        )
                    break
                except Exception as exc:
                    last_error = exc
            if hits is None:
                raise last_error or RuntimeError("News search failed")
            normalized = [
                {
                    "title": hit.get("title", ""),
                    "link": hit.get("url", ""),
                    "time": hit.get("date", ""),
                    "source": hit.get("source", ""),
                }
                for hit in hits
            ]
            cache_set(cache_key, normalized)
            for row in normalized:
                if row["link"] not in seen_urls:
                    seen_urls.add(row["link"])
                    results.append(row)
        except Exception as exc:
            results.append({"error": str(exc), "query": query})

    return results


def search_news_google_tool(queries: list, lang: str = "tr", when: str | None = None) -> list:
    if not queries:
        return []

    geo = _GN_LANG.get(lang, _GN_LANG["tr"])
    results = []
    seen_urls = set()

    for raw_query in queries[:5]:
        query = str(raw_query).strip()
        if not query:
            continue

        full_query = f"{query} {_GN_WHEN[when]}" if when and when in _GN_WHEN else query
        cache_key = f"news_google:{hashlib.md5((full_query + lang).lower().encode()).hexdigest()}"
        cached = cache_get(cache_key)
        if cached is not None:
            for row in cached:
                if row.get("link") not in seen_urls:
                    seen_urls.add(row["link"])
                    results.append(row)
            continue

        rss_url = (
            f"https://news.google.com/rss/search"
            f"?q={url_quote(full_query)}"
            f"&hl={geo['hl']}&gl={geo['gl']}&ceid={geo['ceid']}"
        )
        try:
            resp = None
            last_error = None
            for proxy in get_proxy_candidates(include_direct_fallback=True):
                try:
                    resp = http_requests.get(
                        rss_url,
                        headers=_build_browser_headers(
                            accept="application/rss+xml, application/xml, text/xml, */*;q=0.8"
                        ),
                        timeout=15,
                        proxies=_requests_proxy_dict(proxy),
                    )
                    resp.raise_for_status()
                    break
                except Exception as exc:
                    last_error = exc
                    resp = None
            if resp is None:
                raise last_error or RuntimeError("Could not fetch Google News RSS")
            root = ET.fromstring(resp.content)
            items = root.findall(".//item")[:SEARCH_MAX_RESULTS]

            normalized = []
            for item in items:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub = (item.findtext("pubDate") or "").strip()
                source = ""
                source_element = item.find("source")
                if source_element is not None:
                    source = (source_element.text or "").strip()
                if source and title.endswith(f" - {source}"):
                    title = title[: -(len(source) + 3)]
                if link:
                    normalized.append({"title": title, "link": link, "time": pub, "source": source})

            cache_set(cache_key, normalized)
            for row in normalized:
                if row["link"] not in seen_urls:
                    seen_urls.add(row["link"])
                    results.append(row)
        except Exception as exc:
            results.append({"error": str(exc), "query": raw_query})

    return results
