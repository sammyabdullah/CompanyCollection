#!/usr/bin/env python3
"""
Extract tech company links (and optionally CEO names) from web pages.

Usage:
    python extract_companies.py page1.html page2.html --output companies.csv
    python extract_companies.py https://example.com/portfolio --output companies.csv
    python extract_companies.py page.html --no-ceo --output companies.csv

Outputs a CSV with columns:
    company_name, company_url, ceo_first_name, ceo_last_name, source_url

Requires: ANTHROPIC_API_KEY environment variable
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from urllib.parse import urljoin

import anthropic
from bs4 import BeautifulSoup


def _playwright_proxy_kwargs() -> dict:
    """Build Playwright proxy kwargs from HTTP_PROXY / HTTPS_PROXY env vars, if set."""
    from urllib.parse import urlparse

    raw = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or ""
    if not raw:
        return {}
    parsed = urlparse(raw)
    proxy: dict = {"server": f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"}
    if parsed.username:
        proxy["username"] = parsed.username
    if parsed.password:
        proxy["password"] = parsed.password
    return {"proxy": proxy}


def _fetch_with_playwright(url: str, fast: bool = False) -> str:
    """Render a URL with a headless Chromium browser and return HTML.

    fast=True: quick load for individual company pages (domcontentloaded, no scrolling).
    fast=False: full load for portfolio listing pages (networkidle + scroll passes).
    """
    from playwright.sync_api import sync_playwright

    proxy_kwargs = _playwright_proxy_kwargs()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **proxy_kwargs)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = context.new_page()

        if fast:
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
        else:
            page.goto(url, wait_until="networkidle", timeout=45000)

            # Scroll incrementally to trigger lazy-loading on portfolio pages
            for _ in range(20):
                page.evaluate("window.scrollBy(0, window.innerHeight)")
                page.wait_for_timeout(600)

            # Second pass: scroll back to top then all the way down to catch any remaining lazy loads
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(500)
            for _ in range(20):
                page.evaluate("window.scrollBy(0, window.innerHeight)")
                page.wait_for_timeout(400)

            # Final wait for any network activity triggered by scrolling
            try:
                page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass

        html = page.content()
        browser.close()
    return html


def _looks_empty(html: str) -> bool:
    """Return True if the page appears to be an unrendered JS shell."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return len(text) < 200


def fetch_page(source: str, fast: bool = False) -> tuple[str, str]:
    """Return (html_content, base_url) for a local file or URL.

    Local files are read directly. Web URLs always use Playwright.
    Pass fast=True for individual company pages (founder/detail lookups).
    """
    if os.path.exists(source):
        with open(source, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        base_url = "file://" + os.path.abspath(source)
        return content, base_url

    print("  Fetching with Playwright...", file=sys.stderr)
    html = _fetch_with_playwright(source, fast=fast)
    return html, source


def page_text_and_links(html: str, base_url: str) -> tuple[str, list[dict]]:
    """Extract readable text and links from HTML."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        full_url = urljoin(base_url, href)
        if full_url in seen:
            continue
        seen.add(full_url)
        links.append({"url": full_url, "text": a.get_text(strip=True)})

    return text, links


def extract_json(text: str, array: bool = True) -> list | dict | None:
    """Pull the first JSON array or object out of a Claude response."""
    text = text.strip()
    # Try direct parse first (Claude returned clean JSON)
    try:
        result = json.loads(text)
        if array and isinstance(result, list):
            return result
        if not array and isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    # Greedy match to capture the full outermost array/object
    pattern = r"\[.*\]" if array else r"\{.*\}"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return [] if array else {}


def _find_external_company_url(
    detail_html: str, detail_url: str, company_name: str, client: anthropic.Anthropic
) -> str:
    """Ask Claude to identify the company's own website URL from its VC detail page."""
    _, links = page_text_and_links(detail_html, detail_url)
    soup = BeautifulSoup(detail_html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    page_text = soup.get_text(separator="\n", strip=True)

    prompt = f"""This is a VC investor's detail page for the portfolio company "{company_name}".

Page text:
{page_text[:3000]}

Links on this page:
{json.dumps(links[:100], indent=2)}

What is the company's own external website URL (e.g. https://stripe.com)?
Return ONLY a JSON object: {{"url": "https://..."}}
If you cannot determine it, return {{"url": ""}}"""

    response = call_claude_with_retry(
        client,
        model="claude-opus-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = next((b.text for b in response.content if b.type == "text"), "{}")
    data = extract_json(raw, array=False)
    return data.get("url", "") if isinstance(data, dict) else ""


def _resolve_detail_pages(
    detail_links: list[dict], base_host: str, source_label: str, client: anthropic.Anthropic
) -> list[dict]:
    """Follow each internal company-detail link and scrape the real company URL from it."""
    results = []
    for item in detail_links:
        detail_url = item["detail_url"]
        company_name = item["company_name"]
        try:
            html, _ = fetch_page(detail_url, fast=True)
            ext_url = _find_external_company_url(html, detail_url, company_name, client)
            if ext_url:
                results.append({"company_name": company_name, "company_url": ext_url})
                print(f"    {company_name} -> {ext_url}", file=sys.stderr)
            else:
                print(f"    {company_name}: no external URL found on detail page", file=sys.stderr)
        except Exception as e:
            print(f"    Could not load detail page for {company_name}: {e}", file=sys.stderr)
    return results


def _resolve_name_only(
    name_only: list[dict], client: anthropic.Anthropic
) -> list[dict]:
    """Ask Claude to supply website URLs for companies identified by name only (no links on page)."""
    names = [c["company_name"] for c in name_only]
    prompt = f"""For each of the following tech companies, provide their primary website URL.

Companies:
{json.dumps(names, indent=2)}

Return ONLY a JSON array. Each element must have:
  - "company_name": exactly as given above
  - "company_url": the company's homepage URL (e.g. "https://stripe.com")

If you don't know the URL for a company, omit it from the array.
Respond with the JSON array only — no explanation, no markdown fences."""

    response = call_claude_with_retry(
        client,
        model="claude-opus-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = next((b.text for b in response.content if b.type == "text"), "[]")
    result = extract_json(raw, array=True)
    if not isinstance(result, list):
        return []
    resolved = [c for c in result if c.get("company_name") and c.get("company_url")]
    for c in resolved:
        print(f"    {c['company_name']} -> {c['company_url']}", file=sys.stderr)
    return resolved


def identify_tech_companies(
    html: str, base_url: str, source_label: str, client: anthropic.Anthropic,
    debug: bool = False,
) -> list[dict]:
    """Ask Claude which links on this page lead to tech companies."""
    from urllib.parse import urlparse

    page_text, links = page_text_and_links(html, base_url)

    print(f"  Page: {len(html)} chars HTML, {len(page_text)} chars text, {len(links)} links", file=sys.stderr)

    if debug:
        print(f"[DEBUG] Text preview:\n{page_text[:500]}\n", file=sys.stderr)
        print(f"[DEBUG] First 30 links:", file=sys.stderr)
        for lnk in links[:30]:
            print(f"  {lnk}", file=sys.stderr)
        if len(links) > 30:
            print(f"  ... and {len(links) - 30} more", file=sys.stderr)

    if not links:
        print(f"  No links found in {source_label}; skipping.", file=sys.stderr)
        return []

    base_host = urlparse(base_url).netloc

    prompt = f"""You are a research assistant. I have a web page that lists tech companies.

Page text (truncated):
{page_text[:6000]}

All links found on the page (up to 300):
{json.dumps(links[:300], indent=2)}

Task: Identify every tech company or startup listed or mentioned on this page as a subject of interest
(e.g. portfolio companies, investees, featured startups, ranked companies, profiled businesses, etc.).

For each company, determine which kind of link is available:
- TYPE A: a direct link to the company's own external website (e.g. https://stripe.com)
- TYPE B: an internal link to a company-detail page on THIS same site (e.g. /companies/stripe or /rebels/ribbit)
- TYPE C: company name found in the page text but no usable link of either type

Return ONLY a JSON array. Each element must have:
  - "company_name": the company's name (string)
  - "company_url": for TYPE A, the company's own website URL; for TYPE B and C, leave as ""
  - "detail_url": for TYPE B, the full URL of the internal detail page; for TYPE A and C, leave as ""

Rules:
- Do not include the site/publication itself, nav links, author profiles, social media accounts, or unrelated ads.
- If a company has both TYPE A and TYPE B, prefer TYPE A.
- Skip duplicates.
- If no companies are found, return [].

Respond with the JSON array only — no explanation, no markdown fences."""

    response = call_claude_with_retry(
        client,
        model="claude-opus-4-6",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = next((b.text for b in response.content if b.type == "text"), "[]")
    if not raw.strip() or raw.strip() == "[]":
        print(f"  Warning: Claude returned empty response for {source_label}", file=sys.stderr)
    if debug:
        print(f"\n[DEBUG] Claude raw response:\n{raw[:2000]}\n", file=sys.stderr)
    result = extract_json(raw, array=True)
    if not isinstance(result, list):
        return []

    # Split into direct hits, detail pages needing follow-up, and name-only (TYPE C)
    direct = [c for c in result if c.get("company_url")]
    needs_detail = [c for c in result if not c.get("company_url") and c.get("detail_url")]
    name_only = [c for c in result if not c.get("company_url") and not c.get("detail_url") and c.get("company_name")]
    if debug:
        print(f"[DEBUG] Claude identified {len(direct)} direct URLs, {len(needs_detail)} detail pages, {len(name_only)} name-only", file=sys.stderr)

    if needs_detail:
        print(f"  Following {len(needs_detail)} company detail pages...", file=sys.stderr)
        resolved = _resolve_detail_pages(needs_detail, base_host, source_label, client)
        direct.extend(resolved)

    if name_only:
        print(f"  Looking up URLs for {len(name_only)} name-only companies via Claude...", file=sys.stderr)
        resolved = _resolve_name_only(name_only, client)
        direct.extend(resolved)

    # Normalise: drop detail_url key from output
    for c in direct:
        c.pop("detail_url", None)

    return direct


def call_claude_with_retry(client: anthropic.Anthropic, max_retries: int = 4, **kwargs) -> anthropic.types.Message:
    """Call client.messages.create with exponential backoff on timeout/connection/rate-limit errors."""
    delays = [2, 4, 8, 16]
    rate_limit_delays = [60, 90, 120, 180]
    for attempt in range(max_retries + 1):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            if attempt == max_retries:
                raise
            wait = rate_limit_delays[attempt]
            print(f"    Rate limit hit, waiting {wait}s before retry...", file=sys.stderr)
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if attempt == max_retries:
                raise
            if e.status_code == 529:
                wait = rate_limit_delays[attempt]
                print(f"    API overloaded (529), waiting {wait}s before retry...", file=sys.stderr)
            else:
                wait = delays[attempt]
                print(f"    API error ({e.__class__.__name__}), retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as e:
            if attempt == max_retries:
                raise
            wait = delays[attempt]
            print(f"    API error ({e.__class__.__name__}), retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)


_GOOGLE_BLOCKED_SIGNALS = [
    "unusual traffic",
    "detected unusual",
    "captcha",
    "are you a robot",
    "verify you're a human",
    "access denied",
]

_ABOUT_LINK_PATTERN = re.compile(
    r"\b(about|team|leadership|people|founders?|executives?|management|staff|who we are)\b",
    re.IGNORECASE,
)


def _fetch_text(url: str) -> str:
    """Fetch a URL with Playwright and return plain text."""
    html = _fetch_with_playwright(url, fast=True)
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _is_blocked(text: str) -> bool:
    lower = text.lower()
    return any(signal in lower for signal in _GOOGLE_BLOCKED_SIGNALS)


def _google_result_urls(query: str, max_results: int = 3) -> list[str]:
    """Search Google, return up to max_results non-Google organic result URLs."""
    search_html = _fetch_with_playwright(
        f"https://www.google.com/search?q={query}&num=5", fast=True
    )
    soup = BeautifulSoup(search_html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    page_text = soup.get_text()
    if _is_blocked(page_text):
        print(f"    Google blocked for query: {query[:60]}", file=sys.stderr)
        return []

    _SKIP_DOMAINS = {"google.com", "google.", "youtube.com", "accounts.google"}

    urls, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = a["href"]

        # Playwright-rendered Google: links are direct URLs
        if href.startswith("http") and not any(d in href for d in _SKIP_DOMAINS):
            candidate = href.split("&")[0]
        # Static/cached Google HTML: links wrapped as /url?q=<actual>
        elif href.startswith("/url?q="):
            candidate = href[7:].split("&")[0]
            if not candidate.startswith("http") or any(d in candidate for d in _SKIP_DOMAINS):
                continue
        else:
            continue

        if candidate not in seen:
            seen.add(candidate)
            urls.append(candidate)
            if len(urls) >= max_results:
                break

    print(f"    Google returned {len(urls)} URLs for: {query[:60]}", file=sys.stderr)
    return urls


_FALLBACK_SUBPAGES = ["/about", "/team", "/leadership", "/people", "/about-us"]


def _find_about_links(company_url: str) -> list[str]:
    """
    Fetch the company homepage and extract links that look like About/Team pages.
    Falls back to common subpaths if the homepage can't be loaded or no links found.
    """
    base = company_url.rstrip("/")
    fallback = [base + p for p in _FALLBACK_SUBPAGES]

    try:
        # Use fast=False so JS-rendered nav menus have time to appear
        html = _fetch_with_playwright(company_url, fast=False)
    except Exception:
        return fallback

    soup = BeautifulSoup(html, "lxml")
    found, seen = [], set()

    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        if _ABOUT_LINK_PATTERN.search(text) or _ABOUT_LINK_PATTERN.search(href):
            full = urljoin(company_url, href)
            if full not in seen and full != company_url:
                seen.add(full)
                found.append(full)

    if not found:
        print(f"    No About/Team links found on homepage, trying fallback paths", file=sys.stderr)
        return fallback

    print(f"    Found {len(found)} About/Team links on homepage", file=sys.stderr)
    return found[:8]


def _extract_ceo(text: str, domain: str, client: anthropic.Anthropic) -> tuple[str, str]:
    """Ask Claude to find the CEO name in text. Returns ('', '') if not found."""
    prompt = f"""Below is text scraped from a single webpage. Read it carefully.

{text[:5000]}

Does this text explicitly name a CEO or Chief Executive Officer?
- If YES: return their name exactly as written on the page.
- If NO: return empty strings. Do NOT guess, infer, or use outside knowledge.

Return ONLY: {{"first_name": "...", "last_name": "..."}}"""

    response = call_claude_with_retry(
        client,
        model="claude-opus-4-6",
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = next((b.text for b in response.content if b.type == "text"), "{}")
    data = extract_json(raw, array=False)
    if isinstance(data, dict):
        return data.get("first_name", ""), data.get("last_name", "")
    return "", ""


def get_ceo_info(
    company_url: str, company_name: str, client: anthropic.Anthropic
) -> tuple[str, str]:
    """
    Find the CEO:
    1. Google '<domain> CEO' → try first 3 organic results
    2. Company website: crawl homepage for About/Team links, follow them
    3. Google '<company name> CEO' on Crunchbase, Wikipedia, or news
    """
    domain = re.sub(r"^https?://(www\.)?", "", company_url).split("/")[0]
    time.sleep(2)

    # ── Step 1: Google → first 3 search result pages ───────────────────────
    try:
        result_urls = _google_result_urls(f"{domain} CEO", max_results=3)
    except Exception as e:
        print(f"    Google search error for {domain}: {e}", file=sys.stderr)
        result_urls = []

    for url in result_urls:
        try:
            print(f"    [Step 1] Checking: {url}", file=sys.stderr)
            text = _fetch_text(url)
            first, last = _extract_ceo(text, domain, client)
            if first or last:
                print(f"    CEO found (Step 1): {first} {last}", file=sys.stderr)
                return first, last
        except Exception as e:
            print(f"    Could not fetch {url}: {e}", file=sys.stderr)

    # ── Step 2: Company website About/Team pages ────────────────────────────
    print(f"    [Step 2] Crawling company website for {domain}...", file=sys.stderr)
    about_urls = _find_about_links(company_url)
    for url in about_urls:
        try:
            text = _fetch_text(url)
            first, last = _extract_ceo(text, domain, client)
            if first or last:
                print(f"    CEO found (Step 2) at {url}: {first} {last}", file=sys.stderr)
                return first, last
        except Exception:
            continue

    # ── Step 3: Crunchbase via Google ──────────────────────────────────────
    print(f"    [Step 3] Trying Crunchbase for {domain}...", file=sys.stderr)
    fallback_queries = [
        f'"{company_name}" CEO site:crunchbase.com',
    ]
    for query in fallback_queries:
        try:
            urls = _google_result_urls(query, max_results=2)
            for url in urls:
                try:
                    text = _fetch_text(url)
                    first, last = _extract_ceo(text, domain, client)
                    if first or last:
                        print(f"    CEO found (Step 3) at {url}: {first} {last}", file=sys.stderr)
                        return first, last
                except Exception:
                    continue
        except Exception:
            continue

    print(f"    No CEO found for {domain}", file=sys.stderr)
    return "", ""




def clean_url(url: str) -> str:
    """Strip protocol/www and return only the domain (stop at first '/')."""
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)
    return url.split("/")[0]


def deduplicate(companies: list[dict]) -> list[dict]:
    """Remove duplicate company URLs, keeping the first occurrence."""
    seen_urls = set()
    seen_names = set()
    unique = []
    for c in companies:
        url = c.get("company_url", "").rstrip("/").lower()
        name = c.get("company_name", "").lower().strip()
        if url and url not in seen_urls and name not in seen_names:
            seen_urls.add(url)
            seen_names.add(name)
            unique.append(c)
    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Extract tech company links (and CEO names) from web pages."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="HTML files or URLs containing tech company links.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="companies.csv",
        help="Output CSV file path (default: companies.csv).",
    )
    parser.add_argument(
        "--no-ceo",
        action="store_true",
        help="Skip CEO lookup (faster, columns will be empty).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print fetched HTML length, extracted links, and Claude responses.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    all_companies: list[dict] = []
    source_counts: list[tuple[str, int]] = []

    for source in args.sources:
        print(f"\nProcessing: {source}")
        try:
            html, base_url = fetch_page(source)
        except Exception as e:
            print(f"  Failed to load {source}: {e}", file=sys.stderr)
            source_counts.append((source, 0))
            continue

        print("  Identifying tech companies via Claude...")
        companies = identify_tech_companies(html, base_url, source, client, debug=args.debug)
        print(f"  Found {len(companies)} tech companies.")
        source_counts.append((source, len(companies)))
        for c in companies:
            c["source_url"] = clean_url(source)
        all_companies.extend(companies)

    all_companies = deduplicate(all_companies)
    print(f"\nTotal unique companies: {len(all_companies)}")

    if not all_companies:
        print("No tech companies found. Exiting.")
        sys.exit(0)

    fieldnames = ["company_name", "company_url", "ceo_first_name", "ceo_last_name", "source_url"]

    # Load checkpoint: any rows already written to the output CSV
    completed_urls: set[str] = set()
    if os.path.exists(args.output):
        with open(args.output, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_urls.add(row.get("company_url", "").lower())
        if completed_urls:
            print(f"Resuming: {len(completed_urls)} companies already done, skipping them.")

    # Open output in append mode so we resume where we left off
    write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
    out_f = open(args.output, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        out_f.flush()

    try:
        for i, company in enumerate(all_companies, 1):
            name = company.get("company_name", "")
            url = company.get("company_url", "")
            url_key = clean_url(url).lower()

            if url_key in completed_urls:
                print(f"  [{i}/{len(all_companies)}] Skipping {name} (already done)")
                continue

            first, last = "", ""
            if not args.no_ceo and url:
                print(f"  [{i}/{len(all_companies)}] Looking up CEO for {name}...")
                first, last = get_ceo_info(url, name, client)

            row = {
                "company_name": name,
                "company_url": clean_url(url),
                "ceo_first_name": first,
                "ceo_last_name": last,
                "source_url": company.get("source_url", ""),
            }
            writer.writerow(row)
            out_f.flush()
            completed_urls.add(url_key)
    finally:
        out_f.close()

    print(f"\nDone. Results saved to: {args.output}")
    print(f"Columns: company_name, company_url, ceo_first_name, ceo_last_name, source_url")
    print(f"\n--- Summary ---")
    for source, count in source_counts:
        print(f"  {source}: {count} companies found")
    print(f"  Total unique: {len(all_companies)}")


if __name__ == "__main__":
    main()
