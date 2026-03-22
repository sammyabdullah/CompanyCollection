#!/usr/bin/env python3
"""
Extract tech company links (and optionally founder names) from web pages.

Usage:
    python extract_companies.py page1.html page2.html --output companies.csv
    python extract_companies.py https://example.com/portfolio --output companies.csv
    python extract_companies.py page.html --no-founders --output companies.csv

Outputs a CSV with columns:
    company_name, company_url, founder_first_name, founder_last_name, source_url

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
import requests
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


def _fetch_with_playwright(url: str) -> str:
    """Render a URL with a headless Chromium browser, scroll to load lazy content, return HTML."""
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


def fetch_page(source: str) -> tuple[str, str]:
    """Return (html_content, base_url) for a local file or URL.

    Local files are read directly. Web URLs always use Playwright so that
    JS-rendered portfolio pages load fully before we extract links.
    """
    if os.path.exists(source):
        with open(source, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        base_url = "file://" + os.path.abspath(source)
        return content, base_url

    print("  Fetching with Playwright...", file=sys.stderr)
    html = _fetch_with_playwright(source)
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
            html, _ = fetch_page(detail_url)
            ext_url = _find_external_company_url(html, detail_url, company_name, client)
            if ext_url:
                results.append({"company_name": company_name, "company_url": ext_url})
                print(f"    {company_name} -> {ext_url}", file=sys.stderr)
            else:
                print(f"    {company_name}: no external URL found on detail page", file=sys.stderr)
        except Exception as e:
            print(f"    Could not load detail page for {company_name}: {e}", file=sys.stderr)
    return results


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

Task: Identify every company listed on this page.

For each company, determine which kind of link is available:
- TYPE A: a direct link to the company's own external website (e.g. https://stripe.com)
- TYPE B: an internal link to a company-detail page on THIS same site (e.g. /companies/stripe or /rebels/ribbit)

Return ONLY a JSON array. Each element must have:
  - "company_name": the company's name (string)
  - "company_url": for TYPE A, the company's own website URL; for TYPE B, leave as ""
  - "detail_url": for TYPE B, the full URL of the internal detail page; for TYPE A, leave as ""

Rules:
- Do not include navigation links, blog posts, social media profiles, or news articles.
- If a company has both types, prefer TYPE A.
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

    # Split into direct hits and detail pages that need follow-up
    direct = [c for c in result if c.get("company_url")]
    needs_detail = [c for c in result if not c.get("company_url") and c.get("detail_url")]
    if debug:
        print(f"[DEBUG] Claude identified {len(direct)} direct URLs, {len(needs_detail)} detail pages", file=sys.stderr)

    if needs_detail:
        print(f"  Following {len(needs_detail)} company detail pages...", file=sys.stderr)
        resolved = _resolve_detail_pages(needs_detail, base_host, source_label, client)
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
        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as e:
            if attempt == max_retries:
                raise
            wait = delays[attempt]
            print(f"    API error ({e.__class__.__name__}), retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)


def get_founder_info(
    company_url: str, company_name: str, client: anthropic.Anthropic
) -> tuple[str, str]:
    """
    Try to find founder info by:
    1. Fetching the company's About/Team page.
    2. Asking Claude to extract the founder name, falling back to its own knowledge.
    """
    site_text = ""
    try:
        html, _ = fetch_page(company_url)
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        site_text = soup.get_text(separator="\n", strip=True)[:5000]

        # Also try /about page
        about_url = company_url.rstrip("/") + "/about"
        try:
            about_html, _ = fetch_page(about_url)
            about_soup = BeautifulSoup(about_html, "lxml")
            for tag in about_soup(["script", "style", "noscript"]):
                tag.decompose()
            site_text += "\n" + about_soup.get_text(separator="\n", strip=True)[:3000]
        except Exception:
            pass
    except Exception as e:
        print(
            f"    Could not fetch {company_url}: {e}",
            file=sys.stderr,
        )

    context_section = (
        f"Text scraped from the company's website:\n{site_text}\n\n"
        if site_text
        else ""
    )

    prompt = f"""I need the founder's name for the company "{company_name}" (website: {company_url}).

{context_section}Using both the scraped text above (if any) AND your own knowledge, return the founder's name.
If there are co-founders, return the primary/most well-known one.

Return ONLY a JSON object with keys "first_name" and "last_name".
If the founder is truly unknown, use empty strings.

Example: {{"first_name": "Brian", "last_name": "Chesky"}}"""

    response = call_claude_with_retry(
        client,
        model="claude-opus-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = next((b.text for b in response.content if b.type == "text"), "{}")
    data = extract_json(raw, array=False)
    if isinstance(data, dict):
        return data.get("first_name", ""), data.get("last_name", "")
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
        description="Extract tech company links (and founders) from web pages."
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
        "--no-founders",
        action="store_true",
        help="Skip founder lookup (faster, columns will be empty).",
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

    for source in args.sources:
        print(f"\nProcessing: {source}")
        try:
            html, base_url = fetch_page(source)
        except Exception as e:
            print(f"  Failed to load {source}: {e}", file=sys.stderr)
            continue

        print("  Identifying tech companies via Claude...")
        companies = identify_tech_companies(html, base_url, source, client, debug=args.debug)
        print(f"  Found {len(companies)} tech companies.")
        for c in companies:
            c["source_url"] = clean_url(source)
        all_companies.extend(companies)

    all_companies = deduplicate(all_companies)
    print(f"\nTotal unique companies: {len(all_companies)}")

    if not all_companies:
        print("No tech companies found. Exiting.")
        sys.exit(0)

    fieldnames = ["company_name", "company_url", "founder_first_name", "founder_last_name", "source_url"]

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
            if not args.no_founders and url:
                print(f"  [{i}/{len(all_companies)}] Looking up founder for {name}...")
                first, last = get_founder_info(url, name, client)

            row = {
                "company_name": name,
                "company_url": clean_url(url),
                "founder_first_name": first,
                "founder_last_name": last,
                "source_url": company.get("source_url", ""),
            }
            writer.writerow(row)
            out_f.flush()
            completed_urls.add(url_key)
    finally:
        out_f.close()

    print(f"\nDone. Results saved to: {args.output}")
    print(f"Columns: company_name, company_url, founder_first_name, founder_last_name, source_url")


if __name__ == "__main__":
    main()
