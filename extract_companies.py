#!/usr/bin/env python3
"""
Extract tech company links (and optionally founder names) from web pages.

Usage:
    python extract_companies.py page1.html page2.html --output companies.csv
    python extract_companies.py https://example.com/portfolio --output companies.csv
    python extract_companies.py page.html --no-founders --output companies.csv

Outputs a CSV with columns:
    company_name, company_url, founder_first_name, founder_last_name

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


def fetch_page(source: str) -> tuple[str, str]:
    """Return (html_content, base_url) for a local file or URL."""
    if os.path.exists(source):
        with open(source, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        base_url = "file://" + os.path.abspath(source)
        return content, base_url

    resp = requests.get(
        source,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 (compatible; CompanyExtractor/1.0)"},
    )
    resp.raise_for_status()
    return resp.text, source


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
    pattern = r"\[.*?\]" if array else r"\{.*?\}"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return [] if array else {}


def identify_tech_companies(
    html: str, base_url: str, source_label: str, client: anthropic.Anthropic
) -> list[dict]:
    """Ask Claude which links on this page lead to tech companies."""
    page_text, links = page_text_and_links(html, base_url)

    if not links:
        print(f"  No links found in {source_label}", file=sys.stderr)
        return []

    prompt = f"""You are a research assistant. I have a web page that lists tech companies.

Page text (truncated):
{page_text[:6000]}

All links found on the page (up to 300):
{json.dumps(links[:300], indent=2)}

Task: Identify every link that points directly to a tech company's own website.
Return ONLY a JSON array. Each element must have:
  - "company_name": the company's name (string)
  - "company_url": the company's website URL (string)

Rules:
- Only include links that go to the company's own site (not news articles, not social media profiles, not investor pages about the company).
- If the anchor text or surrounding context makes the company name clear, use it. Otherwise infer from the domain.
- Skip duplicate companies.
- If no tech company links are found, return [].

Respond with the JSON array only — no explanation, no markdown fences."""

    response = call_claude_with_retry(
        client,
        model="claude-opus-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )

    raw = next((b.text for b in response.content if b.type == "text"), "[]")
    result = extract_json(raw, array=True)
    return result if isinstance(result, list) else []


def call_claude_with_retry(client: anthropic.Anthropic, max_retries: int = 4, **kwargs) -> anthropic.types.Message:
    """Call client.messages.create with exponential backoff on timeout/connection errors."""
    delays = [2, 4, 8, 16]
    for attempt in range(max_retries + 1):
        try:
            return client.messages.create(**kwargs)
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
        companies = identify_tech_companies(html, base_url, source, client)
        print(f"  Found {len(companies)} tech companies.")
        all_companies.extend(companies)

    all_companies = deduplicate(all_companies)
    print(f"\nTotal unique companies: {len(all_companies)}")

    if not all_companies:
        print("No tech companies found. Exiting.")
        sys.exit(0)

    fieldnames = ["company_name", "company_url", "founder_first_name", "founder_last_name"]

    # Load checkpoint: any rows already written to the output CSV
    completed_urls: set[str] = set()
    if os.path.exists(args.output):
        with open(args.output, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_urls.add(row.get("company_url", "").rstrip("/").lower())
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
            url_key = url.rstrip("/").lower()

            if url_key in completed_urls:
                print(f"  [{i}/{len(all_companies)}] Skipping {name} (already done)")
                continue

            first, last = "", ""
            if not args.no_founders and url:
                print(f"  [{i}/{len(all_companies)}] Looking up founder for {name}...")
                first, last = get_founder_info(url, name, client)

            row = {
                "company_name": name,
                "company_url": url,
                "founder_first_name": first,
                "founder_last_name": last,
            }
            writer.writerow(row)
            out_f.flush()
            completed_urls.add(url_key)
    finally:
        out_f.close()

    print(f"\nDone. Results saved to: {args.output}")
    print(f"Columns: company_name, company_url, founder_first_name, founder_last_name")


if __name__ == "__main__":
    main()
