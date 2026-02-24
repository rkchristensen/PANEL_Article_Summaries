#!/usr/bin/env python3
"""Build government/nonprofit ethics tiles from academic article metadata."""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "data" / "stories.json"
SUMMARY_CACHE_PATH = ROOT / "data" / "summaries_cache.json"

ACADEMIC_QUERIES = [
    "government corruption",
    "public sector ethics",
    "public sector graft",
    "political greed",
    "nonprofit corruption",
    "charity fraud governance",
    "ngo ethics",
    "civil society accountability",
    "anti-corruption policy",
    "anti-graft institutions",
]

POSITIVE_KEYWORDS = {
    "anti-corruption",
    "anti corruption",
    "anticorruption",
    "anti graft",
    "anti-graft",
    "anti bribery",
    "anti-bribery",
    "anti fraud",
    "anti-fraud",
    "anti-greed",
    "anti greed",
    "integrity",
    "reform",
    "transparency",
    "oversight",
    "accountability",
    "improve",
    "improved",
    "improves",
    "cleaned up",
    "cleared",
    "acquitted",
    "new ethics rules",
    "adopts ethics",
}

NEGATIVE_KEYWORDS = {
    "ethics",
    "unethical",
    "anti-ethics",
    "corruption",
    "graft",
    "greed",
    "bribery",
    "bribe",
    "fraud",
    "embezzlement",
    "kickback",
    "money laundering",
    "scandal",
    "probe",
    "investigation",
    "charged",
    "indicted",
    "convicted",
    "arrested",
    "misuse",
    "misconduct",
}

GOVERNMENT_TERMS = {
    "government",
    "public",
    "municipal",
    "city",
    "state",
    "federal",
    "minister",
    "senate",
    "congress",
    "parliament",
    "mayor",
    "governor",
    "agency",
    "department",
    "county",
}

NONPROFIT_TERMS = {
    "nonprofit",
    "non-profit",
    "non profit",
    "charity",
    "charitable",
    "foundation",
    "ngo",
    "non-governmental",
    "civil society",
    "not-for-profit",
    "not for profit",
    "philanthropy",
    "donor-funded",
}

BUSINESS_TERMS = {
    "business",
    "corporate",
    "corporation",
    "company",
    "earnings",
    "quarterly results",
    "stock",
    "share price",
    "ipo",
    "merger",
    "acquisition",
    "ceo",
    "investor",
    "wall street",
}

MAX_STORIES_PER_COLUMN = 60
ROWS_PER_QUERY = 40
USER_AGENT = "nonmarket-ethics-scholarship-feed/1.0 (+https://github.com/rkchristensen/nonmarket_ethics_scholarship_feed)"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
MAX_NEW_SUMMARIES_PER_RUN = 20


@dataclass
class Story:
    title: str
    short_title: str
    url: str
    source: str
    published_at: str
    sentiment: str
    government: bool
    nonprofit: bool
    plain_summary: str | None = None


def crossref_works_url(query: str) -> str:
    params = urllib.parse.urlencode(
        {
            "query.bibliographic": query,
            "rows": str(ROWS_PER_QUERY),
            "sort": "published",
            "order": "desc",
            "filter": "type:journal-article",
            "select": "DOI,title,URL,published,published-online,published-print,created,container-title,publisher,abstract",
        }
    )
    return f"https://api.crossref.org/works?{params}"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.read()
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLCertVerificationError):
            # Local Python setups occasionally miss trusted cert bundles.
            insecure = ssl._create_unverified_context()
            with urllib.request.urlopen(request, timeout=20, context=insecure) as response:
                return response.read()
        raise


def parse_date_parts(raw: object) -> datetime:
    if not isinstance(raw, dict):
        return datetime.now(timezone.utc)
    date_parts = raw.get("date-parts")
    if not isinstance(date_parts, list) or not date_parts:
        return datetime.now(timezone.utc)
    first = date_parts[0]
    if not isinstance(first, list) or not first:
        return datetime.now(timezone.utc)
    year = int(first[0])
    month = int(first[1]) if len(first) > 1 else 1
    day = int(first[2]) if len(first) > 2 else 1
    try:
        dt = datetime(year, month, day, tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        if dt.year < 1900 or dt.year > now.year + 1:
            return now
        return dt
    except ValueError:
        return datetime.now(timezone.utc)


def short_title(title: str, limit: int = 95) -> str:
    cleaned = re.sub(r"\s+", " ", title).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def contains_any(text: str, terms: Iterable[str]) -> bool:
    for term in terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, text):
            return True
    return False


def clean_text(value: str) -> str:
    # Crossref abstracts can include lightweight markup like <jats:p>.
    no_tags = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", no_tags).strip()


def classify_sentiment(text: str) -> str | None:
    if contains_any(text, NEGATIVE_KEYWORDS):
        return "negative"
    if contains_any(text, POSITIVE_KEYWORDS):
        return "positive"
    return None


def parse_crossref_items(payload_bytes: bytes) -> list[dict]:
    parsed: list[dict] = []
    data = json.loads(payload_bytes.decode("utf-8", errors="replace"))
    items = data.get("message", {}).get("items", [])
    if not isinstance(items, list):
        return parsed

    for item in items:
        title_values = item.get("title") or []
        title = ""
        if isinstance(title_values, list) and title_values:
            title = clean_text(str(title_values[0]))
        elif isinstance(title_values, str):
            title = clean_text(title_values)

        url = clean_text(str(item.get("URL") or ""))
        doi = clean_text(str(item.get("DOI") or ""))
        abstract = clean_text(str(item.get("abstract") or ""))

        source = ""
        container_values = item.get("container-title") or []
        if isinstance(container_values, list) and container_values:
            source = clean_text(str(container_values[0]))
        elif isinstance(container_values, str):
            source = clean_text(container_values)
        if not source:
            source = clean_text(str(item.get("publisher") or "")) or "Unknown source"

        published_at = (
            parse_date_parts(item.get("published-online"))
            if item.get("published-online")
            else parse_date_parts(item.get("published-print"))
            if item.get("published-print")
            else parse_date_parts(item.get("published"))
            if item.get("published")
            else parse_date_parts(item.get("created"))
        )

        if doi and not url:
            url = f"https://doi.org/{doi}"

        parsed.append(
            {
                "title": title,
                "url": url,
                "doi": doi,
                "published_at": published_at,
                "source": source,
                "abstract": abstract,
            }
        )
    return parsed


def should_skip_business_only(text: str) -> bool:
    has_business = contains_any(text, BUSINESS_TERMS)
    has_relevant_domain = contains_any(text, GOVERNMENT_TERMS) or contains_any(text, NONPROFIT_TERMS)
    return has_business and not has_relevant_domain


def normalize_story(raw: dict) -> Story | None:
    if not raw["title"] or not raw["url"]:
        return None

    text = f'{raw["title"]} {raw["source"]} {raw.get("abstract", "")}'.lower()
    if should_skip_business_only(text):
        return None

    sentiment = classify_sentiment(text)
    if sentiment is None:
        return None

    is_government = contains_any(text, GOVERNMENT_TERMS)
    is_nonprofit = contains_any(text, NONPROFIT_TERMS)

    if not is_government and not is_nonprofit:
        return None

    return Story(
        title=raw["title"],
        short_title=short_title(raw["title"]),
        url=raw["url"],
        source=raw["source"],
        published_at=raw["published_at"].isoformat(),
        sentiment=sentiment,
        government=is_government,
        nonprofit=is_nonprofit,
    )


def load_summary_cache() -> dict[str, str]:
    if not SUMMARY_CACHE_PATH.exists():
        return {}
    try:
        raw = json.loads(SUMMARY_CACHE_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items() if isinstance(v, str) and v.strip()}
    except Exception:
        pass
    return {}


def save_summary_cache(cache: dict[str, str]) -> None:
    SUMMARY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def compress_summary(summary: str) -> str:
    cleaned = re.sub(r"\s+", " ", summary).strip()
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 4:
        return " ".join(parts[:4])
    return cleaned


def summarize_plain_language(title: str, source: str, abstract: str) -> str | None:
    if not abstract.strip():
        return None

    prompt = (
        "Write a plain-language summary for practitioners in exactly 3 to 4 sentences. "
        "Use simple wording, avoid jargon, and explain practical implications. "
        "Do not use bullets, markdown, or headings.\n\n"
        f"Title: {title}\n"
        f"Source: {source}\n"
        f"Abstract: {abstract}\n"
    )

    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            raw = json.loads(response.read().decode("utf-8", errors="replace"))
            text = str(raw.get("response", "")).strip()
            text = compress_summary(text)
            return text or None
    except Exception:
        return None


def collect_stories() -> list[Story]:
    collected: list[Story] = []
    seen_keys: set[str] = set()
    summary_cache = load_summary_cache()
    cache_changed = False
    new_summaries = 0

    for query in ACADEMIC_QUERIES:
        try:
            response = fetch(crossref_works_url(query))
        except Exception:
            continue

        for item in parse_crossref_items(response):
            key = item.get("doi") or item["url"]
            if key in seen_keys:
                continue
            story = normalize_story(item)
            if story is None:
                continue

            plain_summary = summary_cache.get(key)
            if plain_summary is None and new_summaries < MAX_NEW_SUMMARIES_PER_RUN:
                plain_summary = summarize_plain_language(
                    title=item["title"],
                    source=item["source"],
                    abstract=item.get("abstract", ""),
                )
                new_summaries += 1
                if plain_summary:
                    summary_cache[key] = plain_summary
                    cache_changed = True

            story.plain_summary = plain_summary
            seen_keys.add(key)
            collected.append(story)

    if cache_changed:
        save_summary_cache(summary_cache)

    return sorted(
        collected,
        key=lambda s: s.published_at,
        reverse=True,
    )


def build_output(stories: list[Story]) -> dict:
    government = [s for s in stories if s.government][:MAX_STORIES_PER_COLUMN]
    nonprofit = [s for s in stories if s.nonprofit][:MAX_STORIES_PER_COLUMN]

    def to_dict(story: Story) -> dict:
        return {
            "title": story.title,
            "short_title": story.short_title,
            "url": story.url,
            "source": story.source,
            "published_at": story.published_at,
            "sentiment": story.sentiment,
            "plain_summary": story.plain_summary,
        }

    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "government": [to_dict(s) for s in government],
        "nonprofit": [to_dict(s) for s in nonprofit],
    }


def main() -> None:
    stories = collect_stories()
    payload = build_output(stories)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} with {len(payload['government'])} government and {len(payload['nonprofit'])} nonprofit stories.")


if __name__ == "__main__":
    main()
