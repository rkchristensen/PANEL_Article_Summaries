#!/usr/bin/env python3
"""Build government/nonprofit ethics tiles from academic article metadata."""

from __future__ import annotations

import json
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


def collect_stories() -> list[Story]:
    collected: list[Story] = []
    seen_keys: set[str] = set()

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
            seen_keys.add(key)
            collected.append(story)

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
