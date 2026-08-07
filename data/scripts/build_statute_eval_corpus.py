"""RETIRED (2026-07-29) — statute is no longer part of the evaluation suite.

Kept as a record of the fetch/QA-generation approach, and because the Cornell
LII scraping logic below is the non-obvious part. Not run by any notebook.

Why it was dropped: this is the only genre whose questions are written by an
LLM rather than taken from a human-authored dataset (narrativeqa, NewsQA,
SQuAD) or derived from real documents (CaseHOLD). No statute-QA dataset
exists, so there was no alternative. That asymmetry would need a caveat on
every genre comparison, and the yield was thin regardless — Chapter 1 gives
14,000 words and 11 usable questions, of which only 1 has its evidence inside
the first 3,000 words, so it could not carry a length ladder at all.

An expansion was attempted (6 chapters, ~59,000 words, 6 QA per section,
generation moved to a larger model so the model under test would not be
setting its own exam). It is reverted here so these settings match the data
actually on disk. The section-fetch cache from that attempt was deleted.

Builds the legal-statute-genre evaluation corpus — the English
replacement for the Korean National Law Information Center approach: fetch real statute text,
then use an LLM to generate QA pairs grounded in it (there's no existing
statute-QA dataset the way SQuAD/NewsQA exist for wiki/news, so this mirrors
`convert_korquad_tables.py`'s "real text + NIM-generated structured content"
pattern instead of a plain load_dataset call).

Note: `uscode.house.gov` (the official source) is unreachable from this
environment (connection times out — likely blocks non-browser/cloud
traffic); Cornell's Legal Information Institute (law.cornell.edu) mirrors
the same U.S. Code text and is reachable, so it's used as the fetch source.

Source: Title 17 (Copyright), Chapter 1 ("Subject Matter and Scope of
Copyright") — a well-scoped, non-obscure chapter with substantive,
question-answerable definitional/rule text (as opposed to purely procedural
cross-reference sections).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

root = Path(__file__).resolve().parent
while not (root / "src").exists() and root != root.parent:
    root = root.parent

CHAPTER_URLS = ["https://www.law.cornell.edu/uscode/text/17/chapter-1"]
BASE_URL = "https://www.law.cornell.edu"
_HEADERS = {"User-Agent": "Mozilla/5.0 (research corpus build; contact: n/a)"}

TARGET_MAX_WORDS = 14_000
N_QA_PER_SECTION = 2

GEN_MODEL = "meta/llama-3.1-8b-instruct"
_CHAT_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
QC_MAX_OCCURRENCE = 1  # answer must appear exactly once in the final corpus

_QA_LINE_PATTERN = re.compile(r"Q:\s*(.+?)\s*\n\s*A:\s*(.+?)(?:\n|$)", re.IGNORECASE)


# The free NIM endpoints are shared, so 503 "ResourceExhausted: Worker local
# total request limit reached (N/M)" means *other people's* requests have filled
# the worker queue — nothing we can send more slowly fixes it, we just have to
# wait for the queue to drain. A few seconds is never enough; these are the
# waits between retries, in seconds.
_CONGESTION_BACKOFF = [15.0, 30.0, 60.0, 90.0, 120.0]


def call_chat(messages: list[dict], api_key: str, timeout: float = 90.0,
              max_retries: int = len(_CONGESTION_BACKOFF)) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                _CHAT_API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
                json={"model": GEN_MODEL, "messages": messages, "temperature": 0.0},
                timeout=timeout,
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt < max_retries:
                wait = _CONGESTION_BACKOFF[min(attempt, len(_CONGESTION_BACKOFF) - 1)]
                print(f"    {type(e).__name__} — retrying in {wait:.0f}s ({attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            raise RuntimeError(f"NIM chat API call failed (retries exhausted): {e}") from e

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        if response.status_code in _RETRYABLE_STATUS and attempt < max_retries:
            wait = _CONGESTION_BACKOFF[min(attempt, len(_CONGESTION_BACKOFF) - 1)]
            print(f"    HTTP {response.status_code} — retrying in {wait:.0f}s ({attempt + 1}/{max_retries})")
            time.sleep(wait)
            continue
        raise RuntimeError(f"NIM chat API failed ({response.status_code}): {response.text}")

    raise RuntimeError(f"NIM chat API call failed (retries exhausted): {last_error}")


def fetch_section_links() -> list[tuple[str, str]]:
    """Returns [(url, heading), ...] for each real (non-renumbered/repealed)
    section across every chapter in CHAPTER_URLS, in order, de-duplicated
    (chapter index pages can cross-link to sections in other chapters)."""
    links: list[tuple[str, str]] = []
    seen: set[str] = set()

    for chapter_url in CHAPTER_URLS:
        r = requests.get(chapter_url, timeout=20, headers=_HEADERS)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        before = len(links)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not re.match(r"^/uscode/text/17/\d+[A-Za-z]?$", href):
                continue
            heading = a.get_text(strip=True)
            if heading.startswith("["):  # e.g. "[§ 116A. Renumbered § 116]" — not real content
                continue
            url = BASE_URL + href
            if url in seen:
                continue
            seen.add(url)
            links.append((url, heading))
        print(f"  {chapter_url.rsplit('/', 1)[-1]}: +{len(links) - before} sections")
        time.sleep(0.3)  # be polite to the source site

    return links


def fetch_section_text(url: str) -> str:
    """The page has three tabs (statute text / Notes / Authorities (CFR)) all
    rendered into the DOM at once; only the active one (class "tab-pane
    active", no other qualifying class) is the actual statute text — the
    other two are annotations/citations we don't want polluting the corpus.
    (The page previously had a `div#main-content` wrapper for just this
    content, per an earlier manual check, but that id no longer appears in
    the live HTML — presumably a site template change — so this matches on
    class instead.)
    """
    r = requests.get(url, timeout=20, headers=_HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    for div in soup.find_all("div", class_=True):
        classes = set(div.get("class"))
        if {"tab-pane", "active"}.issubset(classes) and len(classes) == 2:
            return div.get_text(separator=" ", strip=True)
    return ""


def _word_index(text: str, char_pos: int) -> int:
    return len(text[:char_pos].split()) + 1


_QA_PROMPT_TEMPLATE = (
    "Below is one section of U.S. copyright law (Title 17). Write {n} question-answer "
    "pairs about specific, concrete facts stated in this section (definitions, numeric "
    "thresholds, named exceptions, specific rules — not vague summary questions). "
    "The answer to each question MUST be a short phrase copied verbatim (exact wording) "
    "from the section text below, 3-15 words long. Do not invent facts not in the text.\n\n"
    "[OUTPUT FORMAT] Output only the following, nothing else:\n"
    "Q: question\nA: verbatim answer\n"
    "(repeat for each pair)\n\n"
    "[SECTION TEXT]\n{section_text}"
)


def generate_qa_for_section(heading: str, section_text: str, api_key: str) -> list[dict]:
    prompt = _QA_PROMPT_TEMPLATE.format(n=N_QA_PER_SECTION, section_text=section_text[:3000])
    messages = [
        {"role": "system", "content": "You write precise, fact-grounded quiz questions about legal text."},
        {"role": "user", "content": prompt},
    ]
    raw = call_chat(messages, api_key=api_key)
    pairs = _QA_LINE_PATTERN.findall(raw)
    return [{"question": q.strip(), "answer": a.strip()} for q, a in pairs]


CACHE_DIR = root / "data" / "processed" / "statute_eval"
SECTION_CACHE = CACHE_DIR / "_cache_sections.json"
QA_CACHE = CACHE_DIR / "_cache_qa.json"


def _load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    api_key = os.environ.get("NVIDIA_NIM_API_KEY")
    if not api_key:
        raise SystemExit("NVIDIA_NIM_API_KEY is not set.")

    # Crawling 21 pages and regenerating QA is expensive, and the shared NIM
    # endpoint fails often enough that a run rarely finishes first try. Both
    # stages cache to disk so a re-run picks up where it stopped.
    section_cache = _load_cache(SECTION_CACHE)
    qa_cache = _load_cache(QA_CACHE)
    if section_cache or qa_cache:
        print(f"cache: {len(section_cache)} sections fetched, {len(qa_cache)} sections with QA")

    print(f"Fetching section lists from {len(CHAPTER_URLS)} chapters...")
    links = fetch_section_links()
    print(f"sections found: {len(links)}")

    text_parts: list[str] = []
    section_bounds: list[tuple[str, str, int, int]] = []  # (heading, text, start_char, end_char)
    cumulative_chars = 0

    for url, heading in links:
        if url in section_cache:
            section_text = section_cache[url]
        else:
            section_text = fetch_section_text(url)
            section_cache[url] = section_text
            _save_cache(SECTION_CACHE, section_cache)
        if not section_text or len(section_text.split()) < 30:
            continue

        block = f"{heading}\n\n{section_text}"
        start = cumulative_chars
        text_parts.append(block)
        cumulative_chars += len(block) + len("\n\n")
        section_bounds.append((heading, section_text, start, cumulative_chars))

        current_words = sum(len(p.split()) for p in text_parts)
        print(f"  fetched {heading[:50]:50} ({len(section_text.split())} words, running total {current_words})")
        if current_words >= TARGET_MAX_WORDS:
            break
        time.sleep(0.3)  # be polite to the source site

    full_text = "\n\n".join(text_parts)
    print(f"\ncorpus: {len(full_text)} chars, {len(full_text.split())} words, {len(section_bounds)} sections")

    print("\nGenerating QA per section via NIM...")
    questions = []
    for heading, section_text, start, end in section_bounds:
        if heading in qa_cache:
            qa_items = qa_cache[heading]
            print(f"  [{heading[:40]}] {len(qa_items)} from cache")
        else:
            try:
                qa_items = generate_qa_for_section(heading, section_text, api_key)
            except RuntimeError as e:
                print(f"  [{heading[:40]}] QA generation failed: {e} — skipped (re-run to retry)")
                continue
            qa_cache[heading] = qa_items
            _save_cache(QA_CACHE, qa_cache)
            time.sleep(1.0)  # gentle spacing between generation calls

        for item in qa_items:
            answer = item["answer"]
            local_pos = full_text.find(answer, start, end)
            if local_pos == -1:
                continue  # model didn't copy verbatim — drop rather than guess
            occurrence = full_text.count(answer)
            if not (1 <= occurrence <= QC_MAX_OCCURRENCE):
                continue
            questions.append(
                {
                    "question": item["question"],
                    "answer": answer,
                    "evidence_char_pos": local_pos,
                    "evidence_word_pos": _word_index(full_text, local_pos),
                    "source_section": heading,
                    "type": "statute_llm_generated",
                }
            )
        print(f"  [{heading[:40]}] {len(qa_items)} generated, running QC-passed total: {len(questions)}")

    for i, q in enumerate(questions, start=1):
        q["question_id"] = i

    print(f"\nfinal QA count (passed occurrence-count QC): {len(questions)}")

    out_dir = root / "data" / "processed" / "statute_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = root / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    text_path = out_dir / "statute_eval_corpus.txt"
    text_path.write_text(full_text, encoding="utf-8")

    import pandas as pd

    questions_df = pd.DataFrame(questions)[
        ["question_id", "question", "answer", "evidence_char_pos", "evidence_word_pos", "source_section", "type"]
    ]
    csv_path = sample_dir / "statute_eval_questions.csv"
    questions_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    manifest_path = out_dir / "statute_eval_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"source": CHAPTER_URLS, "gen_model": GEN_MODEL,
             "n_sections": len(section_bounds), "n_questions": len(questions)}, indent=2
        ),
        encoding="utf-8",
    )

    print(f"\nSaved:\n  {text_path}\n  {csv_path}\n  {manifest_path}")


if __name__ == "__main__":
    main()
