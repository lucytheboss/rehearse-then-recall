"""Builds the legal-statute-genre evaluation corpus — the English
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

CHAPTER_URL = "https://www.law.cornell.edu/uscode/text/17/chapter-1"
BASE_URL = "https://www.law.cornell.edu"
_HEADERS = {"User-Agent": "Mozilla/5.0 (research corpus build; contact: n/a)"}

TARGET_MAX_WORDS = 14_000
N_QA_PER_SECTION = 2

GEN_MODEL = "meta/llama-3.1-8b-instruct"
_CHAT_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
QC_MAX_OCCURRENCE = 1  # answer must appear exactly once in the final corpus

_QA_LINE_PATTERN = re.compile(r"Q:\s*(.+?)\s*\n\s*A:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def call_chat(messages: list[dict], api_key: str, timeout: float = 90.0, max_retries: int = 3) -> str:
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
                time.sleep(3.0)
                continue
            raise RuntimeError(f"NIM chat API call failed (retries exhausted): {e}") from e

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        if response.status_code in _RETRYABLE_STATUS and attempt < max_retries:
            time.sleep(3.0)
            continue
        raise RuntimeError(f"NIM chat API failed ({response.status_code}): {response.text}")

    raise RuntimeError(f"NIM chat API call failed (retries exhausted): {last_error}")


def fetch_section_links() -> list[tuple[str, str]]:
    """Returns [(url, heading), ...] for each real (non-renumbered/repealed) section."""
    r = requests.get(CHAPTER_URL, timeout=20, headers=_HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not re.match(r"^/uscode/text/17/\d+[A-Za-z]?$", href):
            continue
        heading = a.get_text(strip=True)
        if heading.startswith("["):  # e.g. "[§ 116A. Renumbered § 116]" — not real content
            continue
        links.append((BASE_URL + href, heading))
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


def main() -> None:
    api_key = os.environ.get("NVIDIA_NIM_API_KEY")
    if not api_key:
        raise SystemExit("NVIDIA_NIM_API_KEY is not set.")

    print(f"Fetching section list from {CHAPTER_URL}...")
    links = fetch_section_links()
    print(f"sections found: {len(links)}")

    text_parts: list[str] = []
    section_bounds: list[tuple[str, str, int, int]] = []  # (heading, text, start_char, end_char)
    cumulative_chars = 0

    for url, heading in links:
        section_text = fetch_section_text(url)
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
        try:
            qa_items = generate_qa_for_section(heading, section_text, api_key)
        except RuntimeError as e:
            print(f"  [{heading[:40]}] QA generation failed: {e} — skipped")
            continue

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
            {"source": CHAPTER_URL, "n_sections": len(section_bounds), "n_questions": len(questions)}, indent=2
        ),
        encoding="utf-8",
    )

    print(f"\nSaved:\n  {text_path}\n  {csv_path}\n  {manifest_path}")


if __name__ == "__main__":
    main()
