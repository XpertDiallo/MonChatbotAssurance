from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader


DEFAULT_PAGE_START = 37
DEFAULT_PAGE_END = 408
ARTICLE_HEADING_RE = re.compile(
    r"^Articles?\s+(?P<number>\d+(?:-\d+)*(?:\s+bis)?)\s*$",
    re.IGNORECASE,
)
PAGE_NUMBER_RE = re.compile(r"^-\s*\d+\s*-$")
FOOTER_RE = re.compile(r"2509-18-CIMA|Mise en page|Page\d+", re.IGNORECASE)
SECTION_HEADING_RE = re.compile(
    r"^(LIVRE\s+[IVX]+|TITRE\s+[IVX]+|CHAPITRE\s+(?:PREMIER|UNIQUE|[IVX]+|\d+ER)|SECTION\s+(?:PREMIERE|PREMIÈRE|[IVX]+|\d+))\b",
    re.IGNORECASE,
)


def clean_line(line: str) -> str:
    line = line.replace("\x00", " ")
    return re.sub(r"\s+", " ", line).strip()


def update_context(context: list[str], heading: str) -> list[str]:
    upper = heading.upper()
    if upper.startswith("LIVRE"):
        return [heading]
    if upper.startswith("TITRE"):
        return context[:1] + [heading]
    if upper.startswith("CHAPITRE"):
        return context[:2] + [heading]
    if upper.startswith("SECTION"):
        return context[:3] + [heading]
    return context


def close_article(current: dict[str, Any] | None, articles: list[dict[str, Any]], page: int) -> None:
    if current is None:
        return

    current["page_end"] = page
    lines = [line for line in current.pop("lines") if line]
    title = ""
    body_lines: list[str] = []
    title_taken = False

    for line in lines:
        if not title_taken and not line.startswith("("):
            title = line
            title_taken = True
        else:
            body_lines.append(line)

    body = "\n".join(body_lines).strip()
    current["title"] = title.strip()
    current["body"] = body
    current["text"] = f"{title}\n{body}".strip()
    articles.append(current)


def extract_articles(pdf_path: Path, page_start: int, page_end: int) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    articles: list[dict[str, Any]] = []
    context: list[str] = []
    current: dict[str, Any] | None = None

    if page_start < 1 or page_end > len(reader.pages):
        raise ValueError(f"Pages demandées hors limites: PDF de {len(reader.pages)} pages.")

    for page_number in range(page_start, page_end + 1):
        text = reader.pages[page_number - 1].extract_text() or ""
        for raw_line in text.splitlines():
            line = clean_line(raw_line)
            if not line or PAGE_NUMBER_RE.match(line) or FOOTER_RE.search(line):
                continue

            article_match = ARTICLE_HEADING_RE.match(line)
            if article_match:
                close_article(current, articles, page_number)
                current = {
                    "article": re.sub(r"\s+", " ", article_match.group("number")).strip(),
                    "page_start": page_number,
                    "page_end": page_number,
                    "section": " > ".join(context[-4:]),
                    "lines": [],
                }
                continue

            section_match = SECTION_HEADING_RE.match(line)
            if section_match and current is None:
                context = update_context(context, line)
                continue

            if section_match and current and current["lines"]:
                close_article(current, articles, page_number)
                current = None
                context = update_context(context, line)
                continue

            if current is not None:
                current["lines"].append(line)

    close_article(current, articles, page_end)
    return articles


def build_payload(pdf_path: Path, page_start: int, page_end: int) -> dict[str, Any]:
    articles = extract_articles(pdf_path, page_start, page_end)
    return {
        "source": {
            "title": "Code des assurances des États membres de la CIMA",
            "edition": "2019",
            "pdf_name": pdf_path.name,
            "page_start": page_start,
            "page_end": page_end,
            "article_count": len(articles),
        },
        "articles": articles,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrait les articles du Code CIMA 2019 en JSON pour le chatbot Streamlit."
    )
    parser.add_argument("pdf", type=Path, help="Chemin du fichier CODE-CIMA-2019.pdf")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "code_cima_articles.json",
        help="Chemin du JSON généré.",
    )
    parser.add_argument("--page-start", type=int, default=DEFAULT_PAGE_START)
    parser.add_argument("--page-end", type=int, default=DEFAULT_PAGE_END)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args.pdf, args.page_start, args.page_end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(
        f"{payload['source']['article_count']} articles extraits dans {args.output} "
        f"(pages {args.page_start}-{args.page_end})."
    )


if __name__ == "__main__":
    main()
