from __future__ import annotations

import html
import re
from difflib import ndiff
from typing import Dict, List


TECHNICAL_HEADINGS = {
    "что было исправлено",
    "ручная проверка педагогом",
    "проверка педагогом",
}

ALWAYS_KEEP_HEADINGS = {
    "финальный отчёт",
    "финальный отчет",
    "отчёт по занятию",
    "отчет по занятию",
    "краткое содержание",
    "карта занятия",
    "что ученик сделал самостоятельно",
    "что ученик сделал после подсказки",
    "что объяснял преподаватель",
    "что вызвало трудности",
    "педагогический вывод",
    "рекомендации на следующее занятие",
    "домашнее задание",
    "памятка для ученика",
    "предлагаемое обновление базы ученика",
    "сомнительные места в анализе",
    "требует подтверждения педагога",
    "что требует подтверждения педагога",
}

TIMESTAMP_LINE_RE = re.compile(
    r"\[\d{1,2}:\d{2}:\d{2}(?:\s*[-–—]\s*\d{1,2}:\d{2}:\d{2})?\]"
)

MATH_TOKEN_RE = re.compile(r"@@EASYREPET_MATH_(\d+)@@")
MATH_PATTERNS = (
    re.compile(r"\$\$.*?\$\$", re.DOTALL),
    re.compile(r"\\\[.*?\\\]", re.DOTALL),
    re.compile(r"\\\(.*?\\\)", re.DOTALL),
    re.compile(r"(?<!\\)\$(?!\s)(.+?)(?<!\\)\$", re.DOTALL),
)


def repair_mojibake(text: str) -> str:
    """Repair text that was decoded as cp1252/latin1 after being encoded as UTF-8."""
    markers = sum(text.count(marker) for marker in ("Ð", "Ñ", "â", "Â", "ðŸ"))
    if markers < 3:
        return text

    candidates = [text]
    for encoding in ("latin1", "cp1252"):
        try:
            candidates.append(text.encode(encoding, errors="ignore").decode("utf-8"))
        except UnicodeError:
            pass

    return min(candidates, key=_mojibake_score)


def _mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in ("Ð", "Ñ", "â", "Â", "ðŸ"))


def normalize_heading(title: str) -> str:
    title = repair_mojibake(title)
    title = re.sub(r"^\s*#+\s*", "", title)
    title = re.sub(r"^\s*\d+[\).]\s*", "", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip(" -*_`").lower()


def is_technical_heading(title: str) -> bool:
    normalized = normalize_heading(title)
    return any(normalized == heading or normalized.startswith(f"{heading}:") for heading in TECHNICAL_HEADINGS)


def split_markdown_sections(markdown_text: str) -> List[Dict[str, str]]:
    text = repair_mojibake(markdown_text).strip()
    sections: List[Dict[str, str]] = []
    current = {"level": "0", "title": "Вступление", "body": ""}

    for line in text.splitlines():
        match = re.match(r"^(#{1,3})\s+(.+?)\s*$", line)
        if match:
            if current["body"].strip() or current["title"] != "Вступление":
                sections.append(current)
            current = {
                "level": str(len(match.group(1))),
                "title": match.group(2).strip(),
                "body": "",
            }
            continue

        current["body"] += f"{line}\n"

    if current["body"].strip() or current["title"] != "Вступление":
        sections.append(current)

    return sections or [{"level": "1", "title": "Отчет", "body": text}]


def compose_markdown_sections(sections: List[Dict[str, str]]) -> str:
    chunks: List[str] = []
    for section in sections:
        title = section.get("title", "").strip()
        body = section.get("body", "").strip()
        level = int(section.get("level") or 2)

        if title and title != "Вступление":
            level = min(max(level, 1), 3)
            chunks.append(f"{'#' * level} {title}")
        if body:
            chunks.append(body)

    return "\n\n".join(chunks).strip() + "\n"


def clean_teacher_report(markdown_text: str) -> str:
    """Keep the teacher-facing report and remove model/reviewer service sections."""
    text = repair_mojibake(markdown_text)
    text = _strip_thinking_blocks(text)
    sections = split_markdown_sections(text)
    cleaned: List[Dict[str, str]] = []

    found_final = False
    for section in sections:
        title = section["title"]
        normalized = normalize_heading(title)

        if is_technical_heading(title):
            continue

        if normalized in {"финальный отчёт", "финальный отчет"}:
            found_final = True
            cleaned = [section]
            continue

        if found_final and re.match(r"^\s*\d+[\).]\s+", title):
            candidate = normalize_heading(title)
            if candidate not in ALWAYS_KEEP_HEADINGS:
                continue

        cleaned.append(section)

    result = compose_markdown_sections(cleaned).strip()
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result + "\n"


def _strip_thinking_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"```(?:thinking|analysis|reasoning)\b.*?```", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def render_markdown(markdown_text: str) -> str:
    """Small safe Markdown renderer for the report UI."""
    text = normalize_transcript_timestamp_breaks(repair_mojibake(markdown_text))
    text, math_fragments = protect_math_fragments(text)
    html_lines: List[str] = []
    paragraph: List[str] = []
    list_type: str | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            html_lines.append(f"<p>{_inline_markdown(' '.join(paragraph))}</p>")
            paragraph = []

    def close_list() -> None:
        nonlocal list_type
        if list_type:
            html_lines.append(f"</{list_type}>")
            list_type = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            close_list()
            continue

        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
            flush_paragraph()
            close_list()
            html_lines.append("<hr>")
            continue

        heading = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            close_list()
            level = min(len(heading.group(1)), 4)
            html_lines.append(f"<h{level}>{_inline_markdown(heading.group(2))}</h{level}>")
            continue

        if TIMESTAMP_LINE_RE.match(stripped):
            flush_paragraph()
            close_list()
            html_lines.append(f'<div class="transcript-line">{_inline_markdown(stripped)}</div>')
            continue

        unordered = re.match(r"^[-*•]\s+(.+)$", stripped)
        ordered = re.match(r"^\d+[\).]\s+(.+)$", stripped)
        if unordered or ordered:
            flush_paragraph()
            expected = "ul" if unordered else "ol"
            if list_type != expected:
                close_list()
                list_type = expected
                html_lines.append(f"<{list_type}>")
            item = unordered.group(1) if unordered else ordered.group(1)
            html_lines.append(f"<li>{_inline_markdown(item)}</li>")
            continue

        close_list()
        paragraph.append(stripped)

    flush_paragraph()
    close_list()
    return restore_math_fragments("\n".join(html_lines), math_fragments)


def protect_math_fragments(text: str) -> tuple[str, List[str]]:
    fragments: List[str] = []

    def replace_match(match: re.Match[str]) -> str:
        token = f"@@EASYREPET_MATH_{len(fragments)}@@"
        fragments.append(match.group(0))
        return token

    protected = text
    for pattern in MATH_PATTERNS:
        protected = pattern.sub(replace_match, protected)
    return protected, fragments


def restore_math_fragments(html_text: str, fragments: List[str]) -> str:
    def replace_token(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if index >= len(fragments):
            return match.group(0)
        return fragments[index]

    return MATH_TOKEN_RE.sub(replace_token, html_text)


def normalize_transcript_timestamp_breaks(text: str) -> str:
    """Put each timestamped transcript segment on its own display line."""
    return re.sub(
        r"\s+(?=\[\d{1,2}:\d{2}:\d{2}(?:\s*[-–—]\s*\d{1,2}:\d{2}:\d{2})?\])",
        "\n",
        text,
    )


def _inline_markdown(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)
    return escaped


def build_html_diff(old_text: str, new_text: str) -> str:
    rows: List[str] = []
    for line in ndiff(repair_mojibake(old_text).splitlines(), repair_mojibake(new_text).splitlines()):
        marker = line[:2]
        value = html.escape(line[2:])
        if marker == "- ":
            rows.append(f'<div class="diff-line removed">- {value}</div>')
        elif marker == "+ ":
            rows.append(f'<div class="diff-line added">+ {value}</div>')
        elif marker == "? ":
            continue
        else:
            rows.append(f'<div class="diff-line same">  {value}</div>')
    return "\n".join(rows)
