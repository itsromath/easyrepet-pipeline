from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MOJIBAKE_MARKERS = ("Ð", "Ñ", "Ã", "Â", "�")


@dataclass
class TranscriptBlock:
    start: float
    end: float
    speaker: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert AnythingLLM master-recording.json to compact transcript files."
    )
    parser.add_argument("input", help="Path to master-recording.json")
    parser.add_argument(
        "--out-dir",
        default="transcripts",
        help="Output directory. Default: transcripts",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output base name without extension. Default: input file stem",
    )
    parser.add_argument(
        "--max-block-seconds",
        type=float,
        default=90.0,
        help="Maximum merged block duration. Default: 90",
    )
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=4.0,
        help="Maximum gap for merging same-speaker segments. Default: 4",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Write only compact JSON.",
    )
    parser.add_argument(
        "--md-only",
        action="store_true",
        help="Write only Markdown transcript.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="replace") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object at the top level.")

    return data


def repair_mojibake(text: str) -> str:
    if mojibake_score(text) < 3:
        return text

    candidates = {text}
    queue = [text]

    for _ in range(2):
        next_queue: list[str] = []
        for candidate in queue:
            for encoding in ("cp1252", "latin1"):
                try:
                    fixed = candidate.encode(encoding).decode("utf-8")
                except UnicodeError:
                    continue
                if fixed not in candidates:
                    candidates.add(fixed)
                    next_queue.append(fixed)
        queue = next_queue

    return min(candidates, key=mojibake_score)


def mojibake_score(text: str) -> int:
    marker_count = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    replacement_count = text.count("\ufffd")
    cyrillic_count = len(re.findall(r"[А-Яа-яЁё]", text))
    return marker_count * 20 + replacement_count * 30 - cyrillic_count


def clean_text(value: Any) -> str:
    text = repair_mojibake(str(value or ""))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_speaker(value: Any) -> str:
    speaker = repair_mojibake(str(value or "")).strip()
    match = re.search(r"(\d+)$", speaker)
    if match:
        return match.group(1)

    speaker = re.sub(r"(?i)\bspeaker\b", "", speaker)
    speaker = re.sub(r"\s+", " ", speaker).strip(" :-")
    return speaker or "?"


def seconds_to_stamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\wА-Яа-яЁё.-]+", "_", stem, flags=re.UNICODE)
    return stem.strip("_") or "transcript"


def extract_segments(data: dict[str, Any]) -> list[TranscriptBlock]:
    raw_segments = data.get("segments")
    if not isinstance(raw_segments, list):
        text = clean_text(data.get("text", ""))
        return [TranscriptBlock(start=0.0, end=float(data.get("durationSeconds") or 0), speaker="", text=text)]

    blocks: list[TranscriptBlock] = []
    for segment in raw_segments:
        if not isinstance(segment, dict):
            continue

        text = clean_text(segment.get("text", ""))
        if not text:
            continue

        blocks.append(
            TranscriptBlock(
                start=float(segment.get("start") or 0.0),
                end=float(segment.get("end") or segment.get("start") or 0.0),
                speaker=normalize_speaker(segment.get("speaker")),
                text=text,
            )
        )

    return blocks


def merge_segments(
    segments: list[TranscriptBlock],
    *,
    max_block_seconds: float,
    max_gap_seconds: float,
) -> list[TranscriptBlock]:
    merged: list[TranscriptBlock] = []

    for segment in segments:
        if not merged:
            merged.append(segment)
            continue

        previous = merged[-1]
        same_speaker = previous.speaker == segment.speaker
        close_enough = segment.start - previous.end <= max_gap_seconds
        short_enough = segment.end - previous.start <= max_block_seconds

        if same_speaker and close_enough and short_enough:
            previous.end = max(previous.end, segment.end)
            previous.text = f"{previous.text} {segment.text}".strip()
        else:
            merged.append(segment)

    return merged


def make_markdown(data: dict[str, Any], blocks: list[TranscriptBlock]) -> str:
    duration = float(data.get("durationSeconds") or 0)
    participants = sorted({block.speaker for block in blocks if block.speaker})

    lines = [
        "# Transcript",
        "",
        f"Duration: {seconds_to_stamp(duration)}",
    ]

    if participants:
        lines.append(f"Participants: {', '.join(participants)}")

    lines.append("")

    for block in blocks:
        stamp = f"{seconds_to_stamp(block.start)}-{seconds_to_stamp(block.end)}"
        speaker = f"{block.speaker}: " if block.speaker else ""
        lines.append(f"[{stamp}] {speaker}{block.text}")

    return "\n".join(lines).strip() + "\n"


def make_compact_json(data: dict[str, Any], blocks: list[TranscriptBlock]) -> dict[str, Any]:
    duration = float(data.get("durationSeconds") or 0)
    participants = sorted({block.speaker for block in blocks if block.speaker})
    lines = []

    for block in blocks:
        stamp = f"{seconds_to_stamp(block.start)}-{seconds_to_stamp(block.end)}"
        speaker = f"{block.speaker}: " if block.speaker else ""
        lines.append(f"[{stamp}] {speaker}{block.text}")

    return {
        "duration_seconds": duration,
        "duration": seconds_to_stamp(duration),
        "participants": participants,
        "block_count": len(blocks),
        "text": "\n".join(lines),
        "blocks": [
            {
                "t": f"{seconds_to_stamp(block.start)}-{seconds_to_stamp(block.end)}",
                "start": round(block.start, 2),
                "end": round(block.end, 2),
                "speaker": block.speaker,
                "text": block.text,
            }
            for block in blocks
        ],
    }


def main() -> None:
    args = parse_args()
    if args.json_only and args.md_only:
        raise SystemExit("Use either --json-only or --md-only, not both.")

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = read_json(input_path)
    segments = extract_segments(data)
    blocks = merge_segments(
        segments,
        max_block_seconds=args.max_block_seconds,
        max_gap_seconds=args.max_gap_seconds,
    )

    name = safe_stem(args.name or input_path.stem)
    written: list[Path] = []

    if not args.json_only:
        md_path = out_dir / f"{name}_compact.md"
        md_path.write_text(make_markdown(data, blocks), encoding="utf-8")
        written.append(md_path)

    if not args.md_only:
        json_path = out_dir / f"{name}_compact.json"
        compact = make_compact_json(data, blocks)
        json_path.write_text(
            json.dumps(compact, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written.append(json_path)

    print(f"Input segments: {len(segments)}")
    print(f"Output blocks: {len(blocks)}")
    for path in written:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
