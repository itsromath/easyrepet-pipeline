from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict

import requests


AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mkv",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}


def format_timestamp(seconds: float | int | None) -> str:
    if seconds is None:
        return "00:00:00"

    total = max(0, int(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_audio_duration_seconds(audio_path: Path, whisper_config: Dict[str, Any]) -> float | None:
    ffprobe_path = find_ffmpeg_tool(whisper_config, "ffprobe")
    if not ffprobe_path:
        return None

    result = subprocess.run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None

    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def find_ffmpeg_tool(whisper_config: Dict[str, Any], tool_name: str) -> str | None:
    configured = str(whisper_config.get(f"{tool_name}_path", "")).strip()
    if configured:
        path = Path(configured)
        if path.exists():
            return str(path)
        return configured

    if tool_name == "ffprobe":
        ffmpeg_path = str(whisper_config.get("ffmpeg_path", "")).strip()
        if ffmpeg_path:
            ffmpeg_file = Path(ffmpeg_path)
            sibling = ffmpeg_file.with_name("ffprobe.exe" if ffmpeg_file.suffix.lower() == ".exe" else "ffprobe")
            if sibling.exists():
                return str(sibling)

    found = shutil.which(tool_name)
    if found:
        return found

    if tool_name == "ffmpeg":
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            return None

    return None


def prepare_audio_chunks(audio_path: Path, whisper_config: Dict[str, Any]) -> tuple[tempfile.TemporaryDirectory[str] | None, list[tuple[Path, float]]]:
    chunk_config = whisper_config.get("chunking", {})
    if not isinstance(chunk_config, dict) or not chunk_config.get("enabled", False):
        return None, [(audio_path, 0.0)]

    ffmpeg_path = find_ffmpeg_tool(whisper_config, "ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            "Для нарезки аудио нужен ffmpeg. Укажи whisper.ffmpeg_path в config.yaml "
            "или добавь ffmpeg.exe в PATH."
        )

    chunk_seconds = int(chunk_config.get("chunk_seconds", 1200))
    if chunk_seconds <= 0:
        return None, [(audio_path, 0.0)]

    temp_dir = tempfile.TemporaryDirectory(prefix="easyrepet_whisper_")
    temp_path = Path(temp_dir.name)
    pattern = temp_path / "chunk_%03d.wav"

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        str(pattern),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        temp_dir.cleanup()
        details = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"ffmpeg не смог подготовить аудио: {details}")

    chunks = sorted(temp_path.glob("chunk_*.wav"))
    if not chunks:
        temp_dir.cleanup()
        raise RuntimeError("ffmpeg не создал аудиофрагменты для транскрипции.")

    return temp_dir, [(chunk, index * float(chunk_seconds)) for index, chunk in enumerate(chunks)]


def make_transcript_markdown(audio_path: Path, response: Any) -> str:
    title = audio_path.stem

    if isinstance(response, str):
        text = response.strip()
        return f"# {title}\n\nИсточник: `{audio_path.name}`\n\n{text}\n"

    if not isinstance(response, dict):
        text = str(response).strip()
        return f"# {title}\n\nИсточник: `{audio_path.name}`\n\n{text}\n"

    segments = response.get("segments") or []
    if segments:
        lines = [
            f"# {title}",
            "",
            f"Источник: `{audio_path.name}`",
            "",
        ]
        language = response.get("language")
        duration = response.get("duration")
        if language or duration:
            meta = []
            if language:
                meta.append(f"язык: `{language}`")
            if duration:
                meta.append(f"длительность: `{format_timestamp(duration)}`")
            lines.extend(["; ".join(meta), ""])

        for segment in segments:
            start = format_timestamp(segment.get("start"))
            end = format_timestamp(segment.get("end"))
            text = str(segment.get("text", "")).strip()
            if text:
                lines.append(f"[{start} - {end}] {text}")

        return "\n".join(lines).strip() + "\n"

    text = str(response.get("text", "")).strip()
    return f"# {title}\n\nИсточник: `{audio_path.name}`\n\n{text}\n"


def apply_segment_offset(response: Any, offset_seconds: float) -> Any:
    if not offset_seconds or not isinstance(response, dict):
        return response

    adjusted = dict(response)
    segments = []
    for segment in response.get("segments") or []:
        adjusted_segment = dict(segment)
        for key in ("start", "end"):
            if adjusted_segment.get(key) is not None:
                adjusted_segment[key] = float(adjusted_segment[key]) + offset_seconds
        if isinstance(adjusted_segment.get("words"), list):
            words = []
            for word in adjusted_segment["words"]:
                adjusted_word = dict(word)
                for key in ("start", "end"):
                    if adjusted_word.get(key) is not None:
                        adjusted_word[key] = float(adjusted_word[key]) + offset_seconds
                words.append(adjusted_word)
            adjusted_segment["words"] = words
        segments.append(adjusted_segment)
    adjusted["segments"] = segments

    if adjusted.get("duration") is not None:
        adjusted["duration"] = float(adjusted["duration"]) + offset_seconds

    return adjusted


def merge_transcription_responses(
    responses: list[Any],
    audio_path: Path,
    duration: float | None,
) -> Any:
    if not responses:
        return {"text": "", "segments": []}

    if len(responses) == 1:
        response = responses[0]
        if isinstance(response, dict):
            merged = dict(response)
            merged.setdefault("source_file", audio_path.name)
            if duration is not None:
                merged["duration"] = duration
            return merged
        return response

    segments = []
    languages = []
    for response in responses:
        if isinstance(response, dict):
            segments.extend(response.get("segments") or [])
            if response.get("language"):
                languages.append(response["language"])
        elif str(response).strip():
            segments.append({
                "id": len(segments),
                "start": None,
                "end": None,
                "text": str(response).strip(),
            })

    for index, segment in enumerate(segments):
        segment["id"] = index

    return {
        "task": "transcribe",
        "language": languages[0] if languages else "",
        "duration": duration if duration is not None else 0,
        "text": " ".join(
            str(segment.get("text", "")).strip()
            for segment in segments
            if str(segment.get("text", "")).strip()
        ),
        "segments": segments,
        "source_file": audio_path.name,
        "chunk_count": len(responses),
    }


def segment_start(segment: Dict[str, Any]) -> float | None:
    value = segment.get("start")
    return float(value) if value is not None else None


def segment_end(segment: Dict[str, Any]) -> float | None:
    value = segment.get("end")
    return float(value) if value is not None else None


def find_transcription_gaps(response: Any, min_gap_seconds: float) -> list[Dict[str, float]]:
    if not isinstance(response, dict) or not isinstance(response.get("segments"), list):
        return []

    segments = sorted(
        (
            segment
            for segment in response["segments"]
            if segment_start(segment) is not None and segment_end(segment) is not None
        ),
        key=lambda segment: float(segment["start"]),
    )

    gaps = []
    for previous, current in zip(segments, segments[1:]):
        previous_end = segment_end(previous)
        current_start = segment_start(current)
        if previous_end is None or current_start is None:
            continue

        gap_seconds = current_start - previous_end
        if gap_seconds >= min_gap_seconds:
            gaps.append({
                "start": previous_end,
                "end": current_start,
                "duration": gap_seconds,
            })

    return gaps


def extract_audio_range(
    audio_path: Path,
    start_seconds: float,
    end_seconds: float,
    output_path: Path,
    whisper_config: Dict[str, Any],
) -> None:
    ffmpeg_path = find_ffmpeg_tool(whisper_config, "ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            "Для ремонта дыр в транскрипции нужен ffmpeg. Укажи whisper.ffmpeg_path "
            "в config.yaml или добавь ffmpeg.exe в PATH."
        )

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-to",
        f"{end_seconds:.3f}",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"ffmpeg не смог вырезать участок для ремонта: {details}")


def repair_transcription_gaps(
    audio_path: Path,
    response: Any,
    whisper_config: Dict[str, Any],
) -> Any:
    repair_config = whisper_config.get("gap_repair", {})
    if not isinstance(repair_config, dict) or not repair_config.get("enabled", True):
        return response

    if not isinstance(response, dict) or not isinstance(response.get("segments"), list):
        return response

    min_gap_seconds = float(repair_config.get("min_gap_seconds", 45))
    padding_seconds = float(repair_config.get("padding_seconds", 8))
    max_repairs = int(repair_config.get("max_repairs", 5))
    if min_gap_seconds <= 0 or max_repairs <= 0:
        return response

    gaps = find_transcription_gaps(response, min_gap_seconds)[:max_repairs]
    if not gaps:
        return response

    duration = get_audio_duration_seconds(audio_path, whisper_config)
    repaired_segments = []
    repair_reports = []

    with tempfile.TemporaryDirectory(prefix="easyrepet_gap_repair_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        for index, gap in enumerate(gaps, start=1):
            extract_start = max(0.0, gap["start"] - padding_seconds)
            extract_end = gap["end"] + padding_seconds
            if duration is not None:
                extract_end = min(duration, extract_end)
            if extract_end <= extract_start:
                continue

            gap_audio = temp_dir / f"gap_{index:02d}.wav"
            extract_audio_range(audio_path, extract_start, extract_end, gap_audio, whisper_config)

            repair_whisper_config = dict(whisper_config)
            repair_whisper_config["vad_filter"] = bool(repair_config.get("vad_filter", False))
            repair_whisper_config["response_format"] = "verbose_json"

            repaired = apply_segment_offset(
                transcribe_audio(gap_audio, repair_whisper_config),
                extract_start,
            )
            candidate_segments = []
            if isinstance(repaired, dict):
                for segment in repaired.get("segments") or []:
                    start = segment_start(segment)
                    end = segment_end(segment)
                    text = str(segment.get("text", "")).strip()
                    if start is None or end is None or not text:
                        continue
                    if end <= gap["start"] or start >= gap["end"]:
                        continue
                    segment["repaired"] = True
                    segment["repair_source"] = "gap_repair"
                    segment["repair_id"] = index
                    segment["repair_vad_filter"] = repair_whisper_config["vad_filter"]
                    candidate_segments.append(segment)

            repaired_segments.extend(candidate_segments)
            repair_reports.append({
                "repair_id": index,
                "gap_start": gap["start"],
                "gap_end": gap["end"],
                "gap_duration": gap["duration"],
                "extract_start": extract_start,
                "extract_end": extract_end,
                "inserted_segments": len(candidate_segments),
                "vad_filter": repair_whisper_config["vad_filter"],
            })

    if not repaired_segments:
        repaired_response = dict(response)
        repaired_response["gap_repairs"] = repair_reports
        return repaired_response

    repair_start = min(
        start
        for start in (segment_start(segment) for segment in repaired_segments)
        if start is not None
    )
    repair_end = max(
        end
        for end in (segment_end(segment) for segment in repaired_segments)
        if end is not None
    )
    base_segments = []
    for segment in response["segments"]:
        start = segment_start(segment)
        end = segment_end(segment)
        overlaps_repair = (
            start is not None
            and end is not None
            and end > repair_start
            and start < repair_end
        )
        if not overlaps_repair:
            base_segments.append(segment)

    all_segments = base_segments + repaired_segments
    all_segments = sorted(
        all_segments,
        key=lambda segment: segment_start(segment) if segment_start(segment) is not None else 0.0,
    )
    for index, segment in enumerate(all_segments):
        segment["id"] = index

    repaired_response = dict(response)
    repaired_response["segments"] = all_segments
    repaired_response["text"] = " ".join(
        str(segment.get("text", "")).strip()
        for segment in all_segments
        if str(segment.get("text", "")).strip()
    )
    repaired_response["gap_repairs"] = repair_reports
    return repaired_response


def build_repair_report(response: Any) -> Dict[str, Any] | None:
    if not isinstance(response, dict):
        return None

    repairs = response.get("gap_repairs") or []
    if not repairs:
        return None

    repaired_segments = [
        segment
        for segment in response.get("segments", [])
        if isinstance(segment, dict) and segment.get("repaired")
    ]
    return {
        "source_file": response.get("source_file"),
        "duration": response.get("duration"),
        "repairs": repairs,
        "repaired_segments": [
            {
                "repair_id": segment.get("repair_id"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text"),
            }
            for segment in repaired_segments
        ],
        "repaired_segment_count": len(repaired_segments),
    }


def filter_suspicious_segments(response: Any, whisper_config: Dict[str, Any]) -> Any:
    filter_config = whisper_config.get("hallucination_filter", {})
    if not isinstance(filter_config, dict) or not filter_config.get("enabled", True):
        return response

    if not isinstance(response, dict) or not isinstance(response.get("segments"), list):
        return response

    max_compression_ratio = float(filter_config.get("max_compression_ratio", 6.0))
    max_consecutive_repeats = int(filter_config.get("max_consecutive_repeats", 2))
    min_repeated_chars = int(filter_config.get("min_repeated_chars", 24))

    filtered_segments = []
    dropped_segments = []
    previous_text = ""
    repeat_count = 0

    for segment in response["segments"]:
        text = str(segment.get("text", "")).strip()
        compression_ratio = segment.get("compression_ratio")
        too_compressed = (
            compression_ratio is not None
            and float(compression_ratio) > max_compression_ratio
        )

        normalized_text = " ".join(text.lower().split())
        if normalized_text and normalized_text == previous_text and len(normalized_text) >= min_repeated_chars:
            repeat_count += 1
        else:
            repeat_count = 0
            previous_text = normalized_text

        repeated_too_much = repeat_count >= max_consecutive_repeats

        if too_compressed or repeated_too_much:
            dropped_segments.append({
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": text,
                "compression_ratio": compression_ratio,
                "reason": "compression_ratio" if too_compressed else "repeated_text",
            })
            continue

        filtered_segments.append(segment)

    if not dropped_segments:
        return response

    filtered_response = dict(response)
    filtered_response["segments"] = filtered_segments
    filtered_response["text"] = " ".join(
        str(segment.get("text", "")).strip()
        for segment in filtered_segments
        if str(segment.get("text", "")).strip()
    )
    filtered_response["dropped_segments"] = dropped_segments
    return filtered_response


def normalize_short_segment_text(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower(), flags=re.UNICODE).strip()


def filter_repeated_short_segments(response: Any, whisper_config: Dict[str, Any]) -> Any:
    filter_config = whisper_config.get("short_repeat_filter", {})
    if not isinstance(filter_config, dict) or not filter_config.get("enabled", True):
        return response

    if not isinstance(response, dict) or not isinstance(response.get("segments"), list):
        return response

    short_segment_max_chars = int(filter_config.get("short_segment_max_chars", 20))
    max_repeats = int(filter_config.get("max_repeats", 2))
    min_segment_duration = float(filter_config.get("min_segment_duration_seconds", 0.05))
    window_seconds = float(filter_config.get("window_seconds", 2.0))
    drop_zero_duration_duplicates = bool(filter_config.get("drop_zero_duration_duplicates", True))

    filtered_segments = []
    dropped_segments = []
    previous_text = ""
    repeat_count = 0
    repeat_window_start: float | None = None

    for segment in response["segments"]:
        text = str(segment.get("text", "")).strip()
        normalized_text = normalize_short_segment_text(text)
        is_short = bool(normalized_text) and len(normalized_text) <= short_segment_max_chars

        start = segment_start(segment)
        end = segment_end(segment)
        duration = None
        if start is not None and end is not None:
            duration = max(0.0, float(end) - float(start))

        in_repeat_window = (
            repeat_window_start is None
            or start is None
            or abs(float(start) - repeat_window_start) <= window_seconds
        )

        if is_short and normalized_text == previous_text and in_repeat_window:
            repeat_count += 1
        elif is_short:
            previous_text = normalized_text
            repeat_count = 1
            repeat_window_start = float(start) if start is not None else None
        else:
            previous_text = ""
            repeat_count = 0
            repeat_window_start = None

        is_tiny = duration is not None and duration <= min_segment_duration
        repeated_too_much = is_short and repeat_count > max_repeats
        tiny_duplicate = (
            is_short
            and drop_zero_duration_duplicates
            and is_tiny
            and repeat_count > 1
        )

        if repeated_too_much or tiny_duplicate:
            dropped_segments.append({
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": text,
                "reason": "zero_duration_short_repeat" if tiny_duplicate else "short_repeat",
            })
            continue

        filtered_segments.append(segment)

    if not dropped_segments:
        return response

    filtered_response = dict(response)
    for index, segment in enumerate(filtered_segments):
        segment["id"] = index
    filtered_response["segments"] = filtered_segments
    filtered_response["text"] = " ".join(
        str(segment.get("text", "")).strip()
        for segment in filtered_segments
        if str(segment.get("text", "")).strip()
    )
    existing_dropped = filtered_response.get("dropped_segments")
    if not isinstance(existing_dropped, list):
        existing_dropped = []
    filtered_response["dropped_segments"] = existing_dropped + dropped_segments
    return filtered_response


def transcribe_audio(
    audio_path: Path,
    whisper_config: Dict[str, Any],
) -> Any:
    base_url = str(whisper_config.get("base_url", "http://127.0.0.1:8000/v1")).rstrip("/")
    url = f"{base_url}/audio/transcriptions"
    model = str(whisper_config.get("model", "Systran/faster-whisper-large-v3"))
    response_format = str(whisper_config.get("response_format", "verbose_json"))
    timeout_seconds = int(whisper_config.get("timeout_seconds", 7200))

    data: Dict[str, Any] = {
        "model": model,
        "response_format": response_format,
        "temperature": str(whisper_config.get("temperature", 0)),
    }

    for key in ("language", "prompt", "hotwords"):
        value = whisper_config.get(key)
        if value:
            data[key] = str(value)

    if "vad_filter" in whisper_config:
        data["vad_filter"] = "true" if whisper_config.get("vad_filter") else "false"

    with audio_path.open("rb") as file_obj:
        response = requests.post(
            url,
            data=data,
            files={"file": (audio_path.name, file_obj, "application/octet-stream")},
            timeout=(10, timeout_seconds),
        )

    response.raise_for_status()

    if response_format == "text":
        return response.text

    try:
        return response.json()
    except ValueError:
        return response.text


def transcribe_audio_to_files(
    audio_path: Path,
    transcripts_dir: Path,
    output_stem: str,
    whisper_config: Dict[str, Any],
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[Path, Path, Any]:
    duration = get_audio_duration_seconds(audio_path, whisper_config)
    temp_dir, chunks = prepare_audio_chunks(audio_path, whisper_config)

    try:
        responses = []
        total_chunks = len(chunks)
        for index, (chunk_path, offset_seconds) in enumerate(chunks, start=1):
            if progress_callback is not None:
                progress_callback(index - 1, total_chunks, chunk_path.name)
            response = transcribe_audio(chunk_path, whisper_config)
            responses.append(apply_segment_offset(response, offset_seconds))
            if progress_callback is not None:
                progress_callback(index, total_chunks, chunk_path.name)

        response = merge_transcription_responses(responses, audio_path, duration)
        if isinstance(response, dict) and chunks:
            response["chunk_count"] = len(chunks)
            response["chunk_seconds"] = int(whisper_config.get("chunking", {}).get("chunk_seconds", 0) or 0)
        filtered_response = filter_suspicious_segments(response, whisper_config)
        filtered_response = repair_transcription_gaps(audio_path, filtered_response, whisper_config)
        filtered_response = filter_repeated_short_segments(filtered_response, whisper_config)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    transcripts_dir.mkdir(parents=True, exist_ok=True)

    md_path = transcripts_dir / f"{output_stem}_whisper.md"
    json_path = transcripts_dir / f"{output_stem}_whisper.json"
    raw_json_path = transcripts_dir / f"{output_stem}_whisper_raw.json"
    repair_report_path = transcripts_dir / f"{output_stem}_whisper_repair_report.json"

    md_path.write_text(make_transcript_markdown(audio_path, filtered_response), encoding="utf-8")
    json_path.write_text(json.dumps(filtered_response, ensure_ascii=False, indent=2), encoding="utf-8")
    if filtered_response != response:
        raw_json_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    repair_report = build_repair_report(filtered_response)
    if repair_report:
        repair_report_path.write_text(json.dumps(repair_report, ensure_ascii=False, indent=2), encoding="utf-8")

    return md_path, json_path, filtered_response
