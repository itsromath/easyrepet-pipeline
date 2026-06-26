import json
import logging
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import yaml

from llm_client import OpenAICompatibleClient, load_model_preset
from report_utils import clean_teacher_report, repair_mojibake


DEFAULT_CONFIG: Dict = {
    "llm": {
        "base_url": "http://127.0.0.1:1234/v1",
        "api_key": "local-key",
    },
    "model_presets": {
        "summarizer": {
            "preset": "draft_4b",
        },
        "reviewer": {
            "preset": "final_9b",
        },
    },
    "paths": {
        "transcripts_dir": "transcripts",
        "outputs_dir": "outputs",
        "logs_dir": "logs",
        "audio_uploads_dir": "audio_uploads",
    },
    "transcript_cleaning": {
        "enabled": True,
        "remove_timestamps_for_llm": True,
        "merge_segments": True,
        "block_seconds": 180,
        "keep_block_timestamps": True,
    },
    "whisper": {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "Systran/faster-whisper-large-v3",
        "language": "ru",
        "response_format": "verbose_json",
        "temperature": 0,
        "vad_filter": True,
        "timeout_seconds": 7200,
        "ffmpeg_path": "",
        "ffprobe_path": "",
        "chunking": {
            "enabled": True,
            "chunk_seconds": 1200,
        },
        "gap_repair": {
            "enabled": True,
            "min_gap_seconds": 45,
            "padding_seconds": 8,
            "max_repairs": 5,
            "vad_filter": False,
        },
        "hallucination_filter": {
            "enabled": True,
            "max_compression_ratio": 6.0,
            "max_consecutive_repeats": 2,
            "min_repeated_chars": 24,
        },
        "short_repeat_filter": {
            "enabled": True,
            "short_segment_max_chars": 20,
            "max_repeats": 2,
            "drop_zero_duration_duplicates": True,
            "min_segment_duration_seconds": 0.05,
            "window_seconds": 2.0,
        },
    },
    "pipeline": {
        "supported_extensions": [".txt", ".md", ".json"],
        "skip_existing": True,
    },
}


def merge_config(defaults: Dict, overrides: Dict | None) -> Dict:
    merged = deepcopy(defaults)
    if not overrides:
        return merged

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Не найден config-файл: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError("config.yaml должен быть YAML-словарём верхнего уровня.")

    return merge_config(DEFAULT_CONFIG, loaded)


def dump_config(config: Dict) -> str:
    return yaml.safe_dump(
        merge_config(DEFAULT_CONFIG, config),
        allow_unicode=True,
        sort_keys=False,
        width=100,
    )


def setup_logging(logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def read_text_file(path: Path) -> str:
    """
    Читает txt/md/json.

    Для json пытается найти полезное текстовое поле. Если структура неизвестна,
    сохраняет json в человекочитаемом виде.
    """
    raw = repair_mojibake(path.read_text(encoding="utf-8", errors="replace"))

    if path.suffix.lower() != ".json":
        return raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()

    possible_keys = [
        "transcript",
        "text",
        "content",
        "body",
        "result",
        "message",
        "segments",
    ]

    if isinstance(data, dict):
        for key in possible_keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

            if isinstance(value, list):
                extracted = extract_text_from_list(value)
                if extracted.strip():
                    return extracted.strip()

    if isinstance(data, list):
        extracted = extract_text_from_list(data)
        if extracted.strip():
            return extracted.strip()

    return json.dumps(data, ensure_ascii=False, indent=2)


def extract_text_from_list(items: List) -> str:
    """
    Пытается собрать транскрипт из списка сегментов.

    Поддерживает варианты:
    [{"speaker": "Speaker 1", "text": "..."}]
    [{"role": "teacher", "content": "..."}]
    [{"start": 1.2, "end": 5.5, "text": "..."}]
    """
    lines: List[str] = []

    for item in items:
        if isinstance(item, str):
            lines.append(item)
            continue

        if not isinstance(item, dict):
            continue

        speaker = (
            item.get("speaker")
            or item.get("role")
            or item.get("author")
            or item.get("name")
            or ""
        )

        text = (
            item.get("text")
            or item.get("content")
            or item.get("body")
            or item.get("message")
            or ""
        )

        if not text:
            continue

        if speaker:
            lines.append(f"{speaker}: {text}")
        else:
            lines.append(str(text))

    return "\n".join(lines)


TIMESTAMPED_LINE_RE = re.compile(
    r"^\s*\[(?P<start>\d{1,2}:?\d{2}:\d{2}|\d{1,2}:\d{2})\s*[-–—]\s*"
    r"(?P<end>\d{1,2}:?\d{2}:\d{2}|\d{1,2}:\d{2})\]\s*(?P<text>.*)$"
)


def parse_timestamp(value: str) -> float | None:
    parts = value.split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + int(seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    except ValueError:
        return None
    return None


def format_block_timestamp(seconds: float | int | None) -> str:
    if seconds is None:
        return "00:00"

    total = max(0, int(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_llm_transcript_from_segments(
    segments: List[Dict],
    *,
    block_seconds: int = 180,
    keep_block_timestamps: bool = True,
) -> str:
    blocks: List[Dict[str, object]] = []
    current_text: List[str] = []
    block_start: float | None = None
    block_end: float | None = None

    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        start = segment.get("start")
        end = segment.get("end", start)
        start_seconds = float(start) if isinstance(start, (int, float)) else None
        end_seconds = float(end) if isinstance(end, (int, float)) else start_seconds

        if block_start is None:
            block_start = start_seconds
            block_end = end_seconds

        should_start_new_block = (
            block_seconds > 0
            and block_start is not None
            and end_seconds is not None
            and end_seconds - block_start > block_seconds
            and current_text
        )
        if should_start_new_block:
            blocks.append({"start": block_start, "end": block_end, "text": " ".join(current_text)})
            current_text = [text]
            block_start = start_seconds
            block_end = end_seconds
            continue

        current_text.append(text)
        block_end = end_seconds

    if current_text:
        blocks.append({"start": block_start, "end": block_end, "text": " ".join(current_text)})

    if not blocks:
        return ""

    output: List[str] = []
    for index, block in enumerate(blocks, start=1):
        text = str(block["text"]).strip()
        if not text:
            continue

        if keep_block_timestamps:
            output.append(
                f"[Блок {index}: {format_block_timestamp(block.get('start'))}–"
                f"{format_block_timestamp(block.get('end'))}]"
            )
        else:
            output.append(f"[Блок {index}]")
        output.append(text)
        output.append("")

    return "\n".join(output).strip()


def build_llm_transcript_from_text(text: str, cleaning_config: Dict | None = None) -> str:
    cleaning_config = cleaning_config or {}
    block_seconds = int(cleaning_config.get("block_seconds", 180))
    keep_block_timestamps = bool(cleaning_config.get("keep_block_timestamps", True))

    segments: List[Dict] = []
    plain_lines: List[str] = []

    for raw_line in repair_mojibake(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("#") or line.lower().startswith("источник:") or line.lower().startswith("язык:"):
            continue

        match = TIMESTAMPED_LINE_RE.match(line)
        if match:
            start = parse_timestamp(match.group("start"))
            end = parse_timestamp(match.group("end"))
            segment_text = match.group("text").strip()
            if segment_text:
                segments.append({"start": start, "end": end, "text": segment_text})
            continue

        plain_lines.append(line)

    if segments and bool(cleaning_config.get("merge_segments", True)):
        return build_llm_transcript_from_segments(
            segments,
            block_seconds=block_seconds,
            keep_block_timestamps=keep_block_timestamps,
        )

    if segments:
        lines = [segment["text"] for segment in segments if str(segment.get("text", "")).strip()]
        return "\n".join(lines).strip()

    return "\n".join(plain_lines).strip()


def build_llm_transcript(path: Path, raw_text: str, cleaning_config: Dict | None = None) -> str:
    cleaning_config = cleaning_config or {}
    if not cleaning_config.get("enabled", True):
        return raw_text.strip()

    if path.suffix.lower() == ".json":
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            return build_llm_transcript_from_text(raw_text, cleaning_config)

        if isinstance(data, dict) and isinstance(data.get("segments"), list):
            return build_llm_transcript_from_segments(
                data["segments"],
                block_seconds=int(cleaning_config.get("block_seconds", 180)),
                keep_block_timestamps=bool(cleaning_config.get("keep_block_timestamps", True)),
            )

    return build_llm_transcript_from_text(raw_text, cleaning_config)


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\wа-яА-ЯёЁ.-]+", "_", stem, flags=re.UNICODE)
    return stem.strip("_") or "lesson"


ProgressCallback = Callable[[str, str, str, str], None]


class LessonPipeline:
    def __init__(self, config: Dict, progress_callback: ProgressCallback | None = None) -> None:
        self.config = config
        self.progress_callback = progress_callback

        paths = config["paths"]
        self.transcripts_dir = Path(paths["transcripts_dir"])
        self.outputs_dir = Path(paths["outputs_dir"])
        self.logs_dir = Path(paths["logs_dir"])

        self.supported_extensions = set(
            ext.lower() for ext in config["pipeline"]["supported_extensions"]
        )
        self.skip_existing = bool(config["pipeline"].get("skip_existing", True))
        self.transcript_cleaning = config.get("transcript_cleaning", {})

        llm_config = config["llm"]
        self.client = OpenAICompatibleClient(
            base_url=llm_config["base_url"],
            api_key=llm_config.get("api_key", "local-key"),
        )

        preset_config = config.get("model_presets") or {}

        def role_preset_name(role: str, default: str) -> str:
            value = preset_config.get(role, default)
            if isinstance(value, dict):
                value = value.get("preset", default)
            return str(value)

        self.summarizer_preset_name = role_preset_name("summarizer", "draft_4b")
        self.reviewer_preset_name = role_preset_name("reviewer", "final_9b")
        self.summarizer_preset = load_model_preset(self.summarizer_preset_name)
        self.reviewer_preset = load_model_preset(self.reviewer_preset_name)

    def emit_progress(
        self,
        stage: str,
        label: str,
        detail: str = "",
        service: str = "",
    ) -> None:
        if self.progress_callback is not None:
            self.progress_callback(stage, label, detail, service)

    def ensure_dirs(self) -> None:
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def iter_transcripts(self) -> Iterable[Path]:
        for path in sorted(self.transcripts_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in self.supported_extensions:
                yield path

    def process_new_files(self, *, force: bool = False) -> None:
        self.ensure_dirs()

        files = list(self.iter_transcripts())
        if not files:
            logging.info("В папке transcripts нет подходящих файлов.")
            return

        for transcript_path in files:
            try:
                self.process_file(transcript_path, force=force)
            except Exception:
                logging.exception("Ошибка при обработке файла: %s", transcript_path)

    def watch(self, *, interval: float = 5.0, force: bool = False) -> None:
        self.ensure_dirs()
        logging.info("Режим наблюдения запущен. Папка: %s", self.transcripts_dir)

        seen: set[str] = set()

        while True:
            for transcript_path in self.iter_transcripts():
                key = str(transcript_path.resolve())

                if key in seen and not force:
                    continue

                final_path = self.get_output_paths(transcript_path)["final"]
                if self.skip_existing and final_path.exists() and not force:
                    seen.add(key)
                    continue

                try:
                    self.process_file(transcript_path, force=force)
                    seen.add(key)
                except Exception:
                    logging.exception("Ошибка при обработке файла: %s", transcript_path)

            time.sleep(interval)

    def get_output_paths(self, transcript_path: Path) -> Dict[str, Path]:
        stem = safe_stem(transcript_path.name)
        lesson_dir = self.outputs_dir / stem

        return {
            "dir": lesson_dir,
            "summary": lesson_dir / f"{stem}_summary.md",
            "review": lesson_dir / f"{stem}_review.md",
            "final": lesson_dir / f"{stem}_final.md",
            "source_copy": lesson_dir / f"{stem}_transcript.txt",
            "llm_transcript": lesson_dir / f"{stem}_llm_transcript.txt",
        }

    def process_file(self, transcript_path: Path, *, force: bool = False) -> None:
        output_paths = self.get_output_paths(transcript_path)
        final_path = output_paths["final"]

        if self.skip_existing and final_path.exists() and not force:
            logging.info("Файл уже обработан, пропускаю: %s", transcript_path.name)
            self.emit_progress(
                "done",
                "Готово",
                f"Файл уже обработан: {transcript_path.name}",
                "pipeline",
            )
            return

        logging.info("Обрабатываю транскрипт: %s", transcript_path)
        self.emit_progress(
            "report_prepare",
            "Подготовка отчёта",
            transcript_path.name,
            "pipeline",
        )

        transcript = read_text_file(transcript_path)

        if not transcript.strip():
            logging.warning("Транскрипт пустой, пропускаю: %s", transcript_path)
            self.emit_progress(
                "done",
                "Готово",
                f"Пустой транскрипт пропущен: {transcript_path.name}",
                "pipeline",
            )
            return

        llm_transcript = build_llm_transcript(transcript_path, transcript, self.transcript_cleaning)
        if not llm_transcript.strip():
            llm_transcript = transcript

        output_paths["dir"].mkdir(parents=True, exist_ok=True)
        output_paths["source_copy"].write_text(transcript, encoding="utf-8")
        output_paths["llm_transcript"].write_text(llm_transcript, encoding="utf-8")
        if llm_transcript != transcript:
            logging.info(
                "LLM-транскрипт очищен: %s -> %s символов",
                len(transcript),
                len(llm_transcript),
            )

        self.emit_progress(
            "llm_4b",
            "LLM 4B",
            f"Создаёт черновой отчёт: {self.summarizer_preset_name} / {self.summarizer_preset['model']}",
            "lmstudio",
        )
        summary = self.make_summary(llm_transcript)
        output_paths["summary"].write_text(summary, encoding="utf-8")
        logging.info("Суммаризация сохранена: %s", output_paths["summary"])

        self.emit_progress(
            "llm_9b",
            "LLM 9B",
            f"Создаёт финальный отчёт: {self.reviewer_preset_name} / {self.reviewer_preset['model']}",
            "lmstudio",
        )
        review = self.review_summary(
            llm_transcript,
            summary,
            lesson_context=f"Файл транскрипции: {transcript_path.name}",
        )
        output_paths["review"].write_text(review, encoding="utf-8")
        logging.info("Ответ финальной модели сохранён: %s", output_paths["review"])

        self.emit_progress(
            "final_report",
            "Финальный отчёт",
            "Собираю итоговый Markdown",
            "pipeline",
        )
        final_report = self.build_final_report(
            transcript_path=transcript_path,
            summary=summary,
            review=review,
        )
        output_paths["final"].write_text(final_report, encoding="utf-8")
        logging.info("Финальный отчёт сохранён: %s", output_paths["final"])

    def make_summary(self, transcript: str) -> str:
        """
        Черновой отчёт создаётся через внутренний пресет draft_4b.
        """
        user_prompt = f"""\
Ниже дана транскрипция учебного занятия.

Составь отчёт по занятию согласно системной инструкции, заданной для модели.

ТРАНСКРИПЦИЯ:
\"\"\"
{transcript}
\"\"\"
"""

        return self.client.chat_with_preset(self.summarizer_preset_name, user_prompt)

    def review_summary(
        self,
        transcript: str,
        summary: str,
        *,
        student_card: str = "",
        lesson_context: str = "",
    ) -> str:
        """
        Финальный отчёт создаётся через внутренний пресет final_9b.
        """
        student_card_text = student_card.strip() or "Карточка ученика не выбрана."
        lesson_context_text = lesson_context.strip() or "Дополнительный контекст занятия не указан."

        user_prompt = f"""\
# Карточка ученика

{student_card_text}

# Контекст занятия

{lesson_context_text}

# Транскрипция

{transcript}

# Черновой отчёт

{summary}

Создай финальный отчёт по системной инструкции.
"""

        return self.client.chat_with_preset(self.reviewer_preset_name, user_prompt)

    def build_final_report(
        self,
        *,
        transcript_path: Path,
        summary: str,
        review: str,
    ) -> str:
        teacher_report = clean_teacher_report(review)
        if len(teacher_report.strip()) < 200:
            teacher_report = clean_teacher_report(summary)

        return teacher_report
