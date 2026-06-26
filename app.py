from __future__ import annotations

import json
import os
import secrets
import shutil
import threading
import html
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, abort, redirect, render_template, request, session, url_for
from flask import jsonify
import requests
import yaml

from llm_client import call_lmstudio_chat, load_model_preset
from pipeline import LessonPipeline, dump_config, load_config, merge_config, DEFAULT_CONFIG, setup_logging
from prepare_anythingllm_transcript import (
    extract_segments,
    make_compact_json,
    make_markdown,
    merge_segments,
    read_json,
    safe_stem as safe_transcript_stem,
)
from report_utils import (
    build_html_diff,
    clean_teacher_report,
    compose_markdown_sections,
    render_markdown,
    repair_mojibake,
    split_markdown_sections,
)
from whisper_client import AUDIO_EXTENSIONS, find_ffmpeg_tool, get_audio_duration_seconds, transcribe_audio_to_files


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"

app = Flask(__name__)
app.secret_key = os.environ.get("EASYREPET_SECRET_KEY") or secrets.token_hex(32)

task_lock = threading.Lock()
readiness_lock = threading.Lock()
readiness_cache: Dict[str, object] = {"checked_at": 0.0, "data": None}
current_task: Dict[str, object] = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "target": None,
    "force": False,
    "status": "Готов к работе",
    "stage": "idle",
    "stage_label": "Готово",
    "stage_detail": "Можно запускать обработку.",
    "service": "idle",
    "workflow": "idle",
    "stage_started_at": None,
    "progress": 0,
    "stage_seconds": None,
    "stage_seconds_stage": None,
    "error": None,
}


def csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return str(token)


@app.context_processor
def inject_csrf_token():
    return {"csrf_token": csrf_token}


@app.before_request
def validate_csrf_token() -> None:
    if request.method != "POST":
        return

    expected = session.get("csrf_token")
    submitted = request.form.get("csrf_token")
    if not expected or not submitted or not secrets.compare_digest(str(expected), str(submitted)):
        abort(400, "Invalid CSRF token")

NAME_STOP_WORDS = {
    "audio",
    "video",
    "videoplayback",
    "recording",
    "record",
    "lesson",
    "copy",
    "копия",
    "запись",
    "занятия",
    "урок",
    "аудио",
    "видео",
}

WORKFLOW_STEPS = (
    {"id": "whisper", "label": "Whisper"},
    {"id": "llm_4b", "label": "LLM 4B"},
    {"id": "llm_9b", "label": "LLM 9B"},
)

PROGRESS_RANGES = {
    "idle": (0, 0, 1),
    "audio_prepare": (0, 4, 12),
    "whisper": (4, 36, 600),
    "report_prepare": (36, 40, 18),
    "llm_4b": (40, 68, 150),
    "llm_9b": (68, 94, 180),
    "final_report": (94, 99, 20),
    "import": (8, 98, 45),
    "done": (100, 100, 1),
    "error": (0, 0, 1),
}

ANYTHINGLLM_RECORDINGS_DIR = (
    Path.home()
    / "AppData"
    / "Roaming"
    / "anythingllm-desktop"
    / "storage"
    / "meeting-assistant"
)
DOWNLOADS_DIR = Path.home() / "Downloads"


def make_pipeline(progress_callback=None) -> LessonPipeline:
    config = load_config(CONFIG_PATH)
    setup_logging(BASE_DIR / config["paths"]["logs_dir"])
    pipeline = LessonPipeline(config, progress_callback=progress_callback)
    pipeline.ensure_dirs()
    ensure_students_dir()
    return pipeline


def get_paths() -> Dict[str, Path]:
    config = load_config(CONFIG_PATH)
    paths = config["paths"]
    return {
        "transcripts": BASE_DIR / paths["transcripts_dir"],
        "outputs": BASE_DIR / paths["outputs_dir"],
        "logs": BASE_DIR / paths["logs_dir"],
        "audio_uploads": BASE_DIR / paths.get("audio_uploads_dir", "audio_uploads"),
        "students": BASE_DIR / "students",
    }


def ensure_students_dir() -> Path:
    students_dir = get_paths()["students"]
    students_dir.mkdir(parents=True, exist_ok=True)
    return students_dir


def ensure_student_proposals_dir() -> Path:
    proposals_dir = ensure_students_dir() / "_proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)
    return proposals_dir


def proposal_path_for_student(filename: str) -> Path:
    filename = safe_student_name(filename)
    return ensure_student_proposals_dir() / filename


def is_test_mode_enabled() -> bool:
    return session.get("test_mode") == "1"


def safe_name(name: str) -> str:
    return Path(name).name


def short_transcript_stem(source_name: str, *, max_words: int = 4) -> str:
    stem = Path(source_name).stem
    stem = re.sub(r"^\d{8}[_-]\d{6}[_-]?", "", stem)
    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", stem, flags=re.UNICODE)
    selected = []
    for word in words:
        normalized = word.lower()
        if normalized in NAME_STOP_WORDS:
            continue
        selected.append(word)
        if len(selected) >= max_words:
            break
    if not selected:
        selected = words[:max_words] or ["transcript"]
    return safe_transcript_stem("_".join(selected))


def build_readiness_snapshot(force: bool = False) -> Dict[str, object]:
    with readiness_lock:
        cached = readiness_cache.get("data")
        checked_at = float(readiness_cache.get("checked_at") or 0.0)
        if not force and cached is not None and now_ts() - checked_at < 15:
            return dict(cached)

    checks = []

    def add_check(check_id: str, label: str, ok: bool, detail: str = "") -> None:
        checks.append({
            "id": check_id,
            "label": label,
            "ok": bool(ok),
            "detail": detail,
        })

    config = None
    try:
        config = load_config(CONFIG_PATH)
        add_check("config", "config.yaml", True, "Конфигурация загружена")
    except Exception as exc:
        add_check("config", "config.yaml", False, str(exc))

    if config is not None:
        for key, path in get_paths().items():
            try:
                path.mkdir(parents=True, exist_ok=True)
                test_path = path / ".easyrepet_write_test"
                test_path.write_text("ok", encoding="utf-8")
                test_path.unlink(missing_ok=True)
                add_check(f"path_{key}", f"Папка {key}", True, str(path))
            except Exception as exc:
                add_check(f"path_{key}", f"Папка {key}", False, f"{path}: {exc}")

        try:
            load_model_preset(model_preset_name(config, "summarizer", "draft_4b"))
            load_model_preset(model_preset_name(config, "reviewer", "final_9b"))
            add_check("llm_presets", "Пресеты LLM", True, "draft/final пресеты найдены")
        except Exception as exc:
            add_check("llm_presets", "Пресеты LLM", False, str(exc))

        llm_config = config.get("llm", {})
        llm_base_url = str(llm_config.get("base_url", "") if isinstance(llm_config, dict) else "").rstrip("/")
        if llm_base_url:
            try:
                response = requests.get(f"{llm_base_url}/models", timeout=(0.4, 1.2))
                add_check(
                    "lmstudio",
                    "LM Studio",
                    200 <= response.status_code < 300,
                    f"{llm_base_url}/models -> HTTP {response.status_code}",
                )
            except Exception as exc:
                add_check("lmstudio", "LM Studio", False, f"{llm_base_url}: {exc}")
        else:
            add_check("lmstudio", "LM Studio", False, "llm.base_url не указан")

        whisper_config = dict(config.get("whisper", {}))
        base_url = str(whisper_config.get("base_url", "")).rstrip("/")
        if base_url:
            try:
                response = requests.get(f"{base_url}/models", timeout=(0.4, 1.2))
                add_check(
                    "speaches",
                    "Speaches",
                    200 <= response.status_code < 300,
                    f"{base_url}/models -> HTTP {response.status_code}",
                )
            except Exception as exc:
                add_check("speaches", "Speaches", False, f"{base_url}: {exc}")
        else:
            add_check("speaches", "Speaches", False, "whisper.base_url не указан")

        chunking = whisper_config.get("chunking", {})
        needs_ffmpeg = isinstance(chunking, dict) and bool(chunking.get("enabled", False))
        if needs_ffmpeg:
            ffmpeg = find_ffmpeg_tool(whisper_config, "ffmpeg")
            ffprobe = find_ffmpeg_tool(whisper_config, "ffprobe")
            add_check("ffmpeg", "ffmpeg", bool(ffmpeg), ffmpeg or "ffmpeg не найден")
            add_check(
                "ffprobe",
                "ffprobe",
                True,
                ffprobe or "ffprobe не найден; длительность аудио будет оцениваться приблизительно",
            )

    issues = [check["detail"] or check["label"] for check in checks if not check["ok"]]
    ready = not issues
    snapshot = {
        "ready": ready,
        "label": "Готов к работе (все системы работают)" if ready else "Не готово",
        "checks": checks,
        "issues": issues,
    }
    with readiness_lock:
        readiness_cache["checked_at"] = now_ts()
        readiness_cache["data"] = dict(snapshot)
    return snapshot


def list_anythingllm_recordings() -> list[Dict[str, object]]:
    try:
        if not ANYTHINGLLM_RECORDINGS_DIR.exists():
            return []
    except OSError:
        return []

    recordings = []
    try:
        recording_paths = sorted(
            ANYTHINGLLM_RECORDINGS_DIR.glob("*/master-recording.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for path in recording_paths:
            stat = path.stat()
            recordings.append({
                "id": path.parent.name,
                "path": str(path),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            })
    except OSError:
        return []

    return recordings


def resolve_anythingllm_recording(recording_id: str) -> Path | None:
    recording_id = safe_name(recording_id)
    if not recording_id:
        return None

    path = ANYTHINGLLM_RECORDINGS_DIR / recording_id / "master-recording.json"
    if not path.exists() or not path.is_file():
        return None

    try:
        path.resolve().relative_to(ANYTHINGLLM_RECORDINGS_DIR.resolve())
    except ValueError:
        return None

    return path


def import_anythingllm_recording(recording_id: str, output_name: str = "") -> tuple[Path, Path, int, int]:
    source_path = resolve_anythingllm_recording(recording_id)
    if source_path is None:
        raise FileNotFoundError("AnythingLLM recording was not found.")

    data = read_json(source_path)
    segments = extract_segments(data)
    blocks = merge_segments(segments, max_block_seconds=90.0, max_gap_seconds=4.0)

    name = safe_transcript_stem(output_name) if output_name else short_transcript_stem(source_path.parent.name)
    transcripts_dir = get_paths()["transcripts"]
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    md_path = transcripts_dir / f"{name}_compact.md"
    json_path = transcripts_dir / f"{name}_compact.json"

    md_path.write_text(make_markdown(data, blocks), encoding="utf-8")
    json_path.write_text(
        json.dumps(make_compact_json(data, blocks), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return md_path, json_path, len(segments), len(blocks)


def list_download_audio_files(limit: int = 30) -> list[Dict[str, object]]:
    try:
        if not DOWNLOADS_DIR.exists():
            return []
    except OSError:
        return []

    files = []
    try:
        for path in DOWNLOADS_DIR.iterdir():
            if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue

            stat = path.stat()
            files.append({
                "name": path.name,
                "path": str(path),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            })
    except OSError:
        return []

    files.sort(key=lambda item: item["modified"], reverse=True)
    return files[:limit]


def resolve_download_audio(path_text: str) -> Path | None:
    if not path_text:
        return None

    path = Path(path_text)
    if not path.is_absolute():
        return None

    try:
        resolved = path.resolve()
        resolved.relative_to(DOWNLOADS_DIR.resolve())
    except (OSError, ValueError):
        return None

    if not resolved.is_file() or resolved.suffix.lower() not in AUDIO_EXTENSIONS:
        return None

    return resolved


def unique_audio_upload_path(filename: str) -> Path:
    uploads_dir = get_paths()["audio_uploads"]
    uploads_dir.mkdir(parents=True, exist_ok=True)

    source_name = safe_name(filename)
    stem = safe_transcript_stem(Path(source_name).stem)
    suffix = Path(source_name).suffix.lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return uploads_dir / f"{timestamp}_{stem}{suffix}"


def model_preset_name(config: Dict[str, object], role: str, default: str) -> str:
    preset_config = config.get("model_presets") or {}
    if not isinstance(preset_config, dict):
        return default
    value = preset_config.get(role, default)
    if isinstance(value, dict):
        value = value.get("preset", default)
    return str(value or default)


def readiness_check_ok(readiness: Dict[str, object], check_id: str) -> bool:
    checks = readiness.get("checks")
    if not isinstance(checks, list):
        return False
    for check in checks:
        if isinstance(check, dict) and check.get("id") == check_id:
            return bool(check.get("ok"))
    return False


def can_analyze(readiness: Dict[str, object]) -> bool:
    return readiness_check_ok(readiness, "llm_presets") and readiness_check_ok(readiness, "lmstudio")


def can_transcribe(readiness: Dict[str, object]) -> bool:
    return readiness_check_ok(readiness, "speaches") and readiness_check_ok(readiness, "ffmpeg")


def block_task(status: str, detail: str, *, service: str, workflow: str) -> None:
    with task_lock:
        apply_task_update({
            "running": False,
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "stage": "error",
            "stage_label": "Не готово",
            "stage_detail": detail,
            "service": service,
            "workflow": workflow,
            "error": detail,
        })


def get_whisper_config() -> Dict[str, object]:
    config = load_config(CONFIG_PATH)
    return dict(config.get("whisper", {}))


def now_ts() -> float:
    return datetime.now().timestamp()


def parse_task_datetime(value: object) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def format_task_datetime(value: object) -> str:
    parsed = parse_task_datetime(value)
    if parsed is None:
        return "—"
    return parsed.strftime("%d.%m.%Y %H:%M:%S")


def minute_word(minutes: int) -> str:
    if minutes % 10 == 1 and minutes % 100 != 11:
        return "минута"
    if minutes % 10 in {2, 3, 4} and minutes % 100 not in {12, 13, 14}:
        return "минуты"
    return "минут"


def format_elapsed_time(task: Dict[str, object]) -> str:
    started = parse_task_datetime(task.get("started_at"))
    if started is None:
        return "—"

    finished = parse_task_datetime(task.get("finished_at"))
    end = finished or datetime.now()
    total_seconds = max(0, int((end - started).total_seconds()))
    minutes = total_seconds // 60
    if minutes < 1:
        return "меньше минуты"
    return f"{minutes} {minute_word(minutes)}"


def estimate_whisper_seconds(duration_seconds: float | None) -> float:
    if not duration_seconds or duration_seconds <= 0:
        return float(PROGRESS_RANGES["whisper"][2])

    return max(90.0, min(2400.0, float(duration_seconds) * 0.25))


def apply_task_update(fields: Dict[str, object]) -> None:
    fields = dict(fields)
    stage = fields.get("stage")
    is_new_run = fields.get("running") is True and current_task.get("running") is not True
    if stage == "done":
        fields["progress"] = 100
    if is_new_run and "progress" not in fields:
        fields["progress"] = 0

    if stage == "error" and "progress" not in fields:
        fields["progress"] = estimate_task_progress(dict(current_task))
    if stage is not None and stage != current_task.get("stage"):
        if "progress" not in fields:
            fields["progress"] = 0 if is_new_run else estimate_task_progress(dict(current_task))
        fields["stage_started_at"] = now_ts()
    current_task.update(fields)


def estimate_task_progress(task: Dict[str, object]) -> int:
    stage = str(task.get("stage") or "idle")
    workflow = str(task.get("workflow") or "idle")

    if stage == "done":
        return 100

    if stage == "error":
        return int(task.get("progress") or 0)

    start, end, seconds = PROGRESS_RANGES.get(stage, PROGRESS_RANGES["idle"])
    if start == end:
        return start

    if task.get("stage_seconds_stage") == stage:
        try:
            seconds = max(float(task.get("stage_seconds")), 1.0)
        except (TypeError, ValueError):
            pass

    try:
        stored_progress = float(task.get("progress"))
    except (TypeError, ValueError):
        stored_progress = None

    if stored_progress is not None and stage not in {"idle", "done"}:
        start = min(max(stored_progress, 0.0), float(end) - 1.0)

    stage_started_at = task.get("stage_started_at")
    try:
        elapsed = max(0.0, now_ts() - float(stage_started_at))
    except (TypeError, ValueError):
        elapsed = 0.0

    ratio = min(0.96, elapsed / max(float(seconds), 1.0))
    return int(round(start + ((end - start) * ratio)))


def build_workflow_steps(task: Dict[str, object]) -> list[Dict[str, str]]:
    stage = str(task.get("stage") or "idle")
    workflow = str(task.get("workflow") or "idle")

    completed: set[str] = set()
    active: str | None = None

    if workflow == "audio_only":
        if stage in {"audio_prepare", "whisper"}:
            active = "whisper"
        elif stage == "done":
            completed.add("whisper")
    elif workflow in {"audio_report", "report"}:
        if workflow == "report":
            completed.add("whisper")

        if stage in {"audio_prepare", "whisper"}:
            active = "whisper"
        elif stage == "report_prepare":
            completed.add("whisper")
            active = "llm_4b"
        elif stage == "llm_4b":
            completed.add("whisper")
            active = "llm_4b"
        elif stage == "llm_9b":
            completed.update({"whisper", "llm_4b"})
            active = "llm_9b"
        elif stage == "final_report":
            completed.update({"whisper", "llm_4b", "llm_9b"})
        elif stage == "done":
            completed.update({"whisper", "llm_4b", "llm_9b"})
    elif workflow == "import":
        if stage == "import":
            active = "whisper"
        elif stage == "done":
            completed.add("whisper")

    if stage == "error" and active:
        active = None

    steps = []
    for step in WORKFLOW_STEPS:
        step_id = step["id"]
        state = "pending"
        if step_id in completed:
            state = "done"
        elif step_id == active:
            state = "active"
        steps.append({
            "id": step_id,
            "label": step["label"],
            "state": state,
        })

    return steps


def task_snapshot() -> Dict[str, object]:
    with task_lock:
        data = dict(current_task)

    data["workflow_steps"] = build_workflow_steps(data)
    data["started_display"] = format_task_datetime(data.get("started_at"))
    data["finished_display"] = format_task_datetime(data.get("finished_at"))
    data["elapsed_display"] = format_elapsed_time(data)
    readiness = build_readiness_snapshot()
    data["readiness"] = readiness
    data["can_analyze"] = can_analyze(readiness)
    data["can_transcribe"] = can_transcribe(readiness)
    if not data.get("running") and data.get("stage") == "idle":
        data["status"] = readiness["label"]
        data["stage_label"] = "Готово" if readiness["ready"] else "Не готово"
        data["stage_detail"] = (
            "Можно запускать обработку."
            if readiness["ready"]
            else "; ".join(str(issue) for issue in readiness["issues"])
        )
    return data


def update_task_stage(
    stage: str,
    label: str,
    detail: str = "",
    service: str = "",
    **extra: object,
) -> None:
    with task_lock:
        apply_task_update({
            "stage": stage,
            "stage_label": label,
            "stage_detail": detail,
            "service": service,
            **extra,
        })


def make_task_progress_callback(default_target: str | None = None):
    def callback(stage: str, label: str, detail: str = "", service: str = "") -> None:
        with task_lock:
            if default_target and not current_task.get("target"):
                current_task["target"] = default_target
            apply_task_update({
                "stage": stage,
                "stage_label": label,
                "stage_detail": detail,
                "service": service,
                "status": detail or label,
            })

    return callback


def run_audio_transcription(
    audio_path_text: str,
    output_name: str,
    process_after: bool,
    force: bool,
    create_test_student: bool = False,
) -> None:
    audio_path = Path(audio_path_text)
    output_stem = safe_transcript_stem(output_name) if output_name else short_transcript_stem(audio_path.stem)

    with task_lock:
        apply_task_update({
            "running": True,
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "finished_at": None,
            "target": audio_path.name,
            "force": force,
            "status": "Транскрипция аудио через Speaches запущена",
            "stage": "audio_prepare",
            "stage_label": "Подготовка аудио",
            "stage_detail": audio_path.name,
            "service": "whisper",
            "workflow": "audio_report" if process_after else "audio_only",
            "error": None,
        })

    try:
        paths = get_paths()
        whisper_config = get_whisper_config()
        audio_duration = get_audio_duration_seconds(audio_path, whisper_config)
        whisper_seconds = estimate_whisper_seconds(audio_duration)

        def whisper_progress(completed: int, total: int, chunk_name: str = "") -> None:
            if total <= 0:
                return

            completed = max(0, min(completed, total))
            progress_start, progress_end, _ = PROGRESS_RANGES["whisper"]
            progress = int(round(progress_start + ((progress_end - progress_start) * (completed / total))))
            current_chunk = min(completed + 1, total)
            if completed >= total:
                detail = f"Speaches завершает распознавание: {audio_path.name}"
            elif total > 1:
                detail = f"Speaches распознаёт фрагмент {current_chunk}/{total}: {audio_path.name}"
            else:
                detail = f"Speaches распознаёт: {audio_path.name}"

            with task_lock:
                apply_task_update({
                    "status": detail,
                    "stage": "whisper",
                    "stage_label": "Whisper",
                    "stage_detail": detail if not chunk_name else f"{detail} · {chunk_name}",
                    "service": "speaches",
                    "progress": progress,
                    "stage_seconds": whisper_seconds,
                    "stage_seconds_stage": "whisper",
                })

        with task_lock:
            apply_task_update({
                "status": f"Speaches распознаёт: {audio_path.name}",
                "stage": "whisper",
                "stage_label": "Whisper",
                "stage_detail": f"Speaches распознаёт: {audio_path.name}",
                "service": "speaches",
                "stage_seconds": whisper_seconds,
                "stage_seconds_stage": "whisper",
            })

        md_path, json_path, response = transcribe_audio_to_files(
            audio_path=audio_path,
            transcripts_dir=paths["transcripts"],
            output_stem=output_stem,
            whisper_config=whisper_config,
            progress_callback=whisper_progress,
        )

        text = response.get("text", "") if isinstance(response, dict) else str(response)
        status = f"Транскрипция готова: {md_path.name}"
        if text:
            status += f" ({len(text)} символов)"

        if process_after:
            with task_lock:
                apply_task_update({
                    "status": f"Создан {md_path.name}; запускаю отчёт",
                    "stage": "report_prepare",
                    "stage_label": "Подготовка отчёта",
                    "stage_detail": md_path.name,
                    "service": "pipeline",
            })
            pipeline = make_pipeline(make_task_progress_callback(md_path.name))
            final_path = pipeline.process_file(md_path, force=force)
            if create_test_student and final_path and final_path.exists():
                student_path = create_test_student_proposal(md_path.name, final_path)
                status += f"; тестовый ученик: {student_path.stem}"
            status += "; отчёт создан"

        with task_lock:
            apply_task_update({
                "running": False,
                "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "target": md_path.name,
                "status": f"{status}; JSON: {json_path.name}",
                "stage": "done",
                "stage_label": "Готово",
                "stage_detail": f"{md_path.name}; JSON: {json_path.name}",
                "service": "pipeline",
            })
    except Exception as exc:
        with task_lock:
            apply_task_update({
                "running": False,
                "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Ошибка транскрипции Speaches",
                "stage": "error",
                "stage_label": "Ошибка",
                "stage_detail": str(exc),
                "service": "speaches",
                "error": str(exc),
            })


def safe_student_name(name: str) -> str:
    stem = Path(name).stem.strip()
    if not stem:
        stem = "student"
    return f"{safe_name(stem)}.md"


def next_test_student_number() -> int:
    students_dir = ensure_students_dir()
    max_number = 0
    for path in students_dir.glob("*.md"):
        match = re.match(r"^Тестовый ученик\s+(\d+)", path.stem, flags=re.IGNORECASE)
        if match:
            max_number = max(max_number, int(match.group(1)))
    return max_number + 1


def test_student_name_for_source(source_name: str) -> str:
    hint = short_transcript_stem(source_name, max_words=3).replace("_", " ")
    number = next_test_student_number()
    return f"Тестовый ученик {number:03d} - {hint}"


def build_initial_test_student_card(student_name: str, source_name: str, report_path: Path | None = None) -> str:
    report_line = f"- Первый отчет: {report_path.name}" if report_path else "- Первый отчет: ожидает создания"
    return f"""# {student_name}

## Служебно

- Тип: тестовый ученик
- Источник: {source_name}
{report_line}
- Создано: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Что важно знать

Ожидает предложенного обновления по отчету. После генерации педагог проверяет diff и сохраняет только подтвержденную версию.

## Темы и навыки

## Самостоятельность

## Что дается легко

## Что дается сложно

## Реакция на подсказки

## Рекомендации для следующего занятия
"""


def create_test_student_card(source_name: str, report_path: Path | None = None) -> Path:
    students_dir = ensure_students_dir()
    student_name = test_student_name_for_source(source_name)
    filename = safe_student_name(student_name)
    path = students_dir / filename
    while path.exists():
        student_name = test_student_name_for_source(source_name)
        filename = safe_student_name(student_name)
        path = students_dir / filename
    write_markdown(path, build_initial_test_student_card(student_name, source_name, report_path))
    return path


def generate_student_knowledge_proposal(student_path: Path, report_path: Path) -> str:
    config = load_config(CONFIG_PATH)
    llm_config = config.get("llm", {})
    current_card = read_markdown(student_path) if student_path.exists() else ""
    report = clean_teacher_report(read_markdown(report_path))
    user_prompt = f"""# Текущая база ученика

{current_card}

# Отчет по занятию

{report}

Создай предложенную новую версию базы ученика. Верни только полный Markdown новой карточки.
"""
    proposed = call_lmstudio_chat(
        "student_knowledge_9b",
        user_prompt,
        base_url=str(llm_config.get("base_url", "http://127.0.0.1:1234/v1")),
        api_key=str(llm_config.get("api_key", "local-key")),
        timeout=900,
    )
    proposed = proposed.strip()
    if len(proposed) < 200 or "# " not in proposed:
        raise RuntimeError("LLM вернула пустое или слишком короткое предложение для базы ученика.")
    return proposed


def create_test_student_proposal(source_name: str, report_path: Path) -> Path:
    student_path = create_test_student_card(source_name, report_path)
    update_task_stage(
        "llm_9b",
        "LLM 9B",
        f"Готовит предложение для базы: {student_path.stem}",
        "lmstudio",
        workflow="report",
    )
    proposed = generate_student_knowledge_proposal(student_path, report_path)
    proposal_path_for_student(student_path.name).write_text(proposed + "\n", encoding="utf-8")
    return student_path


def output_file_path(folder: str, filename: str) -> Path:
    folder = safe_name(folder)
    filename = safe_name(filename)
    return get_paths()["outputs"] / folder / filename


def resolve_output_folder(folder: str) -> Path | None:
    folder = safe_name(folder)
    if not folder:
        return None

    outputs_dir = get_paths()["outputs"]
    path = outputs_dir / folder
    try:
        path.resolve().relative_to(outputs_dir.resolve())
    except (OSError, ValueError):
        return None

    if not path.exists() or not path.is_dir():
        return None

    return path


def transcript_file_path(filename: str) -> Path:
    filename = safe_name(filename)
    return get_paths()["transcripts"] / filename


def read_markdown(path: Path) -> str:
    return repair_mojibake(path.read_text(encoding="utf-8", errors="replace"))


def write_markdown(path: Path, content: str) -> None:
    path.write_text(repair_mojibake(content).strip() + "\n", encoding="utf-8")


def read_log_tail(limit: int = 300) -> str:
    log_file = get_paths()["logs"] / "pipeline.log"
    if not log_file.exists():
        return "Лог пока не создан."

    content = repair_mojibake(log_file.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(content.splitlines()[-limit:])


def run_processing(target_file: Optional[str], force: bool, create_test_student: bool = False) -> None:
    with task_lock:
        apply_task_update({
            "running": True,
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "finished_at": None,
            "target": target_file or "Все новые транскрипты",
            "force": force,
            "status": "Обработка запущена",
            "stage": "report_prepare",
            "stage_label": "Подготовка отчёта",
            "stage_detail": target_file or "Все новые транскрипты",
            "service": "pipeline",
            "workflow": "report",
            "error": None,
        })

    try:
        pipeline = make_pipeline(make_task_progress_callback(target_file))

        if target_file:
            transcript_path = pipeline.transcripts_dir / safe_name(target_file)
            with task_lock:
                current_task["status"] = f"Обрабатывается файл: {transcript_path.name}"
                current_task["stage_detail"] = transcript_path.name
            final_path = pipeline.process_file(transcript_path, force=force)
            if create_test_student and final_path and final_path.exists():
                create_test_student_proposal(transcript_path.name, final_path)
        else:
            with task_lock:
                current_task["status"] = "Обрабатываются все новые транскрипты"
                current_task["stage_detail"] = "Все новые транскрипты"
            for transcript_path in pipeline.iter_transcripts():
                final_path = pipeline.process_file(transcript_path, force=force)
                if create_test_student and final_path and final_path.exists():
                    create_test_student_proposal(transcript_path.name, final_path)

        with task_lock:
            apply_task_update({
                "running": False,
                "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Готово",
                "stage": "done",
                "stage_label": "Готово",
                "stage_detail": "Обработка завершена",
                "service": "pipeline",
            })

    except Exception as exc:
        with task_lock:
            apply_task_update({
                "running": False,
                "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Ошибка",
                "stage": "error",
                "stage_label": "Ошибка",
                "stage_detail": str(exc),
                "service": "pipeline",
                "error": str(exc),
            })


def list_transcripts(order: str = "desc") -> list[Dict[str, object]]:
    transcripts_dir = get_paths()["transcripts"]
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for path in transcripts_dir.iterdir():
        if not path.is_file() or path.name.lower() in {"readme.md", ".gitkeep"}:
            continue

        modified_ts = path.stat().st_mtime
        items.append({
            "name": path.name,
            "size": path.stat().st_size,
            "modified": datetime.fromtimestamp(modified_ts).strftime("%Y-%m-%d %H:%M:%S"),
            "modified_ts": modified_ts,
        })

    reverse = order != "asc"
    items.sort(key=lambda item: (item["modified_ts"], item["name"]), reverse=reverse)
    return items


def list_output_folders() -> list[Dict[str, object]]:
    outputs_dir = get_paths()["outputs"]
    outputs_dir.mkdir(parents=True, exist_ok=True)

    folders = []
    for folder in outputs_dir.iterdir():
        if not folder.is_dir():
            continue

        final_files = []
        work_files = []
        latest_modified_ts = folder.stat().st_mtime
        for file in folder.iterdir():
            if not file.is_file():
                continue
            stat = file.stat()
            modified_ts = stat.st_mtime
            latest_modified_ts = max(latest_modified_ts, modified_ts)
            file_data = {
                "name": file.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(modified_ts).strftime("%Y-%m-%d %H:%M:%S"),
                "modified_ts": modified_ts,
                "is_markdown": file.suffix.lower() == ".md",
                "is_final": file.name.endswith("_final.md"),
            }
            if file_data["is_final"]:
                final_files.append(file_data)
            else:
                work_files.append(file_data)

        final_files.sort(key=lambda item: (item["modified_ts"], item["name"]), reverse=True)
        work_files.sort(key=lambda item: (item["modified_ts"], item["name"]), reverse=True)
        latest_final_ts = final_files[0]["modified_ts"] if final_files else 0

        folders.append({
            "name": folder.name,
            "modified": datetime.fromtimestamp(latest_modified_ts).strftime("%Y-%m-%d %H:%M:%S"),
            "modified_ts": latest_modified_ts,
            "final_modified_ts": latest_final_ts,
            "final_files": final_files,
            "work_files": work_files,
        })

    folders.sort(
        key=lambda item: (item["final_modified_ts"] or item["modified_ts"], item["name"]),
        reverse=True,
    )
    return folders


def list_students() -> list[Dict[str, object]]:
    students_dir = ensure_students_dir()
    students = []
    for path in sorted(students_dir.glob("*.md")):
        students.append({
            "name": path.stem,
            "filename": path.name,
            "size": path.stat().st_size,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return students


@app.route("/")
def index():
    config_text = repair_mojibake(CONFIG_PATH.read_text(encoding="utf-8"))
    whisper_config = get_whisper_config()
    transcript_order = request.args.get("transcript_order", "desc")
    if transcript_order not in {"asc", "desc"}:
        transcript_order = "desc"
    return render_template(
        "index.html",
        transcripts=list_transcripts(order=transcript_order),
        outputs=list_output_folders(),
        students=list_students(),
        anythingllm_recordings=list_anythingllm_recordings(),
        download_audio_files=list_download_audio_files(),
        whisper_config=whisper_config,
        transcript_order=transcript_order,
        task=task_snapshot(),
        config_text=config_text,
        test_mode=is_test_mode_enabled(),
    )


@app.route("/favicon.ico")
def favicon():
    return app.send_static_file("favicon.svg")


@app.route("/test_mode", methods=["POST"])
def test_mode():
    session["test_mode"] = "1" if request.form.get("enabled") == "1" else "0"
    return redirect(url_for("index"))


@app.route("/upload", methods=["POST"])
def upload():
    transcripts_dir = get_paths()["transcripts"]
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    uploaded_file = request.files.get("file")
    if not uploaded_file or not uploaded_file.filename:
        return redirect(url_for("index"))

    filename = safe_name(uploaded_file.filename)
    if not filename.lower().endswith((".txt", ".md", ".json")):
        return redirect(url_for("index"))

    uploaded_file.save(transcripts_dir / filename)
    return redirect(url_for("index"))


@app.route("/transcribe_audio", methods=["POST"])
def transcribe_audio_route():
    with task_lock:
        if current_task.get("running"):
            return redirect(url_for("index"))

    process_after = request.form.get("process_after") == "on"
    create_test_student = (
        is_test_mode_enabled()
        and process_after
        and request.form.get("create_test_student") == "on"
    )
    readiness = build_readiness_snapshot(force=True)
    if process_after and not can_analyze(readiness):
        block_task(
            "Не готово",
            "LM Studio выключен или недоступен; отчёт по транскрипту не запущен.",
            service="lmstudio",
            workflow="audio_report",
        )
        return redirect(url_for("index"))
    if not can_transcribe(readiness):
        block_task(
            "Не готово",
            "Speaches или ffmpeg недоступны; транскрипция аудио не запущена.",
            service="speaches",
            workflow="audio_only",
        )
        return redirect(url_for("index"))

    uploaded_file = request.files.get("audio_file")
    audio_path = None

    if uploaded_file and uploaded_file.filename:
        filename = safe_name(uploaded_file.filename)
        if Path(filename).suffix.lower() not in AUDIO_EXTENSIONS:
            with task_lock:
                apply_task_update({
                    "status": "Файл не похож на аудио/видео для Speaches",
                    "stage": "error",
                    "stage_label": "Ошибка",
                    "stage_detail": "Формат файла не поддерживается",
                    "service": "speaches",
                    "workflow": "audio_only",
                    "error": f"Поддерживаются: {', '.join(sorted(AUDIO_EXTENSIONS))}",
                })
            return redirect(url_for("index"))

        audio_path = unique_audio_upload_path(filename)
        uploaded_file.save(audio_path)
    else:
        audio_path = resolve_download_audio(request.form.get("audio_path", ""))

    if audio_path is None:
        with task_lock:
            apply_task_update({
                "status": "Не выбран аудиофайл для Speaches",
                "stage": "idle",
                "stage_label": "Ожидает файла",
                "stage_detail": "Выберите аудио из Downloads или загрузите файл.",
                "service": "whisper",
                "workflow": "idle",
                "error": None,
            })
        return redirect(url_for("index"))

    output_name = request.form.get("output_name", "")
    force = request.form.get("force") == "on"

    thread = threading.Thread(
        target=run_audio_transcription,
        args=(str(audio_path), output_name, process_after, force, create_test_student),
        daemon=True,
    )
    thread.start()
    return redirect(url_for("index"))


@app.route("/process", methods=["POST"])
def process():
    target_file = request.form.get("target_file") or None
    force = request.form.get("force") == "on"
    create_test_student = (
        is_test_mode_enabled()
        and request.form.get("create_test_student") == "on"
    )

    with task_lock:
        if current_task.get("running"):
            return redirect(url_for("index"))

    readiness = build_readiness_snapshot(force=True)
    if not can_analyze(readiness):
        block_task(
            "Не готово",
            "LM Studio выключен или недоступен; анализ транскрипта не запущен.",
            service="lmstudio",
            workflow="report",
        )
        return redirect(url_for("index"))

    thread = threading.Thread(target=run_processing, args=(target_file, force, create_test_student), daemon=True)
    thread.start()
    return redirect(url_for("index"))


@app.route("/create_test_student_from_report", methods=["POST"])
def create_test_student_from_report():
    folder = safe_name(request.form.get("folder", ""))
    filename = safe_name(request.form.get("filename", ""))
    path = output_file_path(folder, filename)
    if not path.exists() or not path.is_file() or not filename.endswith("_final.md"):
        return redirect(url_for("index") + "#final-reports")

    readiness = build_readiness_snapshot(force=True)
    if not can_analyze(readiness):
        block_task(
            "Не готово",
            "LM Studio выключен или недоступен; предложение для базы ученика не создано.",
            service="lmstudio",
            workflow="report",
        )
        return redirect(url_for("index") + "#final-reports")

    try:
        student_path = create_test_student_proposal(folder, path)
    except Exception as exc:
        update_task_stage(
            "error",
            "Ошибка",
            str(exc),
            "lmstudio",
            workflow="report",
            error=str(exc),
        )
        return redirect(url_for("index") + "#final-reports")

    return redirect(url_for("student_card", filename=student_path.name))


@app.route("/import_anythingllm", methods=["POST"])
def import_anythingllm():
    recording_id = request.form.get("recording_id", "")
    output_name = request.form.get("output_name", "")

    with task_lock:
        if current_task.get("running"):
            return redirect(url_for("index"))

        apply_task_update({
            "running": True,
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "finished_at": None,
            "target": recording_id or "AnythingLLM",
            "force": False,
            "status": "Importing AnythingLLM transcript",
            "stage": "import",
            "stage_label": "AnythingLLM",
            "stage_detail": recording_id or "Импорт записи",
            "service": "anythingllm",
            "workflow": "import",
            "error": None,
        })

    try:
        md_path, json_path, segment_count, block_count = import_anythingllm_recording(
            recording_id,
            output_name,
        )
        with task_lock:
            apply_task_update({
                "running": False,
                "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "target": md_path.name,
                "stage": "done",
                "stage_label": "Готово",
                "stage_detail": f"{md_path.name}; {block_count} блоков",
                "service": "anythingllm",
                "status": (
                    f"AnythingLLM import done: {segment_count} segments -> "
                    f"{block_count} blocks; wrote {md_path.name} and {json_path.name}"
                ),
            })
    except Exception as exc:
        with task_lock:
            apply_task_update({
                "running": False,
                "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "AnythingLLM import error",
                "stage": "error",
                "stage_label": "Ошибка",
                "stage_detail": str(exc),
                "service": "anythingllm",
                "error": str(exc),
            })

    return redirect(url_for("index"))


@app.route("/delete_transcript", methods=["POST"])
def delete_transcript():
    filename = safe_name(request.form.get("filename", ""))
    if not filename:
        return redirect(url_for("index"))

    path = get_paths()["transcripts"] / filename
    if path.exists() and path.is_file():
        path.unlink()

    return redirect(url_for("index"))


@app.route("/delete_output_files", methods=["POST"])
def delete_output_files():
    folder_path = resolve_output_folder(request.form.get("folder", ""))
    if folder_path is None:
        return redirect(url_for("index") + "#work-files")

    for filename in request.form.getlist("filename"):
        filename = safe_name(filename)
        if not filename or filename.endswith("_final.md"):
            continue

        path = folder_path / filename
        try:
            path.resolve().relative_to(folder_path.resolve())
        except (OSError, ValueError):
            continue

        if path.exists() and path.is_file():
            path.unlink()

    return redirect(url_for("index") + "#work-files")


@app.route("/delete_output_folder", methods=["POST"])
def delete_output_folder():
    folder_path = resolve_output_folder(request.form.get("folder", ""))
    if folder_path is not None:
        shutil.rmtree(folder_path)

    return redirect(url_for("index") + "#work-files")


@app.route("/transcripts/<filename>")
def view_transcript(filename: str):
    path = transcript_file_path(filename)
    if not path.exists() or not path.is_file():
        return "Файл не найден", 404

    content = read_markdown(path)
    if path.suffix.lower() == ".md":
        rendered_content = render_markdown(content)
    else:
        rendered_content = f"<pre>{html.escape(content)}</pre>"

    return render_template(
        "view.html",
        title="Транскрипт",
        page_title="Транскрипт",
        page_subtitle=path.name,
        content=content,
        rendered_content=rendered_content,
        can_edit=False,
        is_log=False,
    )


@app.route("/view/<folder>/<filename>")
def view_output(folder: str, filename: str):
    path = output_file_path(folder, filename)
    if not path.exists() or not path.is_file():
        return "Файл не найден", 404

    content = read_markdown(path)
    if filename.endswith("_final.md"):
        content = clean_teacher_report(content)

    return render_template(
        "view.html",
        title="Отчёт по занятию" if filename.endswith("_final.md") else "Рабочий файл",
        page_title="Отчёт по занятию" if filename.endswith("_final.md") else "Рабочий файл",
        page_subtitle=safe_name(folder),
        content=content,
        rendered_content=render_markdown(content),
        can_edit=path.suffix.lower() == ".md",
        folder=safe_name(folder),
        filename=safe_name(filename),
        is_log=False,
    )


@app.route("/edit_report/<folder>/<filename>", methods=["GET", "POST"])
def edit_report(folder: str, filename: str):
    path = output_file_path(folder, filename)
    if not path.exists() or not path.is_file():
        return "Файл не найден", 404

    if request.method == "POST":
        titles = request.form.getlist("title")
        levels = request.form.getlist("level")
        bodies = request.form.getlist("body")
        sections = []
        for title, level, body in zip(titles, levels, bodies):
            sections.append({"title": title, "level": level, "body": body})
        write_markdown(path, clean_teacher_report(compose_markdown_sections(sections)))
        return redirect(url_for("edit_report", folder=safe_name(folder), filename=safe_name(filename), saved="1"))

    content = clean_teacher_report(read_markdown(path))
    sections = split_markdown_sections(content)
    return render_template(
        "edit_report.html",
        title=f"Редактор: {safe_name(folder)}/{safe_name(filename)}",
        folder=safe_name(folder),
        filename=safe_name(filename),
        sections=sections,
        saved=request.args.get("saved") == "1",
    )


@app.route("/logs")
def logs():
    content = read_log_tail()

    return render_template(
        "view.html",
        title="Логи",
        page_title="Логи",
        page_subtitle="Последние 300 строк",
        content=content,
        rendered_content=f'<pre id="logContent">{html.escape(content)}</pre>',
        can_edit=False,
        is_log=True,
    )


@app.route("/api/logs")
def api_logs():
    content = read_log_tail()
    return jsonify({"content": content})


@app.route("/config", methods=["POST"])
def save_config():
    config_text = request.form.get("config_text", "")
    if config_text.strip():
        try:
            loaded = yaml.safe_load(repair_mojibake(config_text)) or {}
            if not isinstance(loaded, dict):
                raise ValueError("config.yaml должен быть YAML-словарём верхнего уровня.")
            CONFIG_PATH.write_text(dump_config(merge_config(DEFAULT_CONFIG, loaded)), encoding="utf-8")
            update_task_stage(
                "done",
                "Config сохранён",
                "Недостающие секции YAML автоматически дополнены.",
                "config",
                workflow="idle",
                error=None,
            )
        except Exception as exc:
            update_task_stage(
                "error",
                "Ошибка config.yaml",
                str(exc),
                "config",
                workflow="idle",
                error=str(exc),
            )
    return redirect(url_for("index"))


@app.route("/students", methods=["GET", "POST"])
def students():
    if request.method == "POST":
        name = request.form.get("student_name", "").strip()
        if name:
            filename = safe_student_name(name)
            path = ensure_students_dir() / filename
            if not path.exists():
                write_markdown(path, f"# {Path(filename).stem}\n\n## Что важно знать\n\n")
            return redirect(url_for("student_card", filename=filename))

    return render_template("students.html", students=list_students())


@app.route("/students/<filename>", methods=["GET", "POST"])
def student_card(filename: str):
    filename = safe_student_name(filename)
    path = ensure_students_dir() / filename
    if not path.exists():
        write_markdown(path, f"# {Path(filename).stem}\n\n## Что важно знать\n\n")

    if request.method == "POST":
        content = request.form.get("content", "")
        write_markdown(path, content)
        return redirect(url_for("student_card", filename=filename, saved="1"))

    content = read_markdown(path)
    pending_proposal_path = proposal_path_for_student(filename)
    pending_proposal = read_markdown(pending_proposal_path) if pending_proposal_path.exists() else ""
    has_pending_proposal = len(pending_proposal.strip()) >= 200
    proposed = pending_proposal if has_pending_proposal else content
    return render_template(
        "student_edit.html",
        student_name=Path(filename).stem,
        filename=filename,
        content=content,
        proposed=proposed,
        rendered_content=render_markdown(content),
        diff_html=None,
        has_pending_proposal=has_pending_proposal,
        saved=request.args.get("saved") == "1",
    )


@app.route("/students/<filename>/diff", methods=["POST"])
def student_diff(filename: str):
    filename = safe_student_name(filename)
    path = ensure_students_dir() / filename
    old_content = read_markdown(path) if path.exists() else ""
    proposed = request.form.get("proposed", "")
    pending_proposal_path = proposal_path_for_student(filename)
    pending_proposal = read_markdown(pending_proposal_path) if pending_proposal_path.exists() else ""
    return render_template(
        "student_edit.html",
        student_name=Path(filename).stem,
        filename=filename,
        content=old_content,
        proposed=proposed,
        rendered_content=render_markdown(old_content),
        diff_html=build_html_diff(old_content, proposed),
        has_pending_proposal=len(pending_proposal.strip()) >= 200,
        saved=False,
    )


@app.route("/students/<filename>/save_proposed", methods=["POST"])
def save_student_proposed(filename: str):
    filename = safe_student_name(filename)
    path = ensure_students_dir() / filename
    proposed = request.form.get("proposed", "")
    write_markdown(path, proposed)
    pending_proposal_path = proposal_path_for_student(filename)
    if pending_proposal_path.exists():
        pending_proposal_path.unlink()
    return redirect(url_for("student_card", filename=filename, saved="1"))


@app.route("/api/status")
def api_status():
    return jsonify(task_snapshot())


if __name__ == "__main__":
    app.run(
        host=os.environ.get("EASYREPET_HOST", "127.0.0.1"),
        port=int(os.environ.get("EASYREPET_PORT", "5050")),
        debug=os.environ.get("EASYREPET_FLASK_DEBUG", "0") == "1",
        use_reloader=False,
    )
