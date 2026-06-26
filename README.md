# EasyRepet Pipeline MVP

Минимальная версия пайплайна:

**транскрипт -> суммаризация 4B -> проверка 9B -> финальный отчёт**

## Важная логика этой версии

EasyRepet использует внутренние пресеты моделей, а не выбранные вручную preset в интерфейсе LM Studio.

Источник истины для LLM-параметров:

```text
config/model_presets.json
prompts/draft_4b_system.txt
prompts/final_9b_system.txt
```

В каждый запрос к LM Studio API отправляются:

- `model`;
- system prompt из отдельного `.txt` файла;
- `temperature`;
- `top_p`;
- `max_tokens`.

## 1. Установка

```bash
pip install -r requirements.txt
```

## 2. Настройка моделей

Открой `config/model_presets.json` и укажи реальные названия моделей:

```json
{
  "draft_4b": {
    "model": "qwen/qwen3-4b-2507",
    "system_prompt_file": "prompts/draft_4b_system.txt",
    "temperature": 0.25,
    "top_p": 0.85,
    "max_tokens": 4096
  },
  "final_9b": {
    "model": "qwen/qwen3.5-9b",
    "system_prompt_file": "prompts/final_9b_system.txt",
    "temperature": 0.1,
    "top_p": 0.95,
    "max_tokens": 4096
  }
}
```

Также проверь адрес локального API:

```yaml
llm:
  base_url: "http://127.0.0.1:1234/v1"
```

Роли пайплайна привязаны к пресетам в `config.yaml`:

```yaml
model_presets:
  summarizer:
    preset: "draft_4b"
  reviewer:
    preset: "final_9b"
```

## 3. Как пользоваться

Положи файл транскрипта в папку:

```text
transcripts/
```

Поддерживаются:

```text
.txt
.md
.json
```

Потом запусти:

```bash
python main.py
```

Результаты появятся в:

```text
outputs/
```

Для каждого занятия будет отдельная папка с файлами:

```text
*_summary.md
*_review.md
*_final.md
*_transcript.txt
```

## 4. Обработать конкретный файл

```bash
python main.py --file transcripts/lesson_001.txt
```

## 5. Режим наблюдения за папкой

```bash
python main.py --watch
```

Теперь можно просто добавлять новые транскрипты в папку `transcripts`, и скрипт будет их обрабатывать.

## 6. Принудительная повторная обработка

```bash
python main.py --force
```

Или:

```bash
python main.py --file transcripts/lesson_001.txt --force
```

## 7. Проверка пресетов LM Studio

Короткий тестовый запрос к 4B и 9B:

```bash
python main.py --test-presets
```

Успешный ответ выглядит так:

```text
draft_4b preset: OK
final_9b preset: OK
```

## 8. Что делает код

1. Берёт транскрипт из `transcripts/`.
2. Отправляет его в модель суммаризации через пресет `draft_4b`.
3. Получает отчёт.
4. Отправляет исходный транскрипт и черновой отчёт в модель финальной редакции через пресет `final_9b`.
5. Сохраняет:
   - отчёт;
   - ответ финальной модели;
   - финальный объединённый файл;
   - копию исходного транскрипта.

## 9. Аудио через Speaches / Whisper

CLI-режим по-прежнему работает с готовыми транскриптами из `transcripts/`.

Веб-интерфейс Flask умеет отправлять аудио или видео в локальный Speaches-сервер:

```yaml
transcript_cleaning:
  enabled: true
  remove_timestamps_for_llm: true
  merge_segments: true
  block_seconds: 180
  keep_block_timestamps: true

whisper:
  base_url: "http://127.0.0.1:8000/v1"
  model: "Systran/faster-whisper-large-v3"
  vad_filter: true
  chunking:
    enabled: true
    chunk_seconds: 1200
  gap_repair:
    enabled: true
    min_gap_seconds: 45
    padding_seconds: 8
    vad_filter: false
  hallucination_filter:
    enabled: true
    max_compression_ratio: 6.0
```

На главной странице есть блок `Аудио через Speaches`: он показывает свежие `.mp3`, `.m4a`, `.mp4`, `.wav`, `.ogg`, `.opus`, `.flac`, `.webm`, `.aac`, `.mkv` из `Downloads` и также принимает ручную загрузку файла.

После распознавания создаются:

```text
transcripts/*_whisper.md
transcripts/*_whisper.json
```

Если включена галка `Сразу создать отчёт после транскрипции`, новый `.md` сразу отправляется в основной пайплайн.

При обработке отчёта пайплайн сохраняет две версии транскрипта:

```text
outputs/<lesson>/*_transcript.txt       # debug-версия с частыми таймкодами
outputs/<lesson>/*_llm_transcript.txt   # очищенная версия для LM Studio
```

`transcript_cleaning` убирает частые таймкоды из текста, который отправляется в LLM, и склеивает короткие Whisper-сегменты в крупные блоки. Таймкоды остаются только на уровне блоков, чтобы модель не читала технический лог вместо занятия.

Для длинных YouTube-записей `chunking.enabled: true` сначала перегоняет исходный MP4/MP3 в WAV 16 kHz mono и режет запись на куски по `chunk_seconds`. Каждый кусок отправляется в Speaches отдельно, затем сегменты склеиваются обратно с правильными таймкодами. Это снижает риск зацикливания Whisper на длинном предыдущем контексте.

Нарезке нужен `ffmpeg`. Проект умеет найти системный `ffmpeg.exe`, путь из `whisper.ffmpeg_path` или бинарник из пакета `imageio-ffmpeg`.

`vad_filter: true` помогает вырезать паузы и шум. Дополнительный `hallucination_filter` чистит Markdown/JSON от явно зацикленных сегментов с очень высоким `compression_ratio`; сырой ответ Speaches сохраняется рядом как `*_whisper_raw.json`, если фильтр что-то удалил.

`gap_repair` проверяет готовую транскрипцию после фильтра: если между соседними сегментами есть дыра больше `min_gap_seconds`, проект вырезает этот участок из исходного аудио с запасом `padding_seconds`, прогоняет его отдельно через Speaches без VAD и вставляет найденные сегменты обратно.

Восстановленные сегменты помечаются в JSON полями `repaired`, `repair_source`, `repair_id`, `repair_vad_filter`. Если ремонт был выполнен, рядом сохраняется `*_whisper_repair_report.json` с диапазоном дыры и списком добавленных сегментов.

## 10. Веб-интерфейс Flask

Установи зависимости:

```bash
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Запусти веб-интерфейс:

```bash
.\.venv\Scripts\python.exe app.py
```

Открой в браузере:

```text
http://127.0.0.1:5050
```

Важно: перед запуском обработки должен быть включён LM Studio Server на адресе из `config.yaml`.
