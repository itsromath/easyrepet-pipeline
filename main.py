
import argparse
from pathlib import Path

from llm_client import test_model_presets
from pipeline import LessonPipeline, load_config, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EasyRepet MVP: транскрипт -> суммаризация -> проверка -> финальный отчёт"
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Путь к config.yaml",
    )

    parser.add_argument(
        "--file",
        default=None,
        help="Обработать конкретный файл транскрипции",
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        help="Следить за папкой transcripts и обрабатывать новые файлы",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Интервал проверки папки в секундах для режима --watch",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать результаты, даже если файл уже обработан",
    )

    parser.add_argument(
        "--test-presets",
        action="store_true",
        help="Отправить короткий тестовый запрос через draft_4b и final_9b",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    logs_dir = Path(config["paths"]["logs_dir"])
    setup_logging(logs_dir)

    if args.test_presets:
        ok = test_model_presets(
            base_url=config["llm"]["base_url"],
            api_key=config["llm"].get("api_key", "local-key"),
        )
        raise SystemExit(0 if ok else 1)

    pipeline = LessonPipeline(config)
    pipeline.ensure_dirs()

    if args.file:
        pipeline.process_file(Path(args.file), force=args.force)
        return

    if args.watch:
        pipeline.watch(interval=args.interval, force=args.force)
        return

    pipeline.process_new_files(force=args.force)


if __name__ == "__main__":
    main()
