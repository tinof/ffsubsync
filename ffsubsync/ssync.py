import argparse
import locale
import sys
from pathlib import Path
from typing import Optional

from ffsubsync.ffsubsync import make_parser, run


DEFAULT_SUB_LANG = "fin"
SUBTITLE_EXT = "srt"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-discover a language-specific subtitle file next to a video and "
            "sync it with ffsubsync."
        )
    )
    parser.add_argument("video", help="Path to the reference video file")
    parser.add_argument(
        "--lang",
        default=DEFAULT_SUB_LANG,
        help=f"Subtitle language suffix to match (default: {DEFAULT_SUB_LANG})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved paths and ffsubsync arguments without running sync",
    )
    return parser


def _candidate_subtitle_paths(video_path: Path, lang: str) -> list[Path]:
    stem = video_path.with_suffix("")
    lower_lang = lang.lower()
    return [
        Path(f"{stem}.{lang}.{SUBTITLE_EXT}"),
        Path(f"{stem}.{lower_lang}.{SUBTITLE_EXT}"),
        Path(f"{stem}.{lang.upper()}.{SUBTITLE_EXT}"),
    ]


def _find_subtitle(video_path: Path, lang: str) -> Optional[Path]:
    for candidate in _candidate_subtitle_paths(video_path, lang):
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_output_path(subtitle_path: Path) -> Path:
    # overwrite input subtitle in-place
    return subtitle_path


def _print(*args: object) -> None:
    print(*args, file=sys.stderr)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    lang = args.lang or DEFAULT_SUB_LANG
    try:
        locale.setlocale(locale.LC_ALL, "")
    except Exception:
        # Keep going with default C locale if unavailable
        pass

    video = Path(args.video)
    _print(f"Processing subtitles for {video.name} (Language: {lang})")

    if not video.exists() or not video.is_file():
        _print(f"Video file not found: {video}")
        return 1

    subtitle = _find_subtitle(video, lang)
    if subtitle is None:
        _print(f"Subtitle file for {video.stem} not found. Skipping gracefully.")
        return 0

    output = _resolve_output_path(subtitle)
    ffsubsync_args = make_parser().parse_args(
        [
            str(video),
            "-i",
            str(subtitle),
            "-o",
            str(output),
            "--output-encoding",
            "same",
        ]
    )

    if args.dry_run:
        _print(f"Reference video: {video}")
        _print(f"Input subtitle: {subtitle}")
        _print(f"Output subtitle: {output}")
        return 0

    result = run(ffsubsync_args)
    return int(result.get("retval", 1))


if __name__ == "__main__":
    raise SystemExit(main())
