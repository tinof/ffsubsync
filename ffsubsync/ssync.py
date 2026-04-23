import argparse
import contextlib
import locale
import sys
from pathlib import Path

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
    parser.add_argument(
        "--preflight",
        "--skip-if-synced",
        action="store_true",
        default=False,
        help="Run a fast pre-check (~2 min of audio). Skip full sync if subtitle "
        "appears already aligned. Default: off.",
    )
    return parser


def _candidate_subtitle_paths(video_path: Path, lang: str) -> list[Path]:
    stem = video_path.with_suffix("")
    # Deduplicate while preserving order so case variants don't double-match
    # on case-insensitive filesystems (e.g. macOS default HFS+).
    seen: set[str] = set()
    candidates: list[Path] = []
    for variant in (lang, lang.lower(), lang.upper()):
        p = Path(f"{stem}.{variant}.{SUBTITLE_EXT}")
        key = str(p).casefold()
        if key not in seen:
            seen.add(key)
            candidates.append(p)
    return candidates


def _find_subtitle(video_path: Path, lang: str) -> Path | None:
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
    with contextlib.suppress(Exception):
        locale.setlocale(locale.LC_ALL, "")

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
    ffsubsync_extra: list[str] = []
    if args.preflight:
        ffsubsync_extra.append("--preflight")

    ffsubsync_args = make_parser().parse_args(
        [
            str(video),
            "-i",
            str(subtitle),
            "-o",
            str(output),
            "--output-encoding",
            "same",
            *ffsubsync_extra,
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
