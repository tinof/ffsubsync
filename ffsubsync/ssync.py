import argparse
import contextlib
import json
import locale
import subprocess
import sys
import tempfile
from pathlib import Path

from ffsubsync.ffsubsync import make_parser, run

DEFAULT_SUB_LANG = "fin"
SUBTITLE_EXT = "srt"
PREFERRED_REFERENCE_LANGS = ("eng", "en")
LANG_ALIASES = {
    "fin": ("fin", "fi"),
    "fi": ("fi", "fin"),
}


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
    lang_roots = LANG_ALIASES.get(_normalize_lang(lang), (lang,))
    for lang_root in lang_roots:
        for variant in (lang_root.lower(), lang_root, lang_root.upper()):
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


def _normalize_lang(lang: str | None) -> str:
    return (lang or "").casefold()


def _embedded_subtitle_streams(video_path: Path) -> list[dict[str, object]]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "s",
        "-show_entries",
        "stream=index,codec_name:stream_tags=language,title",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(proc.stdout or "{}")
    except Exception:
        return []
    streams = data.get("streams", [])
    if not isinstance(streams, list):
        return []
    return [s for s in streams if isinstance(s, dict) and "index" in s]


def _stream_language(stream: dict[str, object]) -> str:
    tags = stream.get("tags", {})
    if not isinstance(tags, dict):
        return ""
    language = tags.get("language", "")
    return _normalize_lang(str(language))


def _pick_reference_subtitle_stream(
    streams: list[dict[str, object]], target_lang: str
) -> dict[str, object] | None:
    if not streams:
        return None

    target = _normalize_lang(target_lang)
    non_target = [s for s in streams if _stream_language(s) != target]
    candidates = non_target or streams
    for preferred in PREFERRED_REFERENCE_LANGS:
        for stream in candidates:
            if _stream_language(stream) == preferred:
                return stream
    return candidates[0]


def _extract_embedded_reference_subtitle(
    video_path: Path, stream: dict[str, object], temp_dir: Path
) -> Path | None:
    stream_index = stream.get("index")
    if stream_index is None:
        return None
    output = temp_dir / f"embedded-reference-{stream_index}.srt"
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-map",
        f"0:{stream_index}",
        "-f",
        "srt",
        str(output),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception:
        return None
    if not output.exists() or output.stat().st_size == 0:
        return None
    return output


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

    embedded_stream = _pick_reference_subtitle_stream(
        _embedded_subtitle_streams(video), lang
    )

    if args.dry_run:
        _print(f"Reference video: {video}")
        if embedded_stream is not None:
            _print(
                "Embedded subtitle reference: "
                f"stream #{embedded_stream.get('index')} "
                f"({ _stream_language(embedded_stream) or 'unknown' })"
            )
        else:
            _print("Embedded subtitle reference: none; would use audio")
        _print(f"Input subtitle: {subtitle}")
        _print(f"Output subtitle: {output}")
        return 0

    with tempfile.TemporaryDirectory(prefix="ffsubsync-ssync-") as temp_name:
        reference: Path = video
        if embedded_stream is not None:
            extracted = _extract_embedded_reference_subtitle(
                video, embedded_stream, Path(temp_name)
            )
            if extracted is not None:
                reference = extracted
                _print(
                    "Synchronizing subtitles for "
                    f"{video.name} using embedded subtitle stream "
                    f"#{embedded_stream.get('index')} as reference"
                )
            else:
                _print(
                    "Embedded subtitle stream could not be extracted; "
                    "using audio track as reference"
                )
        else:
            _print(
                f"Synchronizing subtitles for {video.name} using audio track as reference"
            )

        ffsubsync_args = make_parser().parse_args(
            [
                str(reference),
                "-i",
                str(subtitle),
                "-o",
                str(output),
                "--output-encoding",
                "same",
                *ffsubsync_extra,
            ]
        )

        result = run(ffsubsync_args)
        return int(result.get("retval", 1))


if __name__ == "__main__":
    raise SystemExit(main())
