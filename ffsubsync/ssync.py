import argparse
import contextlib
import json
import locale
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ffsubsync.ffsubsync import make_parser, run

DEFAULT_SUB_LANG = "fin"
SUBTITLE_EXT = "srt"
PREFERRED_REFERENCE_LANGS = ("eng", "en")
LANG_ALIASES = {
    "fin": ("fin", "fi"),
    "fi": ("fi", "fin"),
}
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".m4v",
    ".ts",
    ".mov",
    ".webm",
    ".flv",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".ogv",
    ".3gp",
}

ReferenceSource = Literal["audio", "embedded"]
ResultStatus = Literal["synced", "failed", "skipped", "dry_run"]


@dataclass(frozen=True)
class SsyncOptions:
    input_path: Path
    lang: str
    dry_run: bool = False
    preflight: bool = False
    reference_source: ReferenceSource = "audio"


@dataclass(frozen=True)
class SsyncJob:
    video: Path
    subtitle: Path
    output: Path
    lang: str
    reference_source: ReferenceSource
    preflight: bool = False


@dataclass(frozen=True)
class SsyncSyncRequest:
    reference: Path
    subtitle: Path
    output: Path
    preflight: bool
    force_audio_vad: bool
    message: str


@dataclass(frozen=True)
class SsyncResult:
    video: Path
    job: SsyncJob | None
    status: ResultStatus
    return_code: int = 0
    skipped_reason: str | None = None
    message: str | None = None
    offset_seconds: Any = None
    framerate_scale_factor: Any = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-discover a language-specific subtitle file next to a video and "
            "sync it with ffsubsync."
        )
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=".",
        help="Path to the reference video file or directory (default: current directory)",
    )
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
        "--reference-source",
        choices=("audio", "embedded"),
        default="audio",
        help="Reference source to sync against (default: audio)",
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


def parse_options(argv: Sequence[str] | None = None) -> SsyncOptions:
    args = _build_parser().parse_args(argv)
    return SsyncOptions(
        input_path=Path(args.video),
        lang=args.lang or DEFAULT_SUB_LANG,
        dry_run=args.dry_run,
        preflight=args.preflight,
        reference_source=args.reference_source,
    )


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


def resolve_videos(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Video file or directory not found: {input_path}")

    if not input_path.is_dir():
        return [input_path]

    return sorted(
        p
        for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def resolve_jobs(
    videos: Sequence[Path], options: SsyncOptions
) -> tuple[list[SsyncJob], list[SsyncResult]]:
    jobs: list[SsyncJob] = []
    skipped: list[SsyncResult] = []
    for video in videos:
        subtitle = _find_subtitle(video, options.lang)
        if subtitle is None:
            skipped.append(
                SsyncResult(
                    video=video,
                    job=None,
                    status="skipped",
                    skipped_reason=(
                        f"Subtitle file for {video.stem} not found. "
                        "Skipping gracefully."
                    ),
                )
            )
            continue

        jobs.append(
            SsyncJob(
                video=video,
                subtitle=subtitle,
                output=_resolve_output_path(subtitle),
                lang=options.lang,
                reference_source=options.reference_source,
                preflight=options.preflight,
            )
        )

    return jobs, skipped


def choose_reference_source(job: SsyncJob, temp_dir: Path) -> SsyncSyncRequest:
    if job.reference_source == "audio":
        return SsyncSyncRequest(
            reference=job.video,
            subtitle=job.subtitle,
            output=job.output,
            preflight=job.preflight,
            force_audio_vad=True,
            message=f"Synchronizing subtitles for {job.video.name} using audio track as reference",
        )

    embedded_stream = _pick_reference_subtitle_stream(
        _embedded_subtitle_streams(job.video), job.lang
    )
    if embedded_stream is None:
        return SsyncSyncRequest(
            reference=job.video,
            subtitle=job.subtitle,
            output=job.output,
            preflight=job.preflight,
            force_audio_vad=True,
            message="No embedded subtitle reference found; using audio track as reference",
        )

    extracted = _extract_embedded_reference_subtitle(
        job.video, embedded_stream, temp_dir
    )
    if extracted is None:
        return SsyncSyncRequest(
            reference=job.video,
            subtitle=job.subtitle,
            output=job.output,
            preflight=job.preflight,
            force_audio_vad=True,
            message=(
                "Embedded subtitle stream could not be extracted; "
                "using audio track as reference"
            ),
        )

    return SsyncSyncRequest(
        reference=extracted,
        subtitle=job.subtitle,
        output=job.output,
        preflight=job.preflight,
        force_audio_vad=False,
        message=(
            f"Synchronizing subtitles for {job.video.name} using embedded subtitle "
            f"stream #{embedded_stream.get('index')} as reference"
        ),
    )


def build_sync_args(request: SsyncSyncRequest) -> argparse.Namespace:
    args = make_parser().parse_args([])
    args.reference = str(request.reference)
    args.srtin = [str(request.subtitle)]
    args.srtout = str(request.output)
    args.output_encoding = "same"
    args.preflight = request.preflight
    if request.force_audio_vad:
        # ssync owns embedded-reference selection. When a video is the reference,
        # force audio VAD instead of allowing ffsubsync to prefer subtitle streams.
        args.vad = "webrtc"
    return args


def _dry_run_job(job: SsyncJob) -> SsyncResult:
    _print(f"Reference source: {job.reference_source}")
    _print(f"Reference video: {job.video}")
    if job.reference_source == "embedded":
        embedded_stream = _pick_reference_subtitle_stream(
            _embedded_subtitle_streams(job.video), job.lang
        )
        if embedded_stream is not None:
            _print(
                "Embedded subtitle reference: "
                f"stream #{embedded_stream.get('index')} "
                f"({_stream_language(embedded_stream) or 'unknown'})"
            )
        else:
            _print("Embedded subtitle reference: none; would use audio")
    _print(f"Input subtitle: {job.subtitle}")
    _print(f"Output subtitle: {job.output}")
    return SsyncResult(video=job.video, job=job, status="dry_run")


def execute_job(
    job: SsyncJob,
    executor: Callable[[argparse.Namespace], Mapping[str, Any]],
) -> SsyncResult:
    with tempfile.TemporaryDirectory(prefix="ffsubsync-ssync-") as temp_name:
        request = choose_reference_source(job, Path(temp_name))
        _print(request.message)
        result = executor(build_sync_args(request))

    retval = int(result.get("retval", 1))
    return SsyncResult(
        video=job.video,
        job=job,
        status="synced" if retval == 0 else "failed",
        return_code=retval,
        message=request.message,
        offset_seconds=result.get("offset_seconds"),
        framerate_scale_factor=result.get("framerate_scale_factor"),
    )


def main(
    argv: Sequence[str] | None = None,
    executor: Callable[[argparse.Namespace], Mapping[str, Any]] = run,
) -> int:
    options = parse_options(argv)

    with contextlib.suppress(Exception):
        locale.setlocale(locale.LC_ALL, "")

    try:
        videos = resolve_videos(options.input_path)
    except FileNotFoundError as e:
        _print(e)
        return 1

    if not videos:
        _print(f"No video files found in directory: {options.input_path}")
        return 0

    jobs, skipped = resolve_jobs(videos, options)
    skipped_by_video = {result.video: result for result in skipped}

    exit_code = 0
    for video in videos:
        _print(f"Processing subtitles for {video.name} (Language: {options.lang})")
        skipped_result = skipped_by_video.get(video)
        if skipped_result is not None:
            _print(skipped_result.skipped_reason)
            continue

        job = next(job for job in jobs if job.video == video)
        result = _dry_run_job(job) if options.dry_run else execute_job(job, executor)
        if result.return_code != 0:
            exit_code = result.return_code

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
