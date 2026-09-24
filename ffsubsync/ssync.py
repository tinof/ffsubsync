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
SUB_EXT = "sub"
DEFAULT_FALLBACK_LANG = "en"
DEFAULT_PIECEWISE_WINDOW_MS = 60000
VAD_CHOICES = (
    "subs_then_webrtc",
    "webrtc",
    "subs_then_tenvad",
    "tenvad",
    "whisper",
)
PREFERRED_REFERENCE_LANGS = ("eng", "en")
# Image-based subtitle codecs. ffmpeg cannot convert them to SRT, so they cannot
# be extracted as a text reference.
BITMAP_SUBTITLE_CODECS = frozenset(
    {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "dvb_teletext", "xsub"}
)
# PGS is image-based too, but its caption timings can be read from the container,
# so the engine can use it as a reference directly (--pgs-ref-stream).
PGS_CODEC = "hdmv_pgs_subtitle"
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
ResultStatus = Literal["synced", "failed", "skipped", "dry_run", "kept_original"]


@dataclass(frozen=True)
class SubtitleCandidate:
    subtitle: Path
    convert_from: Path | None
    lang: str
    is_fallback: bool


@dataclass(frozen=True)
class SyncTuning:
    """Advanced ffsubsync knobs exposed through ssync.

    Fields left at ``None``/``False`` keep the ffsubsync parser defaults.
    """

    gss: bool = False
    vad: str | None = None
    max_offset_seconds: float | None = None
    use_segmented_aligner: bool = False
    no_fix_framerate: bool = False
    no_auto_sync: bool = False
    # Keep the original subtitle when the engine flags the sync as low quality.
    quality_gate: bool = True


@dataclass(frozen=True)
class SsyncOptions:
    input_path: Path
    lang: str
    fallback_lang: str = DEFAULT_FALLBACK_LANG
    dry_run: bool = False
    preflight: bool = False
    reference_source: ReferenceSource = "audio"
    tuning: SyncTuning = SyncTuning()
    piecewise: bool = False
    piecewise_window: int = DEFAULT_PIECEWISE_WINDOW_MS


@dataclass(frozen=True)
class SsyncJob:
    video: Path
    subtitle: Path
    output: Path
    lang: str
    reference_source: ReferenceSource
    candidate: SubtitleCandidate | None = None
    preflight: bool = False
    tuning: SyncTuning = SyncTuning()
    piecewise: bool = False
    piecewise_window: int = DEFAULT_PIECEWISE_WINDOW_MS


@dataclass(frozen=True)
class SsyncSyncRequest:
    reference: Path
    subtitle: Path
    output: Path
    preflight: bool
    force_audio_vad: bool
    message: str
    tuning: SyncTuning = SyncTuning()
    piecewise_audio: bool = False
    # ffmpeg specifier of an embedded PGS track to use as the reference.
    pgs_stream: str | None = None


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
    kept_original_reason: str | None = None


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
        "--fallback-lang",
        default=DEFAULT_FALLBACK_LANG,
        help=(
            "Fallback subtitle language suffix to match when target language is "
            f"not found (default: {DEFAULT_FALLBACK_LANG}). Set to empty string to disable."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved paths and ffsubsync arguments without running sync",
    )
    parser.add_argument(
        "--reference-source",
        choices=("audio", "embedded"),
        default=None,
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

    tuning = parser.add_argument_group(
        "advanced sync tuning",
        "Forwarded to the ffsubsync engine when the default sync is off.",
    )
    tuning.add_argument(
        "--gss",
        action="store_true",
        help="Use golden-section search for the framerate ratio",
    )
    tuning.add_argument(
        "--vad",
        choices=VAD_CHOICES,
        default=None,
        help="Voice activity detector to use (overrides ssync's audio default)",
    )
    tuning.add_argument(
        "--max-offset-seconds",
        type=float,
        default=None,
        help="Max allowed offset in seconds (raise when the shift is large)",
    )
    tuning.add_argument(
        "--use-segmented-aligner",
        action="store_true",
        help="Use the segmented voting aligner (long intros, sparse speech)",
    )
    tuning.add_argument(
        "--no-fix-framerate",
        action="store_true",
        help="Do not attempt framerate mismatch correction",
    )
    tuning.add_argument(
        "--no-auto-sync",
        action="store_true",
        help="Disable adaptive auto-sync strategy selection",
    )
    tuning.add_argument(
        "--no-quality-gate",
        dest="quality_gate",
        action="store_false",
        help="Write the sync even when the engine flags it as low quality "
        "(default: keep the original subtitle instead)",
    )

    piecewise = parser.add_argument_group(
        "piecewise mode",
        "For progressive mid-file drift that a single offset cannot fix.",
    )
    piecewise.add_argument(
        "--piecewise",
        action="store_true",
        help="Correct drift piece by piece. Against the audio by default; "
        "with --reference-source embedded, warps against an extracted "
        "embedded subtitle stream instead",
    )
    piecewise.add_argument(
        "--piecewise-window",
        type=int,
        default=DEFAULT_PIECEWISE_WINDOW_MS,
        help="Correction window in milliseconds for the embedded-reference "
        f"piecewise path (default: {DEFAULT_PIECEWISE_WINDOW_MS})",
    )
    return parser


def parse_options(argv: Sequence[str] | None = None) -> SsyncOptions:
    parser = _build_parser()
    args = parser.parse_args(argv)

    reference_source: ReferenceSource = args.reference_source or "audio"

    return SsyncOptions(
        input_path=Path(args.video),
        lang=args.lang or DEFAULT_SUB_LANG,
        fallback_lang=args.fallback_lang
        if args.fallback_lang is not None
        else DEFAULT_FALLBACK_LANG,
        dry_run=args.dry_run,
        preflight=args.preflight,
        reference_source=reference_source,
        tuning=SyncTuning(
            gss=args.gss,
            vad=args.vad,
            max_offset_seconds=args.max_offset_seconds,
            use_segmented_aligner=args.use_segmented_aligner,
            no_fix_framerate=args.no_fix_framerate,
            no_auto_sync=args.no_auto_sync,
            quality_gate=args.quality_gate,
        ),
        piecewise=args.piecewise,
        piecewise_window=args.piecewise_window,
    )


def _candidate_subtitle_paths(
    video_path: Path, lang: str, ext: str = SUBTITLE_EXT
) -> list[Path]:
    stem = video_path.with_suffix("")
    # Deduplicate on the exact path, not a casefolded one: on a case-sensitive
    # filesystem `show.FIN.srt` is a different file that must still be probed.
    # On a case-insensitive filesystem the variants resolve to the same file and
    # discovery returns on the first hit, so the extra probes cost nothing.
    seen: set[str] = set()
    candidates: list[Path] = []
    lang_roots = LANG_ALIASES.get(_normalize_lang(lang), (lang,))
    for lang_root in lang_roots:
        for variant in (lang_root.lower(), lang_root, lang_root.upper()):
            p = Path(f"{stem}.{variant}.{ext}")
            key = str(p)
            if key not in seen:
                seen.add(key)
                candidates.append(p)
    return candidates


def _case_insensitive_index(directory: Path) -> dict[str, Path]:
    """Map casefolded file name -> real path for one directory.

    Subtitle files arrive from many sources and their language suffix casing is
    not predictable (`.fin.srt`, `.FIN.srt`, `.Fin.srt`). Matching through this
    index makes discovery case-insensitive even on a case-sensitive filesystem.
    """
    index: dict[str, Path] = {}
    try:
        for entry in directory.iterdir():
            index.setdefault(entry.name.casefold(), entry)
    except OSError:
        pass
    return index


def _find_subtitle(
    video_path: Path, lang: str, fallback_lang: str = DEFAULT_FALLBACK_LANG
) -> SubtitleCandidate | None:
    stem = video_path.with_suffix("")
    index = _case_insensitive_index(video_path.parent)

    # Ordered probes: target .srt, bare .srt, target .sub, then the same
    # sequence for the fallback language. The bare name is probed once, in the
    # target-language pass only.
    probes: list[tuple[Path, str, bool]] = [
        (path, lang, False)
        for path in _candidate_subtitle_paths(video_path, lang, SUBTITLE_EXT)
    ]
    probes.append((Path(f"{stem}.{SUBTITLE_EXT}"), lang, False))
    probes.extend(
        (path, lang, False)
        for path in _candidate_subtitle_paths(video_path, lang, SUB_EXT)
    )
    if _normalize_lang(fallback_lang):
        for ext in (SUBTITLE_EXT, SUB_EXT):
            probes.extend(
                (path, fallback_lang, True)
                for path in _candidate_subtitle_paths(video_path, fallback_lang, ext)
            )

    seen: set[str] = set()
    for path, candidate_lang, is_fallback in probes:
        key = path.name.casefold()
        if key in seen:
            continue
        seen.add(key)

        match = index.get(key)
        if match is None or not match.is_file():
            continue

        if match.suffix.casefold() == f".{SUB_EXT}":
            return SubtitleCandidate(
                subtitle=match.with_suffix(f".{SUBTITLE_EXT}"),
                convert_from=match,
                lang=candidate_lang,
                is_fallback=is_fallback,
            )
        return SubtitleCandidate(
            subtitle=match,
            convert_from=None,
            lang=candidate_lang,
            is_fallback=is_fallback,
        )

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


def _is_text_subtitle_stream(stream: dict[str, object]) -> bool:
    return str(stream.get("codec_name", "")).lower() not in BITMAP_SUBTITLE_CODECS


def _is_pgs_stream(stream: dict[str, object]) -> bool:
    return str(stream.get("codec_name", "")).lower() == PGS_CODEC


def _pick_reference_subtitle_stream(
    streams: list[dict[str, object]], target_lang: str
) -> dict[str, object] | None:
    """Pick the embedded subtitle stream to use as the sync reference.

    Order: a text stream in another language (English first), then a PGS stream in
    another language (English first), then a stream in the target language (text
    first). Other image-based codecs (VobSub, DVB) are never used: ffmpeg cannot
    convert them to text and they carry no usable timings.
    """
    target = _normalize_lang(target_lang)

    def rank(stream: dict[str, object]) -> tuple[int, int, int]:
        lang = _stream_language(stream)
        is_target = int(lang == target)
        is_pgs = int(_is_pgs_stream(stream))
        not_preferred = int(lang not in PREFERRED_REFERENCE_LANGS)
        return (is_target, is_pgs, not_preferred)

    usable = [s for s in streams if _is_text_subtitle_stream(s) or _is_pgs_stream(s)]
    if not usable:
        return None
    # min() keeps the first stream among equal ranks, so file order breaks ties.
    return min(usable, key=rank)


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
        candidate = _find_subtitle(video, options.lang, options.fallback_lang)
        if candidate is None:
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
                subtitle=candidate.subtitle,
                output=_resolve_output_path(candidate.subtitle),
                lang=options.lang,
                reference_source=options.reference_source,
                candidate=candidate,
                preflight=options.preflight,
                tuning=options.tuning,
                piecewise=options.piecewise,
                piecewise_window=options.piecewise_window,
            )
        )

    return jobs, skipped


def _convert_sub_to_srt(source: Path, target: Path) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-sub_charenc",
        "ISO-8859-15",
        "-i",
        str(source),
        str(target),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception:
        return False
    if not target.exists() or target.stat().st_size == 0:
        return False
    with contextlib.suppress(Exception):
        source.unlink()
    return True


def choose_reference_source(job: SsyncJob, temp_dir: Path) -> SsyncSyncRequest:
    if job.reference_source == "audio":
        return SsyncSyncRequest(
            reference=job.video,
            subtitle=job.subtitle,
            output=job.output,
            preflight=job.preflight,
            force_audio_vad=True,
            message=(
                f"Synchronizing subtitles for {job.video.name} using audio "
                "track as reference"
                + (" with piecewise drift correction" if job.piecewise else "")
            ),
            tuning=job.tuning,
            piecewise_audio=job.piecewise,
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
            tuning=job.tuning,
            piecewise_audio=job.piecewise,
        )

    if _is_pgs_stream(embedded_stream):
        # No extraction: the engine reads the PGS caption timings itself.
        return SsyncSyncRequest(
            reference=job.video,
            subtitle=job.subtitle,
            output=job.output,
            preflight=job.preflight,
            force_audio_vad=False,
            message=(
                f"Synchronizing subtitles for {job.video.name} using embedded PGS "
                f"subtitle stream #{embedded_stream.get('index')} as reference"
                + (" with piecewise drift correction" if job.piecewise else "")
            ),
            tuning=job.tuning,
            piecewise_audio=job.piecewise,
            pgs_stream=f"0:{embedded_stream.get('index')}",
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
            tuning=job.tuning,
            piecewise_audio=job.piecewise,
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
        tuning=job.tuning,
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

    tuning = request.tuning
    # An explicit --vad wins over the forced audio default above.
    if tuning.vad is not None:
        args.vad = tuning.vad
    if tuning.max_offset_seconds is not None:
        args.max_offset_seconds = tuning.max_offset_seconds
    if tuning.gss:
        args.gss = True
    if tuning.use_segmented_aligner:
        args.use_segmented_aligner = True
    if tuning.no_fix_framerate:
        args.no_fix_framerate = True
    if tuning.no_auto_sync:
        args.auto_sync = False
    args.skip_sync_on_low_quality = tuning.quality_gate
    if tuning.max_offset_seconds is not None:
        # An offset the user explicitly allowed must not trip the gate.
        args.quality_max_offset_seconds = max(
            args.quality_max_offset_seconds, tuning.max_offset_seconds
        )
    if request.piecewise_audio:
        args.piecewise_audio = True
    if request.pgs_stream is not None:
        args.pgs_ref_stream = request.pgs_stream
    return args


def _dry_run_job(job: SsyncJob) -> SsyncResult:
    if job.piecewise:
        if job.reference_source == "embedded":
            _print(
                f"Mode: piecewise, embedded reference (window {job.piecewise_window}ms)"
            )
        else:
            _print("Mode: piecewise, audio reference")
    _print(f"Reference source: {job.reference_source}")
    _print(f"Quality gate: {'on' if job.tuning.quality_gate else 'off'}")
    _print(f"Reference video: {job.video}")
    if job.reference_source == "embedded":
        embedded_stream = _pick_reference_subtitle_stream(
            _embedded_subtitle_streams(job.video), job.lang
        )
        if embedded_stream is not None:
            _print(
                "Embedded subtitle reference: "
                f"stream #{embedded_stream.get('index')} "
                f"({_stream_language(embedded_stream) or 'unknown'}, "
                f"{embedded_stream.get('codec_name') or 'unknown codec'})"
            )
        else:
            _print("Embedded subtitle reference: none; would use audio")
    if job.candidate:
        if job.candidate.is_fallback:
            _print(f"Matched fallback subtitle language: {job.candidate.lang}")
        if job.candidate.convert_from is not None:
            _print(
                f"Subtitle conversion: would convert {job.candidate.convert_from.name} "
                f"to {job.subtitle.name}"
            )
    _print(f"Input subtitle: {job.subtitle}")
    _print(f"Output subtitle: {job.output}")
    return SsyncResult(video=job.video, job=job, status="dry_run")


def _skipped(job: SsyncJob, reason: str) -> SsyncResult:
    return SsyncResult(
        video=job.video,
        job=job,
        status="skipped",
        skipped_reason=reason,
    )


def _execute_piecewise_job(
    job: SsyncJob, stream: dict[str, object] | None, temp_dir: Path
) -> SsyncResult:
    from ffsubsync.tools.piecewise_sync import parse_srt, piecewise_sync, write_srt

    if stream is None:
        return _skipped(
            job,
            f"No embedded subtitle stream in {job.video.name}; piecewise sync "
            "needs a subtitle reference. Skipping.",
        )

    reference = _extract_embedded_reference_subtitle(job.video, stream, temp_dir)
    if reference is None:
        return _skipped(
            job,
            f"Embedded subtitle stream #{stream.get('index')} could not be "
            "extracted; piecewise sync needs a subtitle reference. Skipping.",
        )

    message = (
        f"Piecewise-syncing subtitles for {job.video.name} using embedded "
        f"subtitle stream #{stream.get('index')} as reference"
    )
    _print(message)

    ref_subs, _ = parse_srt(str(reference))
    input_subs, _ = parse_srt(str(job.subtitle))
    synced = piecewise_sync(ref_subs, input_subs, job.piecewise_window)
    write_srt(synced, str(job.output))

    return SsyncResult(
        video=job.video,
        job=job,
        status="synced",
        return_code=0,
        message=message,
    )


def execute_job(
    job: SsyncJob,
    executor: Callable[[argparse.Namespace], Mapping[str, Any]],
    converter: Callable[[Path, Path], bool] = _convert_sub_to_srt,
) -> SsyncResult:
    if job.candidate:
        if job.candidate.is_fallback:
            _print(
                f"Using fallback language subtitle ({job.candidate.lang}): {job.subtitle.name}"
            )
        if job.candidate.convert_from is not None:
            _print(
                f"Converting {job.candidate.convert_from.name} to {job.subtitle.name} (ISO-8859-15)..."
            )
            if not converter(job.candidate.convert_from, job.subtitle):
                msg = (
                    f"Failed to convert subtitle {job.candidate.convert_from.name} "
                    f"to {job.subtitle.name}"
                )
                _print(msg)
                return SsyncResult(
                    video=job.video,
                    job=job,
                    status="failed",
                    return_code=1,
                    message=msg,
                )

    with tempfile.TemporaryDirectory(prefix="ffsubsync-ssync-") as temp_name:
        if job.piecewise and job.reference_source == "embedded":
            stream = _pick_reference_subtitle_stream(
                _embedded_subtitle_streams(job.video), job.lang
            )
            # A text stream is warped with tools/piecewise_sync. A PGS stream has
            # no text, so the engine does audio-style piecewise against its timings.
            if stream is None or not _is_pgs_stream(stream):
                return _execute_piecewise_job(job, stream, Path(temp_name))

        request = choose_reference_source(job, Path(temp_name))
        _print(request.message)
        result = executor(build_sync_args(request))

    kept_original_reason = result.get("kept_original_reason")
    if kept_original_reason:
        msg = f"Kept original subtitle for {job.video.name}: {kept_original_reason}"
        _print(msg)
        return SsyncResult(
            video=job.video,
            job=job,
            status="kept_original",
            return_code=1,
            message=msg,
            kept_original_reason=str(kept_original_reason),
        )

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
        if result.status == "skipped" and result.skipped_reason:
            _print(result.skipped_reason)
        if result.return_code != 0:
            exit_code = result.return_code

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
