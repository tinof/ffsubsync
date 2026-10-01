"""Tests for ssync CLI path resolution."""

from pathlib import Path

import pytest

from ffsubsync.ssync import (
    SsyncJob,
    SsyncOptions,
    SubtitleCandidate,
    _candidate_subtitle_paths,
    _find_subtitle,
    _pick_reference_subtitle_stream,
    _stream_language,
    build_sync_args,
    choose_reference_source,
    execute_job,
    main,
    parse_options,
    resolve_jobs,
)


class TestCandidateSubtitlePaths:
    def test_produces_paths_for_all_case_variants(self):
        video = Path("/media/show.mkv")
        candidates = _candidate_subtitle_paths(video, "fin")
        stems = [str(p) for p in candidates]
        assert any("fin" in s for s in stems)

    def test_no_duplicate_paths(self):
        """The same exact path is never probed twice."""
        video = Path("/media/show.mkv")
        candidates = _candidate_subtitle_paths(video, "fin")
        paths = [str(p) for p in candidates]
        assert len(paths) == len(set(paths)), "Duplicate paths found"

    def test_uppercase_lang_deduplicated(self):
        """Repeating a case variant of the same lang adds no duplicate path."""
        video = Path("/media/show.mkv")
        candidates = _candidate_subtitle_paths(video, "EN")
        paths = [str(p) for p in candidates]
        assert len(paths) == len(set(paths))

    def test_includes_case_variants_of_the_language_suffix(self):
        """Case-sensitive filesystems need every case variant probed."""
        video = Path("/media/show.mkv")
        names = {p.name for p in _candidate_subtitle_paths(video, "fin")}
        assert {"show.fin.srt", "show.FIN.srt"} <= names

    def test_path_contains_video_stem(self):
        video = Path("/some/path/Movie Title.mkv")
        candidates = _candidate_subtitle_paths(video, "fin")
        for c in candidates:
            assert "Movie Title" in str(c)

    def test_extension_is_srt(self):
        video = Path("/media/show.mkv")
        for c in _candidate_subtitle_paths(video, "fin"):
            assert c.suffix == ".srt"

    def test_fin_lang_includes_fi_alias(self):
        video = Path("/media/show.mkv")
        candidates = [p.name for p in _candidate_subtitle_paths(video, "fin")]
        assert "show.fi.srt" in candidates


class TestFindSubtitle:
    def test_returns_none_when_no_subtitle_exists(self, tmp_path):
        video = tmp_path / "show.mkv"
        video.touch()
        result = _find_subtitle(video, "fin")
        assert result is None

    def test_finds_exact_lang_match(self, tmp_path):
        video = tmp_path / "show.mkv"
        video.touch()
        sub = tmp_path / "show.fin.srt"
        sub.touch()
        result = _find_subtitle(video, "fin")
        assert result is not None
        assert result.subtitle == sub
        assert result.convert_from is None
        assert result.lang == "fin"
        assert result.is_fallback is False

    def test_finds_finnish_alias(self, tmp_path):
        video = tmp_path / "show.mkv"
        video.touch()
        sub = tmp_path / "show.fi.srt"
        sub.touch()
        result = _find_subtitle(video, "fin")
        assert result is not None
        assert result.subtitle == sub
        assert result.convert_from is None
        assert result.lang == "fin"
        assert result.is_fallback is False

    def test_finds_lowercase_lang(self, tmp_path):
        video = tmp_path / "show.mkv"
        video.touch()
        sub = tmp_path / "show.fin.srt"
        sub.touch()
        result = _find_subtitle(video, "FIN")
        assert result is not None
        assert result.subtitle == sub

    def test_returns_none_for_missing_video(self, tmp_path):
        video = tmp_path / "missing.mkv"
        result = _find_subtitle(video, "fin")
        assert result is None

    def test_finds_uppercase_lang_suffix_on_disk(self, tmp_path):
        """A .FIN.srt is a distinct file on a case-sensitive filesystem."""
        video = tmp_path / "show.mkv"
        video.touch()
        sub = tmp_path / "show.FIN.srt"
        sub.touch()

        result = _find_subtitle(video, "fin")

        assert result is not None
        assert result.subtitle == sub

    def test_finds_mixed_case_lang_suffix_on_disk(self, tmp_path):
        video = tmp_path / "show.mkv"
        video.touch()
        sub = tmp_path / "show.Fin.srt"
        sub.touch()

        result = _find_subtitle(video, "fin")

        assert result is not None
        # The real on-disk path is returned, so in-place overwrite hits it.
        assert result.subtitle == sub

    def test_finds_mixed_case_sub_and_targets_matching_srt(self, tmp_path):
        video = tmp_path / "show.mkv"
        video.touch()
        sub = tmp_path / "show.Fin.sub"
        sub.touch()

        result = _find_subtitle(video, "fin")

        assert result is not None
        assert result.convert_from == sub
        assert result.subtitle == tmp_path / "show.Fin.srt"

    def test_finds_uppercase_fallback_lang_suffix_on_disk(self, tmp_path):
        video = tmp_path / "show.mkv"
        video.touch()
        sub = tmp_path / "show.EN.srt"
        sub.touch()

        result = _find_subtitle(video, "fin", fallback_lang="en")

        assert result is not None
        assert result.subtitle == sub
        assert result.is_fallback is True

    def test_ordering_exact_lang_beats_alias(self, tmp_path):
        video = tmp_path / "show.mkv"
        video.touch()
        fin_srt = tmp_path / "show.fin.srt"
        fin_srt.touch()
        fi_srt = tmp_path / "show.fi.srt"
        fi_srt.touch()
        res = _find_subtitle(video, "fin")
        assert res is not None
        assert res.subtitle == fin_srt

    def test_ordering_alias_beats_bare_srt(self, tmp_path):
        video = tmp_path / "show.mkv"
        video.touch()
        fi_srt = tmp_path / "show.fi.srt"
        fi_srt.touch()
        bare_srt = tmp_path / "show.srt"
        bare_srt.touch()
        res = _find_subtitle(video, "fin")
        assert res is not None
        assert res.subtitle == fi_srt

    def test_ordering_lang_srt_beats_lang_sub(self, tmp_path):
        video = tmp_path / "show.mkv"
        video.touch()
        fin_srt = tmp_path / "show.fin.srt"
        fin_srt.touch()
        fin_sub = tmp_path / "show.fin.sub"
        fin_sub.touch()
        res = _find_subtitle(video, "fin")
        assert res is not None
        assert res.subtitle == fin_srt
        assert res.convert_from is None

    def test_ordering_target_beats_fallback(self, tmp_path):
        video = tmp_path / "show.mkv"
        video.touch()
        fin_sub = tmp_path / "show.fin.sub"
        fin_sub.touch()
        en_srt = tmp_path / "show.en.srt"
        en_srt.touch()
        res = _find_subtitle(video, "fin", fallback_lang="en")
        assert res is not None
        assert res.subtitle == tmp_path / "show.fin.srt"
        assert res.convert_from == fin_sub
        assert res.is_fallback is False

    def test_bare_srt_selected_when_only_subtitle(self, tmp_path):
        video = tmp_path / "show.mkv"
        video.touch()
        bare_srt = tmp_path / "show.srt"
        bare_srt.touch()
        res = _find_subtitle(video, "fin")
        assert res is not None
        assert res.subtitle == bare_srt
        assert res.convert_from is None
        assert res.is_fallback is False

    def test_sub_conversion_selected_with_srt_target(self, tmp_path):
        video = tmp_path / "show.mkv"
        video.touch()
        fin_sub = tmp_path / "show.fin.sub"
        fin_sub.touch()
        jobs, skipped = resolve_jobs(
            [video], SsyncOptions(input_path=tmp_path, lang="fin")
        )
        assert len(jobs) == 1
        assert len(skipped) == 0
        job = jobs[0]
        assert job.subtitle == tmp_path / "show.fin.srt"
        assert job.output == tmp_path / "show.fin.srt"
        assert job.candidate is not None
        assert job.candidate.convert_from == fin_sub

    def test_successful_sub_conversion_deletes_source(self, tmp_path):
        video = tmp_path / "show.mkv"
        video.touch()
        sub_file = tmp_path / "show.fin.sub"
        sub_file.write_text("sub content")

        candidate = SubtitleCandidate(
            subtitle=tmp_path / "show.fin.srt",
            convert_from=sub_file,
            lang="fin",
            is_fallback=False,
        )
        job = SsyncJob(
            video=video,
            subtitle=candidate.subtitle,
            output=candidate.subtitle,
            lang="fin",
            reference_source="audio",
            candidate=candidate,
        )

        def fake_success_converter(source: Path, target: Path) -> bool:
            target.write_text("srt content")
            source.unlink()
            return True

        def fake_executor(args):
            return {"retval": 0}

        res = execute_job(job, fake_executor, converter=fake_success_converter)
        assert res.status == "synced"
        assert not sub_file.exists()
        assert (tmp_path / "show.fin.srt").exists()

    def test_failed_sub_conversion_retains_source_and_fails(self, tmp_path):
        video = tmp_path / "show.mkv"
        video.touch()
        sub_file = tmp_path / "show.fin.sub"
        sub_file.write_text("sub content")

        candidate = SubtitleCandidate(
            subtitle=tmp_path / "show.fin.srt",
            convert_from=sub_file,
            lang="fin",
            is_fallback=False,
        )
        job = SsyncJob(
            video=video,
            subtitle=candidate.subtitle,
            output=candidate.subtitle,
            lang="fin",
            reference_source="audio",
            candidate=candidate,
        )

        def fake_failed_converter(source: Path, target: Path) -> bool:
            return False

        def fake_executor(args):
            raise AssertionError("Executor should not be called when conversion fails")

        res = execute_job(job, fake_executor, converter=fake_failed_converter)
        assert res.status == "failed"
        assert res.return_code == 1
        assert sub_file.exists()

    def test_fallback_en_srt_chosen_mkv_and_mp4(self, tmp_path):
        v1 = tmp_path / "show1.mkv"
        v1.touch()
        s1 = tmp_path / "show1.en.srt"
        s1.touch()

        v2 = tmp_path / "show2.mp4"
        v2.touch()
        s2 = tmp_path / "show2.en.srt"
        s2.touch()

        jobs, _ = resolve_jobs(
            [v1, v2], SsyncOptions(input_path=tmp_path, lang="fin", fallback_lang="en")
        )
        assert len(jobs) == 2
        assert jobs[0].subtitle == s1
        assert jobs[0].candidate is not None and jobs[0].candidate.is_fallback is True
        assert jobs[1].subtitle == s2
        assert jobs[1].candidate is not None and jobs[1].candidate.is_fallback is True

    def test_fallback_en_sub_converted(self, tmp_path):
        v = tmp_path / "show.mkv"
        v.touch()
        sub_file = tmp_path / "show.en.sub"
        sub_file.touch()

        jobs, _ = resolve_jobs(
            [v], SsyncOptions(input_path=tmp_path, lang="fin", fallback_lang="en")
        )
        assert len(jobs) == 1
        assert jobs[0].subtitle == tmp_path / "show.en.srt"
        assert (
            jobs[0].candidate is not None and jobs[0].candidate.convert_from == sub_file
        )
        assert jobs[0].candidate.is_fallback is True

    def test_fallback_is_announced_when_syncing(self, tmp_path, capsys):
        v = tmp_path / "show.mkv"
        v.touch()
        (tmp_path / "show.en.srt").touch()

        exit_code = main(
            [str(v)],
            executor=lambda args: {
                "retval": 0,
                "offset_seconds": 0.0,
                "framerate_scale_factor": 1.0,
            },
        )

        assert exit_code == 0
        assert "Using fallback language subtitle (en): show.en.srt" in (
            capsys.readouterr().err
        )

    def test_fallback_is_announced_in_dry_run(self, tmp_path, capsys):
        v = tmp_path / "show.mkv"
        v.touch()
        (tmp_path / "show.en.srt").touch()

        exit_code = main([str(v), "--dry-run"])

        assert exit_code == 0
        assert "Matched fallback subtitle language: en" in capsys.readouterr().err

    def test_fallback_disabled_empty_lang(self, tmp_path):
        v = tmp_path / "show.mkv"
        v.touch()
        s = tmp_path / "show.en.srt"
        s.touch()

        jobs, skipped = resolve_jobs(
            [v], SsyncOptions(input_path=tmp_path, lang="fin", fallback_lang="")
        )
        assert len(jobs) == 0
        assert len(skipped) == 1
        assert skipped[0].video == v

    def test_dry_run_sub_only_video(self, tmp_path, capsys):
        v = tmp_path / "show.mkv"
        v.touch()
        sub_file = tmp_path / "show.fin.sub"
        sub_file.write_text("sub content")

        exit_code = main([str(v), "--dry-run"])
        assert exit_code == 0
        assert sub_file.exists()
        assert not (tmp_path / "show.fin.srt").exists()
        captured = capsys.readouterr()
        assert (
            "Subtitle conversion: would convert show.fin.sub to show.fin.srt"
            in captured.err
        )


class TestEmbeddedReferenceSubtitleSelection:
    def test_prefers_english_non_target_stream(self):
        streams = [
            {"index": 2, "tags": {"language": "fin"}},
            {"index": 3, "tags": {"language": "eng"}},
            {"index": 4, "tags": {"language": "spa"}},
        ]

        result = _pick_reference_subtitle_stream(streams, "fin")

        assert result is not None
        assert result["index"] == 3

    def test_uses_first_non_target_when_no_english_stream(self):
        streams = [
            {"index": 2, "tags": {"language": "fin"}},
            {"index": 3, "tags": {"language": "swe"}},
        ]

        result = _pick_reference_subtitle_stream(streams, "fin")

        assert result is not None
        assert result["index"] == 3

    def test_falls_back_to_target_stream_when_it_is_the_only_stream(self):
        streams = [{"index": 2, "tags": {"language": "fin"}}]

        result = _pick_reference_subtitle_stream(streams, "fin")

        assert result is not None
        assert result["index"] == 2

    def test_skips_bitmap_stream_for_text_stream(self):
        streams = [
            {
                "index": 2,
                "codec_name": "hdmv_pgs_subtitle",
                "tags": {"language": "eng"},
            },
            {"index": 3, "codec_name": "subrip", "tags": {"language": "swe"}},
        ]

        result = _pick_reference_subtitle_stream(streams, "fin")

        assert result is not None
        assert result["index"] == 3

    def test_picks_pgs_when_no_text_stream(self):
        streams = [
            {"index": 2, "codec_name": "dvd_subtitle", "tags": {"language": "eng"}},
            {
                "index": 3,
                "codec_name": "hdmv_pgs_subtitle",
                "tags": {"language": "swe"},
            },
            {
                "index": 4,
                "codec_name": "hdmv_pgs_subtitle",
                "tags": {"language": "eng"},
            },
        ]

        result = _pick_reference_subtitle_stream(streams, "fin")

        assert result is not None
        assert result["index"] == 4

    def test_non_target_pgs_beats_target_text(self):
        streams = [
            {"index": 2, "codec_name": "subrip", "tags": {"language": "fin"}},
            {
                "index": 3,
                "codec_name": "hdmv_pgs_subtitle",
                "tags": {"language": "fre"},
            },
        ]

        result = _pick_reference_subtitle_stream(streams, "fin")

        assert result is not None
        assert result["index"] == 3

    def test_returns_none_when_only_unusable_bitmap_streams(self):
        streams = [
            {"index": 2, "codec_name": "dvd_subtitle", "tags": {"language": "eng"}},
            {"index": 3, "codec_name": "dvb_subtitle", "tags": {"language": "swe"}},
        ]

        assert _pick_reference_subtitle_stream(streams, "fin") is None

    def test_stream_language_handles_missing_tags(self):
        assert _stream_language({"index": 2}) == ""


class TestSsyncMain:
    def test_resolves_directory_jobs_and_skips_missing_subtitles(self, tmp_path):
        # Create a nested directory structure with video files and subtitle files
        dir1 = tmp_path / "Season 01"
        dir1.mkdir()

        # S01E01 - video + subtitle
        v1 = dir1 / "Lucifer - S01E01.mkv"
        v1.touch()
        s1 = dir1 / "Lucifer - S01E01.fin.srt"
        s1.touch()

        # S01E02 - video + subtitle
        v2 = dir1 / "Lucifer - S01E02.mkv"
        v2.touch()
        s2 = dir1 / "Lucifer - S01E02.fin.srt"
        s2.touch()

        # S01E03 - video with no subtitle (should be skipped gracefully)
        v3 = dir1 / "Lucifer - S01E03.mkv"
        v3.touch()

        jobs, skipped = resolve_jobs(
            [v1, v2, v3],
            SsyncOptions(input_path=tmp_path, lang="fin", reference_source="audio"),
        )

        assert [job.video for job in jobs] == [v1, v2]
        assert [job.subtitle for job in jobs] == [s1, s2]
        assert skipped[0].video == v3
        assert "Skipping gracefully" in (skipped[0].skipped_reason or "")

    def test_ssync_directory_recursive(self, tmp_path, capsys):
        # Create a nested directory structure with video files and subtitle files
        dir1 = tmp_path / "Season 01"
        dir1.mkdir()

        # S01E01 - video + subtitle
        v1 = dir1 / "Lucifer - S01E01.mkv"
        v1.touch()
        s1 = dir1 / "Lucifer - S01E01.fin.srt"
        s1.touch()

        # S01E02 - video + subtitle
        v2 = dir1 / "Lucifer - S01E02.mkv"
        v2.touch()
        s2 = dir1 / "Lucifer - S01E02.fin.srt"
        s2.touch()

        # S01E03 - video with no subtitle (should be skipped gracefully)
        v3 = dir1 / "Lucifer - S01E03.mkv"
        v3.touch()

        exit_code = main([str(tmp_path), "--dry-run"])

        assert exit_code == 0

        captured = capsys.readouterr()
        # Verify the processing order (sorted alphabetically)
        assert "Processing subtitles for Lucifer - S01E01.mkv" in captured.err
        assert "Processing subtitles for Lucifer - S01E02.mkv" in captured.err
        assert "Processing subtitles for Lucifer - S01E03.mkv" in captured.err
        assert (
            "Subtitle file for Lucifer - S01E03 not found. Skipping gracefully."
            in captured.err
        )

        # Verify dry run outputs
        assert f"Reference video: {v1}" in captured.err
        assert f"Input subtitle: {s1}" in captured.err
        assert f"Reference video: {v2}" in captured.err
        assert f"Input subtitle: {s2}" in captured.err

    def test_ssync_directory_default(self, tmp_path, monkeypatch, capsys):
        # Create a video in current working directory (we will mock cwd or change to tmp_path)
        v1 = tmp_path / "Show - S01E01.mkv"
        v1.touch()
        s1 = tmp_path / "Show - S01E01.fin.srt"
        s1.touch()

        # Change cwd to tmp_path
        monkeypatch.chdir(tmp_path)

        exit_code = main(["--dry-run"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Processing subtitles for Show - S01E01.mkv" in captured.err
        assert "Reference video: Show - S01E01.mkv" in captured.err
        assert "Input subtitle: Show - S01E01.fin.srt" in captured.err

    def test_ssync_directory_empty(self, tmp_path, capsys):
        exit_code = main([str(tmp_path), "--dry-run"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert f"No video files found in directory: {tmp_path}" in captured.err

    def test_ssync_missing_path(self, tmp_path, capsys):
        missing_path = tmp_path / "nonexistent"

        exit_code = main([str(missing_path), "--dry-run"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert f"Video file or directory not found: {missing_path}" in captured.err

    def test_audio_default_does_not_probe_embedded_streams(self, tmp_path, monkeypatch):
        video = tmp_path / "Show - S01E01.mkv"
        video.touch()
        subtitle = tmp_path / "Show - S01E01.fin.srt"
        subtitle.touch()
        job = SsyncJob(
            video=video,
            subtitle=subtitle,
            output=subtitle,
            lang="fin",
            reference_source="audio",
        )

        def fail_if_probed(_):
            raise AssertionError("audio policy must not inspect embedded streams")

        monkeypatch.setattr(
            "ffsubsync.ssync._embedded_subtitle_streams", fail_if_probed
        )

        request = choose_reference_source(job, tmp_path)
        args = build_sync_args(request)

        assert request.reference == video
        assert args.reference == str(video)
        assert args.vad == "webrtc"

    def test_explicit_embedded_reference_uses_preferred_stream(
        self, tmp_path, monkeypatch
    ):
        video = tmp_path / "Show - S01E01.mkv"
        video.touch()
        subtitle = tmp_path / "Show - S01E01.fin.srt"
        subtitle.touch()
        job = SsyncJob(
            video=video,
            subtitle=subtitle,
            output=subtitle,
            lang="fin",
            reference_source="embedded",
        )
        monkeypatch.setattr(
            "ffsubsync.ssync._embedded_subtitle_streams",
            lambda _: [
                {"index": 2, "tags": {"language": "fin"}},
                {"index": 3, "tags": {"language": "eng"}},
            ],
        )

        def fake_extract(_, stream, temp_dir):
            extracted = temp_dir / f"embedded-reference-{stream['index']}.srt"
            extracted.write_text("1\n00:00:01,000 --> 00:00:02,000\nHi\n")
            return extracted

        monkeypatch.setattr(
            "ffsubsync.ssync._extract_embedded_reference_subtitle",
            fake_extract,
        )

        request = choose_reference_source(job, tmp_path)
        args = build_sync_args(request)

        assert request.reference.name == "embedded-reference-3.srt"
        assert args.reference.endswith("embedded-reference-3.srt")
        assert args.vad is None

    def test_failed_embedded_reference_falls_back_to_audio_with_message(
        self, tmp_path, monkeypatch
    ):
        video = tmp_path / "Show - S01E01.mkv"
        video.touch()
        subtitle = tmp_path / "Show - S01E01.fin.srt"
        subtitle.touch()
        job = SsyncJob(
            video=video,
            subtitle=subtitle,
            output=subtitle,
            lang="fin",
            reference_source="embedded",
        )
        monkeypatch.setattr(
            "ffsubsync.ssync._embedded_subtitle_streams",
            lambda _: [{"index": 2, "tags": {"language": "eng"}}],
        )
        monkeypatch.setattr(
            "ffsubsync.ssync._extract_embedded_reference_subtitle",
            lambda *_: None,
        )

        request = choose_reference_source(job, tmp_path)
        args = build_sync_args(request)

        assert request.reference == video
        assert args.vad == "webrtc"
        assert "could not be extracted" in request.message

    def test_full_cli_entrypoint_with_fake_executor(self, tmp_path, capsys):
        video = tmp_path / "Show - S01E01.mkv"
        video.touch()
        subtitle = tmp_path / "Show - S01E01.fin.srt"
        subtitle.touch()
        captured_args = []

        def fake_executor(args):
            captured_args.append(args)
            return {
                "retval": 0,
                "offset_seconds": 1.25,
                "framerate_scale_factor": 1.0,
            }

        exit_code = main([str(video), "--preflight"], executor=fake_executor)

        assert exit_code == 0
        assert len(captured_args) == 1
        assert captured_args[0].reference == str(video)
        assert captured_args[0].srtin == [str(subtitle)]
        assert captured_args[0].srtout == str(subtitle)
        assert captured_args[0].output_encoding == "same"
        assert captured_args[0].preflight is True
        assert captured_args[0].vad == "webrtc"
        captured = capsys.readouterr()
        assert "using audio track as reference" in captured.err


SRT_SAMPLE = (
    "1\n00:00:01,000 --> 00:00:02,000\nOne\n\n2\n00:00:10,000 --> 00:00:11,000\nTwo\n"
)


def _make_video_and_subtitle(tmp_path):
    video = tmp_path / "Show - S01E01.mkv"
    video.touch()
    subtitle = tmp_path / "Show - S01E01.fin.srt"
    subtitle.write_text(SRT_SAMPLE)
    return video, subtitle


def _fake_executor(captured_args):
    def executor(args):
        captured_args.append(args)
        return {"retval": 0, "offset_seconds": 0.0, "framerate_scale_factor": 1.0}

    return executor


class TestSyncTuning:
    def test_tuning_flags_reach_the_executor(self, tmp_path):
        video, _ = _make_video_and_subtitle(tmp_path)
        captured_args = []

        exit_code = main(
            [
                str(video),
                "--gss",
                "--max-offset-seconds",
                "120",
                "--use-segmented-aligner",
                "--no-fix-framerate",
                "--no-auto-sync",
            ],
            executor=_fake_executor(captured_args),
        )

        assert exit_code == 0
        args = captured_args[0]
        assert args.gss is True
        assert args.max_offset_seconds == 120
        assert args.use_segmented_aligner is True
        assert args.no_fix_framerate is True
        assert args.auto_sync is False

    def test_defaults_leave_engine_options_untouched(self, tmp_path):
        video, subtitle = _make_video_and_subtitle(tmp_path)
        job = SsyncJob(
            video=video,
            subtitle=subtitle,
            output=subtitle,
            lang="fin",
            reference_source="audio",
        )

        args = build_sync_args(choose_reference_source(job, tmp_path))

        assert args.gss is False
        assert args.use_segmented_aligner is False
        assert args.no_fix_framerate is False
        assert args.auto_sync is True
        assert args.vad == "webrtc"


PGS_STREAMS = [
    {"index": 5, "codec_name": "hdmv_pgs_subtitle", "tags": {"language": "eng"}},
]


class TestPgsReference:
    def _job(self, tmp_path, **kwargs):
        video, subtitle = _make_video_and_subtitle(tmp_path)
        return SsyncJob(
            video=video,
            subtitle=subtitle,
            output=subtitle,
            lang="fin",
            reference_source="embedded",
            **kwargs,
        )

    def test_pgs_stream_is_passed_to_the_engine_without_extraction(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            "ffsubsync.ssync._embedded_subtitle_streams", lambda _: PGS_STREAMS
        )

        def fail_if_extracted(*_):
            raise AssertionError("a PGS stream must not be extracted to text")

        monkeypatch.setattr(
            "ffsubsync.ssync._extract_embedded_reference_subtitle", fail_if_extracted
        )
        job = self._job(tmp_path)

        request = choose_reference_source(job, tmp_path)
        args = build_sync_args(request)

        assert request.pgs_stream == "0:5"
        assert request.reference == job.video
        assert args.pgs_ref_stream == "0:5"
        assert args.reference == str(job.video)
        assert "PGS" in request.message

    def test_piecewise_with_pgs_uses_engine_piecewise(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "ffsubsync.ssync._embedded_subtitle_streams", lambda _: PGS_STREAMS
        )

        def fail_if_tool_used(*_):
            raise AssertionError("PGS has no text for tools/piecewise_sync")

        monkeypatch.setattr("ffsubsync.ssync._execute_piecewise_job", fail_if_tool_used)
        captured_args = []
        job = self._job(tmp_path, piecewise=True)

        result = execute_job(job, _fake_executor(captured_args))

        assert result.status == "synced"
        assert captured_args[0].pgs_ref_stream == "0:5"
        assert captured_args[0].piecewise_audio is True

    def test_unusable_pgs_track_falls_back_to_audio(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            "ffsubsync.ssync._embedded_subtitle_streams", lambda _: PGS_STREAMS
        )
        captured_args = []

        def executor(args):
            captured_args.append(args)
            if args.pgs_ref_stream is not None:
                raise ValueError("No usable PGS caption timings")
            return {"retval": 0, "sync_was_successful": True}

        result = execute_job(self._job(tmp_path), executor)

        assert result.status == "synced"
        assert captured_args[-1].pgs_ref_stream is None
        assert captured_args[-1].vad == "webrtc"
        assert "PGS subtitle reference failed" in capsys.readouterr().err

    def test_pgs_is_not_offered_outside_mkv(self, tmp_path, monkeypatch):
        video = tmp_path / "Show - S01E01.m2ts"
        video.touch()
        subtitle = tmp_path / "Show - S01E01.fin.srt"
        subtitle.write_text(SRT_SAMPLE)
        monkeypatch.setattr(
            "ffsubsync.ssync._embedded_subtitle_streams", lambda _: PGS_STREAMS
        )
        job = SsyncJob(
            video=video,
            subtitle=subtitle,
            output=subtitle,
            lang="fin",
            reference_source="embedded",
        )

        request = choose_reference_source(job, tmp_path)

        assert request.pgs_stream is None
        assert request.force_audio_vad is True

    def test_dry_run_shows_codec(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            "ffsubsync.ssync._embedded_subtitle_streams", lambda _: PGS_STREAMS
        )
        video, _ = _make_video_and_subtitle(tmp_path)

        main([str(video), "--dry-run", "--reference-source", "embedded"])

        assert "(eng, hdmv_pgs_subtitle)" in capsys.readouterr().err


class TestQualityGate:
    def test_gate_is_on_by_default(self, tmp_path):
        video, _ = _make_video_and_subtitle(tmp_path)
        captured_args = []

        main([str(video)], executor=_fake_executor(captured_args))

        assert captured_args[0].skip_sync_on_low_quality is True

    def test_no_quality_gate_turns_it_off(self, tmp_path):
        video, _ = _make_video_and_subtitle(tmp_path)
        captured_args = []

        main([str(video), "--no-quality-gate"], executor=_fake_executor(captured_args))

        assert captured_args[0].skip_sync_on_low_quality is False

    def test_raised_max_offset_raises_the_gate_offset_limit(self, tmp_path):
        video, _ = _make_video_and_subtitle(tmp_path)
        captured_args = []

        main(
            [str(video), "--max-offset-seconds", "120"],
            executor=_fake_executor(captured_args),
        )

        assert captured_args[0].quality_max_offset_seconds == 120

    def test_kept_original_is_reported_and_fails_the_video(self, tmp_path, capsys):
        video, subtitle = _make_video_and_subtitle(tmp_path)
        job = SsyncJob(
            video=video,
            subtitle=subtitle,
            output=subtitle,
            lang="fin",
            reference_source="audio",
        )

        def executor(_args):
            return {"retval": 1, "kept_original_reason": "score -3.0 < 0.0"}

        result = execute_job(job, executor)

        assert result.status == "kept_original"
        assert result.return_code == 1
        assert result.kept_original_reason == "score -3.0 < 0.0"
        assert "Kept original subtitle" in capsys.readouterr().err

    def test_kept_original_sets_nonzero_exit_code(self, tmp_path):
        video, _ = _make_video_and_subtitle(tmp_path)

        def executor(_args):
            return {"retval": 1, "kept_original_reason": "|offset| 90.0s > 60.0s"}

        assert main([str(video)], executor=executor) == 1

    def test_unsuccessful_sync_is_reported_as_failed(self, tmp_path):
        video, _ = _make_video_and_subtitle(tmp_path)

        def executor(_args):
            # What run() returns when no alignment strategy succeeded.
            return {"retval": 0, "sync_was_successful": False}

        assert main([str(video)], executor=executor) == 1

    def test_dry_run_reports_gate(self, tmp_path, capsys):
        video, _ = _make_video_and_subtitle(tmp_path)

        main([str(video), "--dry-run", "--no-quality-gate"])

        assert "Quality gate: off" in capsys.readouterr().err


class TestSplitMode:
    def test_split_mode_sets_engine_split_penalty(self, tmp_path):
        video, _ = _make_video_and_subtitle(tmp_path)
        captured_args = []

        main(
            [str(video), "--piecewise", "--piecewise-mode", "split"],
            executor=_fake_executor(captured_args),
        )

        assert captured_args[0].split_penalty == 30.0
        assert captured_args[0].piecewise_audio is False

    def test_split_penalty_is_forwarded(self, tmp_path):
        video, _ = _make_video_and_subtitle(tmp_path)
        captured_args = []

        main(
            [
                str(video),
                "--piecewise",
                "--piecewise-mode",
                "split",
                "--split-penalty",
                "2",
            ],
            executor=_fake_executor(captured_args),
        )

        assert captured_args[0].split_penalty == 2.0

    def test_split_mode_needs_piecewise(self, tmp_path):
        video, _ = _make_video_and_subtitle(tmp_path)
        captured_args = []

        main(
            [str(video), "--piecewise-mode", "split"],
            executor=_fake_executor(captured_args),
        )

        assert captured_args[0].split_penalty is None

    def test_split_mode_uses_engine_with_embedded_text_reference(
        self, tmp_path, monkeypatch
    ):
        video, _ = _make_video_and_subtitle(tmp_path)
        monkeypatch.setattr(
            "ffsubsync.ssync._embedded_subtitle_streams",
            lambda _: [
                {"index": 3, "codec_name": "subrip", "tags": {"language": "eng"}}
            ],
        )

        def fake_extract(_, stream, temp_dir):
            extracted = temp_dir / f"embedded-reference-{stream['index']}.srt"
            extracted.write_text(SRT_SAMPLE)
            return extracted

        monkeypatch.setattr(
            "ffsubsync.ssync._extract_embedded_reference_subtitle", fake_extract
        )

        def fail_if_tool_used(*_):
            raise AssertionError("split mode runs in the engine, not the tool")

        monkeypatch.setattr("ffsubsync.ssync._execute_piecewise_job", fail_if_tool_used)
        captured_args = []

        main(
            [
                str(video),
                "--reference-source",
                "embedded",
                "--piecewise",
                "--piecewise-mode",
                "split",
            ],
            executor=_fake_executor(captured_args),
        )

        assert captured_args[0].split_penalty == 30.0
        assert captured_args[0].reference.endswith("embedded-reference-3.srt")

    def test_dry_run_reports_split_mode(self, tmp_path, capsys):
        video, _ = _make_video_and_subtitle(tmp_path)

        main([str(video), "--dry-run", "--piecewise", "--piecewise-mode", "split"])

        assert "Mode: piecewise (split, penalty 30s), audio reference" in (
            capsys.readouterr().err
        )


class TestPiecewiseMode:
    def test_piecewise_defaults_to_audio_reference(self):
        options = parse_options(["video.mkv", "--piecewise"])

        assert options.piecewise is True
        assert options.reference_source == "audio"

    def test_piecewise_audio_reaches_the_executor(self, tmp_path):
        video, _ = _make_video_and_subtitle(tmp_path)
        captured_args = []

        exit_code = main(
            [str(video), "--piecewise"], executor=_fake_executor(captured_args)
        )

        assert exit_code == 0
        assert captured_args[0].piecewise_audio is True

    def test_piecewise_audio_is_off_by_default(self, tmp_path):
        video, _ = _make_video_and_subtitle(tmp_path)
        captured_args = []

        main([str(video)], executor=_fake_executor(captured_args))

        assert captured_args[0].piecewise_audio is False

    def test_piecewise_writes_output_from_embedded_reference(
        self, tmp_path, monkeypatch, capsys
    ):
        video, subtitle = _make_video_and_subtitle(tmp_path)
        monkeypatch.setattr(
            "ffsubsync.ssync._embedded_subtitle_streams",
            lambda _: [{"index": 3, "tags": {"language": "eng"}}],
        )

        def fake_extract(_, stream, temp_dir):
            extracted = temp_dir / f"embedded-reference-{stream['index']}.srt"
            extracted.write_text(
                "1\n00:00:04,000 --> 00:00:05,000\nUn\n\n"
                "2\n00:00:13,000 --> 00:00:14,000\nDeux\n"
            )
            return extracted

        monkeypatch.setattr(
            "ffsubsync.ssync._extract_embedded_reference_subtitle", fake_extract
        )

        def fail_executor(_):
            raise AssertionError("piecewise mode must not call the sync engine")

        exit_code = main(
            [str(video), "--piecewise", "--reference-source", "embedded"],
            executor=fail_executor,
        )

        assert exit_code == 0
        assert "00:00:0" in subtitle.read_text(encoding="utf-8-sig")
        assert "Piecewise-syncing" in capsys.readouterr().err

    def test_piecewise_skips_when_no_embedded_stream(
        self, tmp_path, monkeypatch, capsys
    ):
        video, subtitle = _make_video_and_subtitle(tmp_path)
        original = subtitle.read_text()
        monkeypatch.setattr("ffsubsync.ssync._embedded_subtitle_streams", lambda _: [])

        exit_code = main(
            [str(video), "--piecewise", "--reference-source", "embedded"],
            executor=_fake_executor([]),
        )

        assert exit_code == 0
        assert subtitle.read_text() == original
        assert "No embedded subtitle stream" in capsys.readouterr().err

    def test_piecewise_skips_when_extraction_fails(self, tmp_path, monkeypatch, capsys):
        video, subtitle = _make_video_and_subtitle(tmp_path)
        original = subtitle.read_text()
        monkeypatch.setattr(
            "ffsubsync.ssync._embedded_subtitle_streams",
            lambda _: [{"index": 3, "tags": {"language": "eng"}}],
        )
        monkeypatch.setattr(
            "ffsubsync.ssync._extract_embedded_reference_subtitle", lambda *_: None
        )

        exit_code = main(
            [str(video), "--piecewise", "--reference-source", "embedded"],
            executor=_fake_executor([]),
        )

        assert exit_code == 0
        assert subtitle.read_text() == original
        assert "could not be extracted" in capsys.readouterr().err

    def test_piecewise_dry_run_does_not_extract(self, tmp_path, monkeypatch, capsys):
        video, _ = _make_video_and_subtitle(tmp_path)
        monkeypatch.setattr(
            "ffsubsync.ssync._embedded_subtitle_streams",
            lambda _: [{"index": 3, "tags": {"language": "eng"}}],
        )

        def fail_extract(*_):
            raise AssertionError("dry run must not extract")

        monkeypatch.setattr(
            "ffsubsync.ssync._extract_embedded_reference_subtitle", fail_extract
        )

        exit_code = main(
            [str(video), "--piecewise", "--dry-run", "--reference-source", "embedded"]
        )

        assert exit_code == 0
        assert "Mode: piecewise" in capsys.readouterr().err


class TestPiecewiseTool:
    def test_main_round_trips_srt_files(self, tmp_path):
        from ffsubsync.tools.piecewise_sync import main as piecewise_main

        reference = tmp_path / "ref.srt"
        reference.write_text(
            "1\n00:00:04,000 --> 00:00:05,000\nUn\n\n"
            "2\n00:00:13,000 --> 00:00:14,000\nDeux\n"
        )
        source = tmp_path / "in.srt"
        source.write_text(SRT_SAMPLE)
        output = tmp_path / "out.srt"

        assert piecewise_main([str(reference), str(source), str(output)]) == 0
        assert "One" in output.read_text(encoding="utf-8-sig")


AI_REFERENCE_LINES = [
    "Check the pulse now.",
    "Thanks, lads, one should pull through.",
    "This is a disaster for all of us.",
    "Where did you get it from this time?",
    "It has all gone out already.",
    "We can get some of it back.",
    "I keep a record of the drops.",
    "It is worth a try, I am saying.",
    "Go back to all the dealers.",
    "Tell them it is a bad batch.",
    "Get back whatever you can.",
    "You stick with me, kid.",
    "So what happens now?",
    "I am thinking about the girls.",
    "That is my job, guess what.",
    "And then they pay me monthly.",
]


def _srt(cues):
    blocks = []
    for n, (start, end, text) in enumerate(cues, start=1):

        def stamp(t):
            ms = round(t * 1000)
            return (
                f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:"
                f"{ms // 1000 % 60:02d},{ms % 1000:03d}"
            )

        blocks.append(f"{n}\n{stamp(start)} --> {stamp(end)}\n{text}\n")
    return "\n".join(blocks)


def _make_ai_case(tmp_path, monkeypatch, *, streams=None):
    """A subtitle of two scenes; the reference has 40 s added between them."""
    video = tmp_path / "Show - S01E01.mkv"
    video.touch()
    subtitle = tmp_path / "Show - S01E01.fin.srt"
    sub_cues = [(10.0 + 3.1 * i, 12.4 + 3.1 * i, f"rivi {i + 1}") for i in range(16)]
    subtitle.write_text(_srt(sub_cues))
    ref_cues = [
        (start + (5.0 if i < 8 else 45.0), end + (5.0 if i < 8 else 45.0), text)
        for i, ((start, end, _), text) in enumerate(
            zip(sub_cues, AI_REFERENCE_LINES, strict=True)
        )
    ]
    ref_cues.insert(8, (42.0, 44.0, "SIREN WAILS"))
    if streams is None:
        streams = [{"index": 2, "codec_name": "subrip", "tags": {"language": "eng"}}]
    monkeypatch.setattr("ffsubsync.ssync._embedded_subtitle_streams", lambda _: streams)

    def fake_extract(_, stream, temp_dir):
        extracted = temp_dir / f"embedded-reference-{stream['index']}.srt"
        extracted.write_text(_srt(ref_cues))
        return extracted

    monkeypatch.setattr(
        "ffsubsync.ssync._extract_embedded_reference_subtitle", fake_extract
    )
    return video, subtitle


def _cue_starts(path):
    import srt

    return [c.start.total_seconds() for c in srt.parse(path.read_text())]


def _fail_executor(args):
    raise AssertionError("AI mode must not call the sync engine")


class TestAiMode:
    def test_flags_are_parsed(self):
        options = parse_options(
            [
                "ep.mkv",
                "--ai",
                "--ai-model",
                "opus",
                "--ai-timeout",
                "60",
                "--ai-budget-usd",
                "0.5",
            ]
        )

        assert options.ai.enabled is True
        assert options.ai.fallback is False
        assert options.ai.judge is True
        assert options.ai.model == "opus"
        assert options.ai.timeout == 60
        assert options.ai.budget_usd == 0.5

    def test_ai_is_off_by_default(self):
        options = parse_options(["ep.mkv"])

        assert options.ai.enabled is False
        assert options.ai.fallback is False

    @pytest.mark.parametrize(
        "argv", [["--ai", "--piecewise"], ["--ai", "--ai-fallback"]]
    )
    def test_conflicting_flags_are_rejected(self, argv):
        with pytest.raises(SystemExit):
            parse_options(["ep.mkv", *argv])

    def test_syncs_across_an_added_scene_and_consults_the_judge(
        self, tmp_path, monkeypatch, capsys
    ):
        video, subtitle = _make_ai_case(tmp_path, monkeypatch)
        prompts = []

        def judge(prompt):
            prompts.append(prompt)
            return []

        code = main([str(video), "--ai"], executor=_fail_executor, judge=judge)

        assert code == 0
        starts = _cue_starts(subtitle)
        assert starts[0] == pytest.approx(15.0, abs=0.05)
        assert starts[7] == pytest.approx(10.0 + 3.1 * 7 + 5.0, abs=0.05)
        assert starts[8] == pytest.approx(10.0 + 3.1 * 8 + 45.0, abs=0.05)
        assert starts[15] == pytest.approx(10.0 + 3.1 * 15 + 45.0, abs=0.05)
        assert len(prompts) == 1
        # The cues around the jump, the reference text, and no sound description.
        assert "S9 " in prompts[0] and "rivi 9" in prompts[0]
        assert "Go back to all the dealers." in prompts[0]
        assert "SIREN WAILS" not in prompts[0]
        err = capsys.readouterr().err
        assert "AI sync: 2 segment(s), 1 jump(s)" in err

    def test_judge_matches_become_anchors(self, tmp_path, monkeypatch):
        video, subtitle = _make_ai_case(tmp_path, monkeypatch)

        def judge(prompt):
            # Cues 7 and 8 already belong to the scene after the jump.
            return [{"sub": 7, "ref": 9}, {"sub": 8, "ref": 10}]

        main([str(video), "--ai"], executor=_fail_executor, judge=judge)

        starts = _cue_starts(subtitle)
        # R9/R10 are the dialogue cues at 79.8 s and 82.9 s.
        assert starts[6] == pytest.approx(79.8, abs=0.05)
        assert starts[7] == pytest.approx(82.9, abs=0.05)

    def test_unavailable_judge_falls_back_to_the_aligner(
        self, tmp_path, monkeypatch, capsys
    ):
        from ffsubsync.ai_judge import JudgeUnavailable

        video, subtitle = _make_ai_case(tmp_path, monkeypatch)

        def judge(prompt):
            raise JudgeUnavailable("the 'claude' CLI is not on PATH")

        code = main([str(video), "--ai"], executor=_fail_executor, judge=judge)

        assert code == 0
        assert _cue_starts(subtitle)[15] == pytest.approx(101.5, abs=0.05)
        assert "judge unavailable" in capsys.readouterr().err

    def test_no_judge_flag_never_asks(self, tmp_path, monkeypatch):
        video, subtitle = _make_ai_case(tmp_path, monkeypatch)

        def judge(prompt):
            raise AssertionError("--ai-no-judge must not ask the judge")

        code = main(
            [str(video), "--ai", "--ai-no-judge"], executor=_fail_executor, judge=judge
        )

        assert code == 0
        assert _cue_starts(subtitle)[0] == pytest.approx(15.0, abs=0.05)

    def test_skips_without_an_embedded_text_stream(self, tmp_path, monkeypatch, capsys):
        pgs = [{"index": 2, "codec_name": "hdmv_pgs_subtitle", "tags": {}}]
        video, subtitle = _make_ai_case(tmp_path, monkeypatch, streams=pgs)
        before = subtitle.read_text()

        code = main([str(video), "--ai"], executor=_fail_executor)

        assert code == 0
        assert subtitle.read_text() == before
        assert "AI mode needs one as reference" in capsys.readouterr().err

    def test_gate_keeps_the_original_when_cues_do_not_match(
        self, tmp_path, monkeypatch, capsys
    ):
        video, subtitle = _make_ai_case(tmp_path, monkeypatch)
        # An unrelated subtitle: irregular cue spacing that fits no reference cue.
        gaps = [2.3, 4.9, 3.7, 6.1, 2.9, 5.3, 4.1, 7.7, 3.3, 5.9, 2.6, 6.8]
        cues, t = [], 3.0
        for i, gap in enumerate(gaps):
            cues.append((t, t + 1.2, f"muu {i}"))
            t += gap
        subtitle.write_text(_srt(cues))
        before = subtitle.read_text()

        def judge(prompt):
            raise AssertionError("a hopeless fit must not reach the judge")

        code = main([str(video), "--ai"], executor=_fail_executor, judge=judge)

        assert code == 1
        assert subtitle.read_text() == before
        assert "Kept original subtitle" in capsys.readouterr().err

    def test_dry_run_names_the_stream_and_asks_nobody(
        self, tmp_path, monkeypatch, capsys
    ):
        video, _ = _make_ai_case(tmp_path, monkeypatch)

        main([str(video), "--ai", "--dry-run"], executor=_fail_executor)

        err = capsys.readouterr().err
        assert "Mode: AI (Claude judge), embedded text subtitle stream #2 (eng)" in err


class TestAiFallback:
    def test_runs_after_the_engine_keeps_the_original(self, tmp_path, monkeypatch):
        video, subtitle = _make_ai_case(tmp_path, monkeypatch)

        def executor(args):
            return {"retval": 0, "kept_original_reason": "offset too large"}

        code = main(
            [str(video), "--ai-fallback"], executor=executor, judge=lambda p: []
        )

        assert code == 0
        assert _cue_starts(subtitle)[15] == pytest.approx(101.5, abs=0.05)

    def test_runs_after_the_engine_fails(self, tmp_path, monkeypatch):
        video, subtitle = _make_ai_case(tmp_path, monkeypatch)

        def executor(args):
            return {"retval": 0, "sync_was_successful": False}

        code = main(
            [str(video), "--ai-fallback"], executor=executor, judge=lambda p: []
        )

        assert code == 0
        assert _cue_starts(subtitle)[0] == pytest.approx(15.0, abs=0.05)

    def test_does_not_run_after_a_sync_that_fits_the_embedded_subtitle(
        self, tmp_path, monkeypatch
    ):
        video, subtitle = _make_ai_case(tmp_path, monkeypatch)
        good = _srt(
            [
                (10.0 + 3.1 * i + shift, 12.4 + 3.1 * i + shift, f"rivi {i + 1}")
                for i in range(16)
                for shift in [5.0 if i < 8 else 45.0]
            ]
        )

        def executor(args):
            Path(args.srtout).write_text(good)
            return {"retval": 0, "offset_seconds": 5.0, "framerate_scale_factor": 1.0}

        def judge(prompt):
            raise AssertionError("fallback must not run after a good sync")

        code = main([str(video), "--ai-fallback"], executor=executor, judge=judge)

        assert code == 0
        assert subtitle.read_text() == good

    def test_runs_when_a_successful_sync_misses_the_embedded_subtitle(
        self, tmp_path, monkeypatch, capsys
    ):
        video, subtitle = _make_ai_case(tmp_path, monkeypatch)
        original = subtitle.read_text()

        def executor(args):
            # The engine "succeeds" with a wrong framerate scale, in place.
            Path(args.srtout).write_text(original.replace("00:00:1", "00:00:4"))
            return {"retval": 0, "offset_seconds": -3.1, "framerate_scale_factor": 0.96}

        code = main(
            [str(video), "--ai-fallback"], executor=executor, judge=lambda p: []
        )

        assert code == 0
        # AI mode started from the original timings, not from the engine's output.
        starts = _cue_starts(subtitle)
        assert starts[0] == pytest.approx(15.0, abs=0.05)
        assert starts[15] == pytest.approx(101.5, abs=0.05)
        assert "of cues on an embedded subtitle cue" in capsys.readouterr().err

    def test_successful_sync_stands_without_an_embedded_text_stream(
        self, tmp_path, monkeypatch
    ):
        video, subtitle = _make_ai_case(tmp_path, monkeypatch, streams=[])
        before = subtitle.read_text()

        code = main([str(video), "--ai-fallback"], executor=_fake_executor([]))

        assert code == 0
        assert subtitle.read_text() == before

    def test_engine_output_is_restored_when_ai_mode_keeps_the_original(
        self, tmp_path, monkeypatch
    ):
        video, subtitle = _make_ai_case(tmp_path, monkeypatch)
        gaps = [2.3, 4.9, 3.7, 6.1, 2.9, 5.3, 4.1, 7.7, 3.3, 5.9, 2.6, 6.8]
        cues, t = [], 3.0
        for i, gap in enumerate(gaps):
            cues.append((t, t + 1.2, f"muu {i}"))
            t += gap
        subtitle.write_text(_srt(cues))
        engine_output = _srt([(a + 0.25, b + 0.25, text) for a, b, text in cues])

        def executor(args):
            Path(args.srtout).write_text(engine_output)
            return {"retval": 0, "offset_seconds": 0.25, "framerate_scale_factor": 1.0}

        code = main(
            [str(video), "--ai-fallback"], executor=executor, judge=lambda p: []
        )

        assert code == 0
        assert subtitle.read_text() == engine_output

    def test_engine_result_stands_when_ai_mode_cannot_run(
        self, tmp_path, monkeypatch, capsys
    ):
        video, _ = _make_ai_case(tmp_path, monkeypatch, streams=[])

        def executor(args):
            return {"retval": 0, "kept_original_reason": "offset too large"}

        code = main([str(video), "--ai-fallback"], executor=executor)

        assert code == 1
        err = capsys.readouterr().err
        assert "Kept original subtitle" in err
        assert "AI mode needs one as reference" in err
