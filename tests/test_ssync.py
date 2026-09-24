"""Tests for ssync CLI path resolution."""

from pathlib import Path

from ffsubsync.ssync import (
    SsyncJob,
    SsyncOptions,
    SubtitleCandidate,
    SyncTuning,
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

    def test_explicit_vad_overrides_forced_audio_vad(self, tmp_path):
        video, subtitle = _make_video_and_subtitle(tmp_path)
        job = SsyncJob(
            video=video,
            subtitle=subtitle,
            output=subtitle,
            lang="fin",
            reference_source="audio",
            tuning=SyncTuning(vad="tenvad"),
        )

        args = build_sync_args(choose_reference_source(job, tmp_path))

        assert args.vad == "tenvad"

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

    def test_dry_run_reports_gate(self, tmp_path, capsys):
        video, _ = _make_video_and_subtitle(tmp_path)

        main([str(video), "--dry-run", "--no-quality-gate"])

        assert "Quality gate: off" in capsys.readouterr().err


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
