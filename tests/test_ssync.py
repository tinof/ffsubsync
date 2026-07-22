"""Tests for ssync CLI path resolution."""

from pathlib import Path

from ffsubsync.ssync import (
    SsyncJob,
    SsyncOptions,
    _candidate_subtitle_paths,
    _find_subtitle,
    _pick_reference_subtitle_stream,
    _stream_language,
    build_sync_args,
    choose_reference_source,
    main,
    resolve_jobs,
)


class TestCandidateSubtitlePaths:
    def test_produces_paths_for_all_case_variants(self):
        video = Path("/media/show.mkv")
        candidates = _candidate_subtitle_paths(video, "fin")
        stems = [str(p) for p in candidates]
        assert any("fin" in s for s in stems)

    def test_no_duplicate_case_insensitive_paths(self):
        """When lang is already lowercase, no duplicate paths on case-insensitive FS."""
        video = Path("/media/show.mkv")
        candidates = _candidate_subtitle_paths(video, "fin")
        # Casefold all paths and check for duplicates
        casefolded = [str(p).casefold() for p in candidates]
        assert len(casefolded) == len(set(casefolded)), (
            "Duplicate case-folded paths found"
        )

    def test_uppercase_lang_deduplicated(self):
        """Single-case lang like 'EN' should not produce duplicate 'en' path if equal."""
        video = Path("/media/show.mkv")
        candidates = _candidate_subtitle_paths(video, "EN")
        casefolded = [str(p).casefold() for p in candidates]
        assert len(casefolded) == len(set(casefolded))

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
        assert result == sub

    def test_finds_finnish_alias(self, tmp_path):
        video = tmp_path / "show.mkv"
        video.touch()
        sub = tmp_path / "show.fi.srt"
        sub.touch()
        result = _find_subtitle(video, "fin")
        assert result == sub

    def test_finds_lowercase_lang(self, tmp_path):
        video = tmp_path / "show.mkv"
        video.touch()
        sub = tmp_path / "show.fin.srt"
        sub.touch()
        result = _find_subtitle(video, "FIN")
        # Should find the file regardless of case input
        assert result is not None

    def test_returns_none_for_missing_video(self, tmp_path):
        video = tmp_path / "missing.mkv"
        result = _find_subtitle(video, "fin")
        assert result is None


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
