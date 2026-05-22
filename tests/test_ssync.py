"""Tests for ssync CLI path resolution."""

from pathlib import Path

from ffsubsync.ssync import (
    _candidate_subtitle_paths,
    _find_subtitle,
    _pick_reference_subtitle_stream,
    _stream_language,
    main,
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
    def test_ssync_directory_recursive(self, tmp_path, monkeypatch, capsys):
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

        # Mock sys.argv to run on tmp_path in dry-run mode
        monkeypatch.setattr("sys.argv", ["ssync", str(tmp_path), "--dry-run"])

        # Run main
        exit_code = main()

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

        # Mock sys.argv to run with no arguments (which defaults to ".") in dry-run mode
        monkeypatch.setattr("sys.argv", ["ssync", "--dry-run"])

        # Import main from ssync
        exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Processing subtitles for Show - S01E01.mkv" in captured.err
        assert "Reference video: Show - S01E01.mkv" in captured.err
        assert "Input subtitle: Show - S01E01.fin.srt" in captured.err

    def test_ssync_directory_empty(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["ssync", str(tmp_path), "--dry-run"])

        exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert f"No video files found in directory: {tmp_path}" in captured.err

    def test_ssync_missing_path(self, tmp_path, monkeypatch, capsys):
        missing_path = tmp_path / "nonexistent"
        monkeypatch.setattr("sys.argv", ["ssync", str(missing_path), "--dry-run"])

        exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert f"Video file or directory not found: {missing_path}" in captured.err
