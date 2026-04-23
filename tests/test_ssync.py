"""Tests for ssync CLI path resolution."""

from pathlib import Path

from ffsubsync.ssync import _candidate_subtitle_paths, _find_subtitle


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
        assert len(casefolded) == len(
            set(casefolded)
        ), "Duplicate case-folded paths found"

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
