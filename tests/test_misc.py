import pytest

from ffsubsync.ffsubsync import get_alignment_strategies, make_parser
from ffsubsync.version import make_version_tuple


@pytest.mark.parametrize(
    "vstr, expected",
    [("v0.1.1", (0, 1, 1)), ("v1.2.3", (1, 2, 3)), ("4.5.6.1", (4, 5, 6, 1))],
)
def test_version_tuple_from_string(vstr, expected):
    assert make_version_tuple(vstr) == expected


def test_auto_sync_enabled_by_default():
    parser = make_parser()
    args = parser.parse_args([])
    assert args.auto_sync is True


def test_no_auto_sync_flag_disables_adaptive_strategy():
    parser = make_parser()
    args = parser.parse_args(["--no-auto-sync"])
    assert args.auto_sync is False
    assert get_alignment_strategies(args) == [("primary", False, False)]


def test_auto_sync_adds_adaptive_strategy():
    parser = make_parser()
    args = parser.parse_args([])
    assert get_alignment_strategies(args) == [
        ("primary", False, False),
        ("adaptive", True, True),
    ]


def test_auto_sync_respects_no_fix_framerate_for_adaptive_strategy():
    parser = make_parser()
    args = parser.parse_args(["--no-fix-framerate"])
    assert get_alignment_strategies(args) == [
        ("primary", False, False),
        ("adaptive", False, True),
    ]
