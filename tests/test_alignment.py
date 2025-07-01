import pytest

from ffsubsync.aligners import FFTAligner, MaxScoreAligner


@pytest.mark.parametrize(
    "s1, s2, true_offset",
    [("111001", "11001", -1), ("1001", "1001", 0), ("10010", "01001", 1)],
)
def test_fft_alignment(s1, s2, true_offset):
    assert FFTAligner().fit_transform(s2, s1) == true_offset
    assert MaxScoreAligner(FFTAligner).fit_transform(s2, s1)[0][1] == true_offset
    assert MaxScoreAligner(FFTAligner()).fit_transform(s2, s1)[0][1] == true_offset
