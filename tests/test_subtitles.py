import itertools
from datetime import timedelta
from io import BytesIO

import numpy as np
import pytest

from ffsubsync.sklearn_shim import make_pipeline
from ffsubsync.speech_transformers import SubtitleSpeechTransformer
from ffsubsync.subtitle_parser import GenericSubtitleParser
from ffsubsync.subtitle_transformers import SubtitleShifter

fake_srt = b"""1
00:00:00,178 --> 00:00:01,1416
<i>Previously on "Your favorite TV show..."</i>

2
00:00:01,1828 --> 00:00:04,549
Oh hi, Mark.

3
00:00:04,653 --> 00:00:03,3062
You are tearing me apart, Lisa!
"""

# Occasionally some srt files have timestamps whose 'milliseconds'
# field has more than 3 digits... Ideally we should test that these
# are handled properly with dedicated tests, but in the interest of
# development speed I've opted to sprinkle in a few >3 digit
# millisecond fields into the dummy string above in order to exercise
# this case integration-test style in the below unit tests.


@pytest.mark.parametrize("start_seconds", [0, 2, 4, 6])
def test_start_seconds(start_seconds):
    parser_zero = GenericSubtitleParser(start_seconds=0)
    parser_zero.fit(BytesIO(fake_srt))
    parser = GenericSubtitleParser(start_seconds=start_seconds)
    parser.fit(BytesIO(fake_srt))
    expected = [
        sub
        for sub in parser_zero.subs_
        if sub.start >= timedelta(seconds=start_seconds)
    ]
    assert all(esub == psub for esub, psub in zip(expected, parser.subs_))


@pytest.mark.parametrize("max_seconds", [1, 1.5, 2.0, 2.5])
def test_max_seconds(max_seconds):
    parser = GenericSubtitleParser(max_subtitle_seconds=max_seconds)
    parser.fit(BytesIO(fake_srt))
    assert max(sub.end - sub.start for sub in parser.subs_) <= timedelta(
        seconds=max_seconds
    )


@pytest.mark.parametrize("encoding", ["utf-8", "ascii", "latin-1"])
def test_same_encoding(encoding):
    parser = GenericSubtitleParser(encoding=encoding)
    offseter = SubtitleShifter(1)
    pipe = make_pipeline(parser, offseter)
    pipe.fit(BytesIO(fake_srt))
    assert parser.subs_._encoding == encoding
    assert offseter.subs_._encoding == parser.subs_._encoding
    assert offseter.subs_.set_encoding("same")._encoding == encoding
    assert offseter.subs_.set_encoding("utf-8")._encoding == "utf-8"


@pytest.mark.parametrize("offset", [1, 1.5, -2.3])
def test_offset(offset):
    parser = GenericSubtitleParser()
    offseter = SubtitleShifter(offset)
    pipe = make_pipeline(parser, offseter)
    pipe.fit(BytesIO(fake_srt))
    for sub_orig, sub_offset in zip(parser.subs_, offseter.subs_):
        assert (
            abs(
                sub_offset.start.total_seconds()
                - sub_orig.start.total_seconds()
                - offset
            )
            < 1e-6
        )
        assert (
            abs(sub_offset.end.total_seconds() - sub_orig.end.total_seconds() - offset)
            < 1e-6
        )


@pytest.mark.parametrize(
    "sample_rate,start_seconds", itertools.product([10, 20, 100, 300], [0, 2, 4, 6])
)
def test_speech_extraction(sample_rate, start_seconds):
    parser = GenericSubtitleParser(start_seconds=start_seconds)
    extractor = SubtitleSpeechTransformer(
        sample_rate=sample_rate, start_seconds=start_seconds
    )
    pipe = make_pipeline(parser, extractor)
    bitstring = pipe.fit_transform(BytesIO(fake_srt)).astype(bool)
    bitstring_shifted_left = np.append(bitstring[1:], [False])
    bitstring_shifted_right = np.append([False], bitstring[:-1])
    bitstring_cumsum = np.cumsum(bitstring)
    consec_ones_end_pos = np.nonzero(
        bitstring_cumsum
        * (bitstring ^ bitstring_shifted_left)
        * (bitstring_cumsum != np.cumsum(bitstring_shifted_right))
    )[0]
    prev = 0
    for pos, sub in zip(consec_ones_end_pos, parser.subs_):
        start = round(sub.start.total_seconds() * sample_rate)
        duration = sub.end.total_seconds() - sub.start.total_seconds()
        stop = start + round(duration * sample_rate)
        assert bitstring_cumsum[pos] - prev == stop - start
        prev = bitstring_cumsum[pos]


def test_max_time_found():
    parser = GenericSubtitleParser()
    extractor = SubtitleSpeechTransformer(sample_rate=100)
    pipe = make_pipeline(parser, extractor)
    pipe.fit(BytesIO(fake_srt))
    assert extractor.max_time_ == 6.062


def test_microdvd_roundtrip(tmp_path):
    # Create a minimal MicroDVD sample with an explicit FPS declaration
    # First line encodes FPS per MicroDVD convention: {1}{1}<fps>
    fps = 25.0
    microdvd_text = "\n".join(
        [
            f"{{1}}{{1}}{fps}",
            "{0}{24}Hello|World",
            "{30}{60}Second line",
            "",
        ]
    )
    src_path = tmp_path / "sample.sub"
    src_path.write_text(microdvd_text, encoding="latin-1")

    parser = GenericSubtitleParser(fmt="sub", encoding="latin-1")
    parser.fit(str(src_path))
    subs = parser.subs_

    out_path = tmp_path / "roundtrip.sub"
    subs.write_file(str(out_path))

    parser_roundtrip = GenericSubtitleParser(fmt="sub", encoding="latin-1")
    parser_roundtrip.fit(str(out_path))

    assert len(parser_roundtrip.subs_) == len(subs)
    assert parser_roundtrip.subs_[0].content == subs.subs_[0].content
    assert (
        abs(
            parser_roundtrip.subs_[0].start.total_seconds()
            - subs.subs_[0].start.total_seconds()
        )
        < 1e-6
    )
