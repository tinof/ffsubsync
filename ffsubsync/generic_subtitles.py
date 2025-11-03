import copy
import logging
import os
import sys
from collections.abc import Iterator
from datetime import timedelta
from typing import Any, Optional, cast

import pysubs2
import srt

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


class GenericSubtitle:
    def __init__(self, start, end, inner):
        self.start = start
        self.end = end
        self.inner = inner

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GenericSubtitle):
            return False
        eq = True
        eq = eq and self.start == other.start
        eq = eq and self.end == other.end
        eq = eq and self.inner == other.inner
        return eq

    @property
    def content(self) -> str:
        if isinstance(self.inner, srt.Subtitle):
            ret = self.inner.content
        elif isinstance(self.inner, pysubs2.SSAEvent):
            ret = self.inner.text
        else:
            raise NotImplementedError(f"unsupported subtitle type: {type(self.inner)}")
        return ret

    def resolve_inner_timestamps(self):
        ret = copy.deepcopy(self.inner)
        if isinstance(self.inner, srt.Subtitle):
            ret.start = self.start
            ret.end = self.end
        elif isinstance(self.inner, pysubs2.SSAEvent):
            ret.start = pysubs2.make_time(s=self.start.total_seconds())
            ret.end = pysubs2.make_time(s=self.end.total_seconds())
        else:
            raise NotImplementedError(f"unsupported subtitle type: {type(self.inner)}")
        return ret

    def merge_with(self, other):
        assert isinstance(self.inner, type(other.inner))
        inner_merged = copy.deepcopy(self.inner)
        if isinstance(self.inner, srt.Subtitle):
            inner_merged.content = f"{inner_merged.content}\n{other.inner.content}"
            return self.__class__(self.start, self.end, inner_merged)
        else:
            raise NotImplementedError(f"unsupported subtitle type: {type(self.inner)}")

    @classmethod
    def wrap_inner_subtitle(cls, sub) -> "GenericSubtitle":
        if isinstance(sub, srt.Subtitle):
            return cls(sub.start, sub.end, sub)
        elif isinstance(sub, pysubs2.SSAEvent):
            return cls(
                timedelta(milliseconds=sub.start), timedelta(milliseconds=sub.end), sub
            )
        else:
            raise NotImplementedError(f"unsupported subtitle type: {type(sub)}")


class GenericSubtitlesFile:
    def __init__(self, subs: list[GenericSubtitle], *_, **kwargs: Any):
        sub_format: str = cast(str, kwargs.pop("sub_format", None))
        if sub_format is None:
            raise ValueError("format must be specified")
        encoding: str = cast(str, kwargs.pop("encoding", None))
        if encoding is None:
            raise ValueError("encoding must be specified")
        self.subs_: list[GenericSubtitle] = subs
        self._sub_format: str = sub_format
        self._encoding: str = encoding
        self._styles: Optional[dict[str, pysubs2.SSAStyle]] = kwargs.pop("styles", None)
        self._fonts_opaque: Optional[dict[str, Any]] = kwargs.pop("fonts_opaque", None)
        self._info: Optional[dict[str, str]] = kwargs.pop("info", None)
        self._fps: Optional[float] = kwargs.pop("fps", None)

    def set_encoding(self, encoding: str) -> "GenericSubtitlesFile":
        if encoding != "same":
            self._encoding = encoding
        return self

    def __len__(self) -> int:
        return len(self.subs_)

    def __getitem__(self, item: int) -> GenericSubtitle:
        return self.subs_[item]

    def __iter__(self) -> Iterator[GenericSubtitle]:
        return iter(self.subs_)

    def clone_props_for_subs(
        self, new_subs: list[GenericSubtitle]
    ) -> "GenericSubtitlesFile":
        return GenericSubtitlesFile(
            new_subs,
            sub_format=self._sub_format,
            encoding=self._encoding,
            styles=self._styles,
            fonts_opaque=self._fonts_opaque,
            info=self._info,
            fps=self._fps,
        )

    def gen_raw_resolved_subs(self):
        for sub in self.subs_:
            yield sub.resolve_inner_timestamps()

    def offset(self, td: timedelta) -> "GenericSubtitlesFile":
        offset_subs = []
        for sub in self.subs_:
            offset_subs.append(GenericSubtitle(sub.start + td, sub.end + td, sub.inner))
        return self.clone_props_for_subs(offset_subs)

    def write_file(self, fname: str) -> None:
        # TODO: converter to go between self.subs_format and out_format
        if fname is None:
            out_format = self._sub_format
        else:
            out_format = os.path.splitext(fname)[-1][1:]
        subs = list(self.gen_raw_resolved_subs())

        def _build_ssa_from_events(events):
            ssaf = pysubs2.SSAFile()
            ssaf.events = events
            if self._styles is not None:
                ssaf.styles = self._styles
            if self._info is not None:
                ssaf.info = self._info
            if self._fonts_opaque is not None:
                ssaf.fonts_opaque = self._fonts_opaque
            if self._fps is not None:
                ssaf.fps = self._fps
            return ssaf

        to_write: str
        if out_format == "srt":
            if subs and isinstance(subs[0], pysubs2.SSAEvent):
                ssaf = _build_ssa_from_events(subs)
                to_write = ssaf.to_string("srt")
            else:
                to_write = srt.compose(subs) if subs else ""
        elif out_format in ("ssa", "ass", "vtt", "sub"):
            if subs and isinstance(subs[0], pysubs2.SSAEvent):
                ssaf = _build_ssa_from_events(subs)
            else:
                ssaf = _build_ssa_from_events(
                    pysubs2.SSAFile.from_string(srt.compose(subs)).events
                    if subs
                    else []
                )
            if out_format == "sub":
                fps = getattr(ssaf, "fps", None) or self._fps or 23.976
                ssaf.fps = fps
                to_write = ssaf.to_string("microdvd", fps=fps)
            else:
                to_write = ssaf.to_string(out_format)
        else:
            raise NotImplementedError(f"unsupported output format: {out_format}")

        with open(fname or sys.stdout.fileno(), "wb") as f:
            f.write(to_write.encode(self._encoding))


class SubsMixin:
    def __init__(self, subs: Optional[GenericSubtitlesFile] = None) -> None:
        self.subs_: Optional[GenericSubtitlesFile] = subs

    def set_encoding(self, encoding: str) -> "SubsMixin":
        self.subs_.set_encoding(encoding)
        return self
