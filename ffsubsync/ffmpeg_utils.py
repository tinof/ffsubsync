# -*- coding: utf-8 -*-
import logging
import os
import subprocess

from ffsubsync.constants import SUBSYNC_RESOURCES_ENV_MAGIC

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def subprocess_args(include_stdout=True):
    """Create subprocess arguments for Unix-like systems (Linux/macOS)."""
    if include_stdout:
        ret = {"stdout": subprocess.PIPE}
    else:
        ret = {}

    ret.update(
        {
            "stdin": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }
    )
    return ret


def ffmpeg_bin_path(bin_name, ffmpeg_resources_path=None):
    if ffmpeg_resources_path is not None:
        if not os.path.isdir(ffmpeg_resources_path):
            if bin_name.lower().startswith("ffmpeg"):
                return ffmpeg_resources_path
            ffmpeg_resources_path = os.path.dirname(ffmpeg_resources_path)
        return os.path.join(ffmpeg_resources_path, bin_name)
    try:
        resource_path = os.environ[SUBSYNC_RESOURCES_ENV_MAGIC]
        if len(resource_path) > 0:
            return os.path.join(resource_path, "ffmpeg-bin", bin_name)
    except KeyError:
        pass
    return bin_name
