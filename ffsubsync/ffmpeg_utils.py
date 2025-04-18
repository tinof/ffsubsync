# -*- coding: utf-8 -*-
import logging
import os
import platform
import subprocess

from ffsubsync.constants import SUBSYNC_RESOURCES_ENV_MAGIC

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


# ref: https://github.com/pyinstaller/pyinstaller/wiki/Recipe-subprocess
# Create a set of arguments which make a ``subprocess.Popen`` (and
# variants) call work with or without Pyinstaller, ``--noconsole`` or
# not, on Windows and Linux. Typical use::
#
#   subprocess.call(['program_to_run', 'arg_1'], **subprocess_args())
#
# When calling ``check_output``::
#
#   subprocess.check_output(['program_to_run', 'arg_1'],
#                           **subprocess_args(False))
def subprocess_args(include_stdout=True):
    # ``subprocess.check_output`` doesn't allow specifying ``stdout``::
    #
    #   Traceback (most recent call last):
    #     File "test_subprocess.py", line 58, in <module>
    #       **subprocess_args(stdout=None))
    #     File "C:\Python27\lib\subprocess.py", line 567, in check_output
    #       raise ValueError('stdout argument not allowed, it will be overridden.')
    #   ValueError: stdout argument not allowed, it will be overridden.
    #
    # So, add it only if it's needed.
    if include_stdout:
        ret = {"stdout": subprocess.PIPE}
    else:
        ret = {}

    # On Windows, running this from the binary produced by Pyinstaller
    # with the ``--noconsole`` option requires redirecting everything
    # (stdin, stdout, stderr) to avoid an OSError exception
    # "[Error 6] the handle is invalid."
    ret.update({"stdin": subprocess.PIPE, "stderr": subprocess.PIPE})
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
        # Fall back to system path if environment variable not set
        pass
    return bin_name
