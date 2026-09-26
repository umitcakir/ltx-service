"""Finalise generated MP4s for HTTP playback."""

import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from imageio_ffmpeg import get_ffmpeg_exe


def make_faststart(path: Path) -> None:
    with TemporaryDirectory(dir=path.parent) as temporary_dir:
        remuxed = Path(temporary_dir) / path.name
        subprocess.run(
            [
                get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path),
                "-map", "0", "-c", "copy", "-movflags", "+faststart", str(remuxed),
            ],
            check=True,
            capture_output=True,
        )
        os.replace(remuxed, path)