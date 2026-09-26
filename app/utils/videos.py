"""Finalise generated MP4s for HTTP playback."""

import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


def make_faststart(path: Path) -> None:
    with TemporaryDirectory(dir=path.parent) as temporary_dir:
        remuxed = Path(temporary_dir) / path.name
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
                "-map", "0", "-c", "copy", "-movflags", "+faststart", str(remuxed),
            ],
            check=True,
            capture_output=True,
        )
        os.replace(remuxed, path)