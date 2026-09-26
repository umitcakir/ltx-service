"""LTX-2.5 capability matrix for the resolutions this service supports.

Scope is deliberately limited to 480p and 720p; 1440p/4K are out of scope.

Hard model constraints (from the LTX-2.5 model card):
  * ``num_frames % 8 == 1``
  * width and height must be divisible by 32
"""

from __future__ import annotations

from typing import Literal

Resolution = Literal["480p", "720p"]
AspectRatio = Literal["16:9", "9:16"]
OutputFormat = Literal["mp4", "webm", "mkv"]

#: Ordered from smallest to largest, used for "max allowed resolution" checks.
RESOLUTION_ORDER: tuple[Resolution, ...] = ("480p", "720p")

#: (resolution, aspect_ratio) -> (width, height). All values divisible by 32.
RESOLUTION_MATRIX: dict[tuple[str, str], tuple[int, int]] = {
    ("480p", "16:9"): (832, 480),
    ("480p", "9:16"): (480, 832),
    ("720p", "16:9"): (1280, 704),
    ("720p", "9:16"): (704, 1280),
}

#: Frame rates LTX-2.5 is validated for at 480p/720p.
ALLOWED_FPS: tuple[int, ...] = (24, 25)

#: Longest clip LTX-2.5 supports at these resolutions, in seconds.
MAX_DURATION_SECONDS: dict[str, int] = {"480p": 20, "720p": 20}

MIN_DURATION_SECONDS = 1

FRAME_COUNT_MODULUS = 8
FRAME_COUNT_REMAINDER = 1
DIMENSION_MULTIPLE = 32

#: Sigma schedules exposed by name in config so presets never carry raw float lists.
SIGMA_SCHEDULE_NAMES = ("distilled", "distilled_stage2")

DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "watermark, text, low resolution"
)


def resolution_rank(resolution: str) -> int:
    """Position of ``resolution`` in :data:`RESOLUTION_ORDER`."""
    try:
        return RESOLUTION_ORDER.index(resolution)  # type: ignore[arg-type]
    except ValueError as exc:  # pragma: no cover - guarded by enum validation
        raise ValueError(f"Unsupported resolution: {resolution!r}") from exc


def dimensions_for(resolution: str, aspect_ratio: str) -> tuple[int, int]:
    """Return ``(width, height)`` for a supported resolution/aspect pair."""
    try:
        return RESOLUTION_MATRIX[(resolution, aspect_ratio)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported resolution/aspect combination: {resolution} {aspect_ratio}"
        ) from exc


def snap_frame_count(num_frames: int) -> int:
    """Round ``num_frames`` down to the nearest value satisfying ``n % 8 == 1``."""
    if num_frames < 9:
        return 9
    return ((num_frames - FRAME_COUNT_REMAINDER) // FRAME_COUNT_MODULUS) * FRAME_COUNT_MODULUS + FRAME_COUNT_REMAINDER


def frames_for_duration(duration_seconds: float, fps: int) -> int:
    """Frame count for a clip length, snapped to the model's ``n % 8 == 1`` rule."""
    return snap_frame_count(round(duration_seconds * fps))


def duration_for_frames(num_frames: int, fps: int) -> float:
    return round(num_frames / fps, 3)


def snap_dimension(value: int) -> int:
    """Round ``value`` down to a multiple of 32 (minimum 32)."""
    return max(DIMENSION_MULTIPLE, (value // DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE)
