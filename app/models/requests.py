"""Client-facing request schemas.

Only content/generation parameters are accepted. Output directories, model
paths, precision and concurrency are config-only — see ``app/config.py``.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from app.constants import (
    ALLOWED_FPS,
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
    RESOLUTION_ORDER,
)

_FILENAME_STEM_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,190})$")


class LoraSelection(BaseModel):
    """Pick a named profile from the config registry. File paths are never exposed."""

    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(min_length=1, max_length=64)]
    weight: Annotated[float | None, Field(default=None, ge=0.0, le=4.0)] = None


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: Annotated[str, Field(min_length=1, max_length=8000)]
    negative_prompt: Annotated[str | None, Field(default=None, max_length=4000)] = None

    start_image_url: HttpUrl | None = None
    #: Relative path under the configured ``generation.input_image_dir``.
    start_image_path: Annotated[str | None, Field(default=None, max_length=512)] = None
    end_image_url: HttpUrl | None = None
    end_image_path: Annotated[str | None, Field(default=None, max_length=512)] = None

    #: Clip length in seconds. ``None`` lets LTX-2.5 pick via the duration head.
    duration: Annotated[
        int | None, Field(default=None, ge=MIN_DURATION_SECONDS, le=max(MAX_DURATION_SECONDS.values()))
    ] = None
    resolution: Literal["480p", "720p"] | None = None
    aspect_ratio: Literal["16:9", "9:16"] | None = None
    fps: int | None = None
    seed: Annotated[int, Field(default=-1, ge=-1, le=2**31 - 1)] = -1
    generate_audio: bool | None = None

    lora: LoraSelection | None = None
    #: Filename only (no directories). Extension must be an allowed output format.
    output_filename: Annotated[str | None, Field(default=None, max_length=200)] = None

    @field_validator("prompt")
    @classmethod
    def _non_blank_prompt(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("prompt must not be blank")
        return stripped

    @field_validator("fps")
    @classmethod
    def _supported_fps(cls, value: int | None) -> int | None:
        if value is not None and value not in ALLOWED_FPS:
            raise ValueError(
                f"fps must be one of {list(ALLOWED_FPS)} for 480p/720p LTX-2.5 generation"
            )
        return value

    @field_validator("resolution")
    @classmethod
    def _supported_resolution(cls, value: str | None) -> str | None:
        if value is not None and value not in RESOLUTION_ORDER:
            raise ValueError(f"resolution must be one of {list(RESOLUTION_ORDER)}")
        return value

    @field_validator("start_image_path", "end_image_path")
    @classmethod
    def _relative_image_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        candidate = value.strip().replace("\\", "/")
        if not candidate:
            raise ValueError("image path must not be blank")
        if candidate.startswith("/") or re.match(r"^[A-Za-z]:", candidate):
            raise ValueError("absolute image paths are not accepted; use a path relative to the configured input dir")
        if ".." in candidate.split("/"):
            raise ValueError("image path must not contain '..'")
        return candidate

    @field_validator("output_filename")
    @classmethod
    def _filename_shape(cls, value: str | None) -> str | None:
        if value is None:
            return None
        candidate = value.strip()
        if not candidate:
            return None
        if "/" in candidate or "\\" in candidate or candidate in {".", ".."}:
            raise ValueError("output_filename must be a bare filename without directories")
        stem, _, suffix = candidate.rpartition(".")
        if not suffix or not stem:
            raise ValueError("output_filename must include a file extension, e.g. 'clip.mp4'")
        if not _FILENAME_STEM_RE.match(stem):
            raise ValueError(
                "output_filename may only contain letters, digits, '.', '_' and '-' "
                "and must start with a letter or digit"
            )
        return candidate

    @model_validator(mode="after")
    def _one_source_per_conditioning_slot(self) -> "GenerateRequest":
        if self.start_image_url and self.start_image_path:
            raise ValueError("provide either start_image_url or start_image_path, not both")
        if self.end_image_url and self.end_image_path:
            raise ValueError("provide either end_image_url or end_image_path, not both")
        if (self.end_image_url or self.end_image_path) and not (
            self.start_image_url or self.start_image_path
        ):
            raise ValueError("an end image requires a start image")
        return self
