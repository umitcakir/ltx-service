"""Configuration loading.

Every operational setting (paths, device, concurrency, presets) lives in
``config.yaml``. The REST API never accepts these values from clients.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.constants import (
    ALLOWED_FPS,
    DEFAULT_NEGATIVE_PROMPT,
    MAX_DURATION_SECONDS,
    RESOLUTION_ORDER,
    resolution_rank,
)

DEFAULT_CONFIG_FILENAME = "config.yaml"

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(RuntimeError):
    """Raised when config.yaml is missing or invalid."""


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in strings."""
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            resolved = os.environ.get(name)
            if resolved is None:
                if default is None:
                    raise ConfigError(
                        f"Environment variable {name!r} referenced in config is not set"
                    )
                return default
            return resolved

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    #: Uvicorn workers. Must stay 1 — the model is held in a single process.
    workers: int = Field(default=1, ge=1, le=1)
    log_level: str = "info"
    #: Seconds to let an in-flight job finish during graceful shutdown.
    shutdown_grace_seconds: float = Field(default=60.0, ge=0)


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_output_dir: Path = Path("./outputs")
    allowed_output_dirs: list[Path] = Field(default_factory=lambda: [Path("./outputs")])
    filename_pattern: str = "{date}_{uuid}_{resolution}.mp4"
    allowed_formats: list[str] = Field(default_factory=lambda: ["mp4"])
    dated_subfolders: bool = True
    dated_subfolder_pattern: str = "%Y-%m-%d"
    #: Max length of a client-supplied filename stem after sanitisation.
    max_filename_length: int = Field(default=120, ge=8, le=255)

    @field_validator("allowed_formats")
    @classmethod
    def _normalise_formats(cls, value: list[str]) -> list[str]:
        formats = [fmt.lower().lstrip(".") for fmt in value if fmt.strip()]
        if not formats:
            raise ValueError("storage.allowed_formats must not be empty")
        return formats

    @model_validator(mode="after")
    def _pattern_has_placeholders(self) -> "StorageConfig":
        if "{uuid}" not in self.filename_pattern:
            raise ValueError("storage.filename_pattern must contain '{uuid}'")
        suffix = Path(self.filename_pattern).suffix.lstrip(".").lower()
        if suffix not in self.allowed_formats:
            raise ValueError(
                f"storage.filename_pattern extension '{suffix}' is not in allowed_formats"
            )
        return self


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_id: str = "Lightricks/LTX-2.5-Diffusers"
    #: Local directory the Diffusers pack is downloaded into / loaded from.
    checkpoint_path: Path = Path("./models/ltx-2.5-diffusers")
    #: Download the pack on startup when checkpoint_path is missing.
    auto_download: bool = True
    #: Extra single-file assets (LoRAs etc.) fetched from this repo.
    assets_repo_id: str = "Lightricks/LTX-2.5"
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    #: "none" keeps everything resident; "model" and "sequential" stream weights
    #: from system RAM. A 22B transformer does not fit in 16 GB at bf16 (~44 GB)
    #: or int8 (~22 GB), so "sequential" is the only workable mode on such cards.
    cpu_offload: Literal["none", "model", "sequential"] = "sequential"
    #: int8 weight-only quantization of the transformer + text encoder. Halves the
    #: per-step RAM->VRAM streaming under sequential offload at some quality cost.
    quantization: Literal["none", "int8"] = "none"
    #: Components to quantize when quantization is enabled.
    quantize_components: list[str] = Field(
        default_factory=lambda: ["transformer", "text_encoder"]
    )
    vae_tiling: bool = True
    vae_slicing: bool = True
    #: Subfolder of the Diffusers pack holding the x2 latent upsampler.
    latent_upsampler_subfolder: str = "latent_upsampler"
    #: Load the upsampler so presets can opt into two-stage generation.
    enable_latent_upsampler: bool = True
    #: Skip model loading entirely — useful for smoke-testing the API surface.
    dry_run: bool = False


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_resolution: str = "720p"
    max_resolution: str = "720p"
    default_aspect_ratio: str = "16:9"
    default_fps: int = 24
    allowed_fps: list[int] = Field(default_factory=lambda: list(ALLOWED_FPS))
    default_duration_seconds: int = 5
    max_duration_seconds: int = 20
    default_negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    default_preset: str = "fast"
    generate_audio_by_default: bool = False
    #: Allow start/end conditioning images to be fetched over http(s).
    allow_remote_images: bool = True
    max_image_download_bytes: int = Field(default=32 * 1024 * 1024, ge=1)
    #: Directory clients may reference with ``start_image_path`` (relative only).
    input_image_dir: Path = Path("./inputs")

    @field_validator("default_resolution", "max_resolution")
    @classmethod
    def _known_resolution(cls, value: str) -> str:
        if value not in RESOLUTION_ORDER:
            raise ValueError(
                f"resolution must be one of {list(RESOLUTION_ORDER)}, got {value!r}"
            )
        return value

    @model_validator(mode="after")
    def _coherent(self) -> "GenerationConfig":
        if resolution_rank(self.default_resolution) > resolution_rank(self.max_resolution):
            raise ValueError("generation.default_resolution exceeds max_resolution")
        if not self.allowed_fps:
            raise ValueError("generation.allowed_fps must not be empty")
        unsupported = sorted(set(self.allowed_fps) - set(ALLOWED_FPS))
        if unsupported:
            raise ValueError(
                f"generation.allowed_fps contains values outside the LTX-2.5 matrix: {unsupported}"
            )
        if self.default_fps not in self.allowed_fps:
            raise ValueError("generation.default_fps must be listed in allowed_fps")
        ceiling = MAX_DURATION_SECONDS[self.max_resolution]
        if self.max_duration_seconds > ceiling:
            raise ValueError(
                f"generation.max_duration_seconds exceeds the LTX-2.5 limit of {ceiling}s "
                f"at {self.max_resolution}"
            )
        if self.default_duration_seconds > self.max_duration_seconds:
            raise ValueError(
                "generation.default_duration_seconds exceeds max_duration_seconds"
            )
        return self


class LoraProfile(BaseModel):
    """A named generation profile, optionally backed by a LoRA file.

    ``path: null`` means "base model only" — used for the ``quality`` preset.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    path: Path | None = None
    default_weight: float = Field(default=1.0, ge=0.0, le=4.0)
    num_inference_steps: int = Field(default=8, ge=1, le=200)
    guidance_scale: float = Field(default=1.0, ge=0.0, le=20.0)
    #: Named sigma schedule from ``diffusers.pipelines.ltx2.utils`` (or null).
    sigma_schedule: Literal["distilled", "none"] = "none"
    #: Run stage-1 at half resolution then x2 latent upsample + refine.
    multi_stage: bool = False
    stage2_num_inference_steps: int = Field(default=4, ge=1, le=100)
    #: Surfaced in /loras so clients can pick sensibly.
    is_default: bool = False

    @field_validator("name")
    @classmethod
    def _slug(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", value):
            raise ValueError(
                "lora name must be lowercase alphanumeric with '.', '_' or '-' (max 64 chars)"
            )
        return value


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: GPU job semaphore. 1 is safe; 2 is viable for 480p/720p with headroom.
    max_concurrent_jobs: int = Field(default=1, ge=1, le=8)
    reject_when_busy: bool = True
    job_timeout_seconds: float = Field(default=1800.0, gt=0)
    max_queue_size: int = Field(default=32, ge=1)
    #: How long finished jobs stay queryable via /jobs/{id}.
    job_retention_seconds: float = Field(default=86400.0, gt=0)
    #: Exit with RESTART_EXIT_CODE after a sticky CUDA fault so start.sh/systemd relaunches.
    restart_on_cuda_context_lost: bool = True


class PromptEnhancerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    base_url: str = "http://192.168.2.117:1234/v1"
    model: str = "ministral-3-8b-instruct-2512"
    timeout_seconds: float = Field(default=30.0, gt=0, le=120)


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: Path = Path("./logs")
    level: str = "INFO"
    max_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    backup_count: int = Field(default=10, ge=0)
    #: Per-job log files under ``dir/jobs/``.
    per_job_files: bool = True


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: ServerConfig = Field(default_factory=ServerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    loras: list[LoraProfile] = Field(default_factory=list)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    prompt_enhancer: PromptEnhancerConfig = Field(default_factory=PromptEnhancerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    #: Absolute path of the file this config was loaded from (None when built in-memory).
    source_path: Path | None = None

    @model_validator(mode="after")
    def _validate_registry(self) -> "AppConfig":
        names = [profile.name for profile in self.loras]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate lora profile names: {sorted(duplicates)}")
        if names and self.generation.default_preset not in names:
            raise ValueError(
                f"generation.default_preset {self.generation.default_preset!r} "
                f"is not a registered lora profile ({names})"
            )
        if self.model.dry_run is False and any(
            p.multi_stage for p in self.loras
        ) and not self.model.enable_latent_upsampler:
            raise ValueError(
                "a lora profile requests multi_stage but model.enable_latent_upsampler is false"
            )
        return self

    def profile(self, name: str) -> LoraProfile | None:
        for entry in self.loras:
            if entry.name == name:
                return entry
        return None

    @property
    def default_profile(self) -> LoraProfile | None:
        return self.profile(self.generation.default_preset)


def _resolve_paths(config: AppConfig, base_dir: Path) -> AppConfig:
    """Make every configured path absolute, relative to the config file."""

    def absolute(path: Path) -> Path:
        return path if path.is_absolute() else (base_dir / path)

    # ``Path.resolve()`` is used consistently so later containment checks compare
    # fully-normalised, symlink-free paths.
    config.storage.default_output_dir = absolute(config.storage.default_output_dir)
    config.storage.allowed_output_dirs = [
        absolute(path) for path in config.storage.allowed_output_dirs
    ]
    config.model.checkpoint_path = absolute(config.model.checkpoint_path)
    config.generation.input_image_dir = absolute(config.generation.input_image_dir)
    config.logging.dir = absolute(config.logging.dir)
    for profile in config.loras:
        if profile.path is not None:
            profile.path = absolute(profile.path)

    default_dir = config.storage.default_output_dir
    if not any(
        default_dir == allowed or _is_relative_to(default_dir, allowed)
        for allowed in config.storage.allowed_output_dirs
    ):
        raise ConfigError(
            "storage.default_output_dir must be inside one of storage.allowed_output_dirs"
        )
    return config


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    """Read, env-expand and validate ``config.yaml``."""
    config_path = Path(path) if path else Path(DEFAULT_CONFIG_FILENAME)
    config_path = config_path.expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()

    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")

    raw = _expand_env(raw)
    raw.pop("source_path", None)

    try:
        config = AppConfig.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError and friends
        raise ConfigError(f"Invalid config at {config_path}: {exc}") from exc

    config.source_path = config_path
    return _resolve_paths(config, config_path.parent)


@lru_cache(maxsize=1)
def _cached_config(path: str) -> AppConfig:
    return load_config(path)


def get_config() -> AppConfig:
    """Process-wide config, selected by the ``LTX_SERVICE_CONFIG`` env var."""
    return _cached_config(os.environ.get("LTX_SERVICE_CONFIG", DEFAULT_CONFIG_FILENAME))


def reset_config_cache() -> None:
    _cached_config.cache_clear()
