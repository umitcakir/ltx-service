"""Response schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

    @property
    def is_terminal(self) -> bool:
        return self in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMEOUT,
        }


class JobResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_path: str
    filename: str
    format: str
    file_size_bytes: int
    resolution: str
    width: int
    height: int
    aspect_ratio: str
    duration_seconds: float
    num_frames: int
    fps: int
    seed: int
    lora: str | None
    lora_weight: float | None
    num_inference_steps: int
    guidance_scale: float
    multi_stage: bool
    audio: bool
    generation_time_seconds: float
    peak_vram_bytes: int | None = None


class JobError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    detail: str | None = None


class GenerateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: JobStatus = JobStatus.QUEUED
    queue_position: int
    created_at: datetime


class JobStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    stage: str | None = None
    queue_position: int | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: JobResult | None = None
    error: JobError | None = None


class LoraProfileInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    mode: str
    default_weight: float
    num_inference_steps: int
    guidance_scale: float
    multi_stage: bool
    available: bool
    is_default: bool


class LoraListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: str
    loaded: str | None
    profiles: list[LoraProfileInfo]


class VramInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_bytes: int | None = None
    used_bytes: int | None = None
    free_bytes: int | None = None
    reserved_bytes: int | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    version: str
    model_loaded: bool
    model_repo_id: str
    device: str
    precision: str
    cpu_offload: str
    loaded_lora: str | None
    available_loras: list[str]
    vram: VramInfo
    queue_length: int
    running_jobs: int
    max_concurrent_jobs: int


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    detail: str | None = None
