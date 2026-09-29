"""In-memory job queue; a single Uvicorn worker owns the model and job state."""

import asyncio
import logging
import secrets
import signal
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.config import AppConfig
from app.constants import dimensions_for, duration_for_frames, frames_for_duration, resolution_rank
from app.models.requests import GenerateRequest
from app.models.responses import GenerateResponse, JobError, JobResult, JobStatus, JobStatusResponse
from app.services.model_manager import GenerationSpec, ModelManager
from app.services.prompt_enhancer import enhance_prompt
from app.utils.gpu import (
    empty_cache,
    is_cuda_context_lost,
    is_cuda_oom,
    peak_vram_bytes,
    reset_peak_vram,
)
from app.utils.images import load_image_from_path, load_image_from_url
from app.utils.logging_setup import job_log_context
from app.utils.paths import OutputPathError, build_filename, resolve_input_image, resolve_output_path

log = logging.getLogger(__name__)

#: Process exit code telling start.sh / systemd that a restart is required.
RESTART_EXIT_CODE = 75
restart_requested = False


def format_elapsed(seconds: float) -> str:
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining_seconds:06.3f}s"


class JobBusyError(Exception):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def validate_request(request: GenerateRequest, config: AppConfig) -> None:
    resolution = request.resolution or config.generation.default_resolution
    fps = request.fps or config.generation.default_fps
    if resolution_rank(resolution) > resolution_rank(config.generation.max_resolution):
        raise ValueError("resolution exceeds configured maximum")
    if fps not in config.generation.allowed_fps:
        raise ValueError("fps is not allowed by configuration")
    if request.duration is not None and request.duration > config.generation.max_duration_seconds:
        raise ValueError("duration exceeds the configured/model maximum")
    if request.lora and config.profile(request.lora.name) is None:
        raise ValueError(f"unknown preset: {request.lora.name}")
    profile = config.profile(request.lora.name) if request.lora else config.default_profile
    if request.lora and request.lora.weight is not None and profile and profile.path is None:
        raise ValueError("weight override requires a LoRA-backed preset")
    if profile and profile.path and not profile.path.is_file():
        raise ValueError(f"LoRA preset '{profile.name}' is not installed")
    if (request.start_image_url or request.end_image_url) and not config.generation.allow_remote_images:
        raise ValueError("remote conditioning images are disabled")
    if request.output_filename:
        resolve_output_path(config.storage, filename=request.output_filename, create_parents=False)


@dataclass
class Job:
    request: GenerateRequest
    state: JobStatusResponse


class JobManager:
    def __init__(self, config: AppConfig, model: ModelManager):
        self.config = config
        self.model = model
        self.jobs: dict[str, Job] = {}
        self.queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=config.runtime.max_queue_size)
        self.workers: list[asyncio.Task] = []
        self.closing = False
        # Set when a CUDA fault poisons the context; only a restart clears it.
        self._cuda_context_lost = False

    @property
    def cuda_context_lost(self) -> bool:
        return self._cuda_context_lost

    async def start(self) -> None:
        self.workers = [asyncio.create_task(self._worker()) for _ in range(self.config.runtime.max_concurrent_jobs)]

    async def stop(self) -> None:
        self.closing = True
        try:
            await asyncio.wait_for(self.queue.join(), self.config.server.shutdown_grace_seconds)
        except asyncio.TimeoutError:
            log.warning("Shutdown grace expired; waiting for running inference to return")
        for _ in self.workers:
            await self.queue.put(None)
        await asyncio.gather(*self.workers)

    def submit(self, request: GenerateRequest) -> GenerateResponse:
        if self.closing:
            raise RuntimeError("server is shutting down")
        if self._cuda_context_lost:
            raise RuntimeError(
                "the CUDA context was lost by a previous job; restart the service"
            )
        validate_request(request, self.config)
        if self.config.runtime.reject_when_busy and any(
            job.state.status in (JobStatus.QUEUED, JobStatus.RUNNING) for job in self.jobs.values()
        ):
            raise JobBusyError("another generation is in progress")
        job_id = uuid.uuid4().hex
        state = JobStatusResponse(job_id=job_id, status=JobStatus.QUEUED, progress=0, stage="queued", created_at=utcnow())
        self.queue.put_nowait(job_id)
        self.jobs[job_id] = Job(request, state)
        return GenerateResponse(job_id=job_id, queue_position=self.queue.qsize(), created_at=state.created_at)

    def get(self, job_id: str) -> JobStatusResponse | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        if job.state.status == JobStatus.QUEUED:
            try:
                job.state.queue_position = list(self.queue._queue).index(job_id) + 1
            except ValueError:
                job.state.queue_position = None
        return job.state

    def cancel(self, job_id: str) -> JobStatusResponse | None:
        state = self.get(job_id)
        if state and state.status == JobStatus.QUEUED:
            state.status = JobStatus.CANCELLED
            state.stage = "cancelled"
            state.finished_at = utcnow()
        return state

    @property
    def running_count(self) -> int:
        return sum(job.state.status == JobStatus.RUNNING for job in self.jobs.values())

    async def _worker(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                if job_id is None:
                    return
                job = self.jobs[job_id]
                if job.state.status == JobStatus.CANCELLED:
                    continue
                job.state.status = JobStatus.RUNNING
                job.state.started_at = utcnow()
                job.state.stage = "preparing"
                await asyncio.to_thread(self._execute, job)
                if self._cuda_context_lost:
                    self._request_restart()
            finally:
                self.queue.task_done()

    def _request_restart(self) -> None:
        global restart_requested
        if restart_requested or not self.config.runtime.restart_on_cuda_context_lost:
            return
        restart_requested = True
        log.critical("CUDA context lost; shutting down so the service can be restarted (exit code %d)", RESTART_EXIT_CODE)
        signal.raise_signal(signal.SIGTERM)

    def _load_image(self, url: object, path: str | None):
        if url:
            return load_image_from_url(str(url), max_bytes=self.config.generation.max_image_download_bytes)
        if path:
            resolved = resolve_input_image(path, self.config.generation.input_image_dir)
            return load_image_from_path(resolved, max_bytes=self.config.generation.max_image_download_bytes)
        return None

    def _execute(self, job: Job) -> None:
        state, request = job.state, job.request
        started = time.monotonic()
        deadline = started + self.config.runtime.job_timeout_seconds
        profile = self.config.profile(request.lora.name) if request.lora else self.config.default_profile
        if profile is None:
            state.status = JobStatus.FAILED
            state.error = JobError(code="CONFIG_ERROR", message="No generation preset configured")
            state.finished_at = utcnow()
            return
        output_path: Path | None = None
        with job_log_context(self.config.logging, state.job_id) as job_log:
            job_log.info("starting preset=%s params=%s", profile.name, request.model_dump_json(exclude={"start_image_url", "end_image_url"}))
            try:
                if self.config.prompt_enhancer.enabled:
                    state.stage = "enhancing_prompt"
                prompt = enhance_prompt(request.prompt, self.config.prompt_enhancer)
                start_image = self._load_image(request.start_image_url, request.start_image_path)
                end_image = self._load_image(request.end_image_url, request.end_image_path)
                resolution = request.resolution or self.config.generation.default_resolution
                aspect = request.aspect_ratio or self.config.generation.default_aspect_ratio
                fps = request.fps or self.config.generation.default_fps
                width, height = dimensions_for(resolution, aspect)
                frames = frames_for_duration(request.duration, fps) if request.duration else None
                seed = request.seed if request.seed >= 0 else secrets.randbelow(2**31)
                weight = request.lora.weight if request.lora and request.lora.weight is not None else profile.default_weight
                spec = GenerationSpec(
                    prompt=prompt,
                    negative_prompt=request.negative_prompt or self.config.generation.default_negative_prompt,
                    width=width, height=height, num_frames=frames, fps=fps, seed=seed,
                    guidance_scale=profile.guidance_scale,
                    num_inference_steps=profile.num_inference_steps,
                    sigma_schedule=profile.sigma_schedule,
                    multi_stage=profile.multi_stage and width % 64 == 0 and height % 64 == 0,
                    stage2_num_inference_steps=profile.stage2_num_inference_steps,
                    generate_audio=request.generate_audio if request.generate_audio is not None else self.config.generation.generate_audio_by_default,
                    start_image=start_image, end_image=end_image,
                )
                filename = request.output_filename or build_filename(self.config.storage, resolution=resolution, job_id=state.job_id)
                output_path = resolve_output_path(self.config.storage, filename=filename)
                if output_path.exists() or output_path.is_symlink():
                    raise OutputPathError("output filename already exists")

                def progress(step: int, total: int, stage: str) -> None:
                    if time.monotonic() > deadline:
                        raise TimeoutError("generation exceeded the configured job timeout")
                    state.stage = stage
                    state.progress = min(0.95, step / max(1, total) * 0.9)

                if time.monotonic() > deadline:
                    raise TimeoutError("job timed out during preparation")
                reset_peak_vram()
                generated = self.model.generate(spec, profile, weight, progress)
                progress(1, 1, "encoding")
                from diffusers.utils import encode_video

                encode_video(
                    generated.video, fps=fps, output_path=str(output_path),
                    audio=generated.audio.float().cpu() if generated.audio is not None else None,
                    audio_sample_rate=generated.audio_sample_rate if generated.audio is not None else None,
                )
                from app.utils.videos import make_faststart

                make_faststart(output_path)
                duration = duration_for_frames(generated.num_frames, fps)
                state.result = JobResult(
                    output_path=str(output_path), filename=filename, format=output_path.suffix.lstrip("."),
                    file_size_bytes=output_path.stat().st_size, resolution=resolution,
                    width=generated.width, height=generated.height, aspect_ratio=aspect,
                    duration_seconds=duration, num_frames=generated.num_frames, fps=fps, seed=seed,
                    lora=profile.name if profile.path else None,
                    lora_weight=weight if profile.path else None,
                    num_inference_steps=profile.num_inference_steps, guidance_scale=profile.guidance_scale,
                    multi_stage=spec.multi_stage, audio=generated.audio is not None,
                    generation_time_seconds=round(time.monotonic() - started, 3),
                    peak_vram_bytes=peak_vram_bytes(),
                )
                state.status = JobStatus.COMPLETED
                state.progress = 1.0
                state.stage = "completed"
                job_log.info("completed in %s peak_vram=%s path=%s", format_elapsed(state.result.generation_time_seconds), state.result.peak_vram_bytes, output_path)
            except Exception as exc:
                if output_path is not None and output_path.is_file():
                    output_path.unlink(missing_ok=True)
                if isinstance(exc, TimeoutError):
                    state.status, code = JobStatus.TIMEOUT, "TIMEOUT"
                elif is_cuda_oom(exc):
                    state.status, code = JobStatus.FAILED, "CUDA_OOM"
                    empty_cache()
                elif is_cuda_context_lost(exc):
                    state.status, code = JobStatus.FAILED, "CUDA_CONTEXT_LOST"
                    self._cuda_context_lost = True
                else:
                    state.status, code = JobStatus.FAILED, "GENERATION_FAILED"
                state.stage = "failed"
                state.error = JobError(code=code, message=str(exc)[:500])
                job_log.exception("job failed: %s elapsed=%s (peak_vram=%s)", code, format_elapsed(time.monotonic() - started), peak_vram_bytes())
            finally:
                state.finished_at = utcnow()