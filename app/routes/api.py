"""REST endpoints."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app import __version__
from app.models import GenerateRequest, GenerateResponse, HealthResponse, JobStatus, JobStatusResponse, LoraListResponse, LoraProfileInfo
from app.utils.gpu import vram_snapshot

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse, status_code=202)
async def generate(body: GenerateRequest, request: Request):
    manager = request.app.state.jobs
    try:
        return manager.submit(body)
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc)) from exc
    except asyncio.QueueFull as exc:
        raise HTTPException(429, detail="job queue is full") from exc


def job_or_404(request: Request, job_id: str) -> JobStatusResponse:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, detail="job not found")
    return job


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, request: Request):
    return job_or_404(request, job_id)


@router.get("/jobs/{job_id}/result")
def get_result(job_id: str, request: Request):
    state = job_or_404(request, job_id)
    if state.status != JobStatus.COMPLETED or state.result is None:
        raise HTTPException(409, detail=f"job is {state.status.value}")
    path = Path(state.result.output_path)
    if not path.is_file():
        raise HTTPException(404, detail="result file no longer exists")
    return FileResponse(path, filename=state.result.filename, media_type="video/mp4")


@router.delete("/jobs/{job_id}", response_model=JobStatusResponse)
async def cancel_job(job_id: str, request: Request):
    state = job_or_404(request, job_id)
    if state.status != JobStatus.QUEUED:
        raise HTTPException(409, detail="only queued jobs can be cancelled")
    return request.app.state.jobs.cancel(job_id)


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    config = request.app.state.config
    model = request.app.state.model
    jobs = request.app.state.jobs
    return HealthResponse(
        status="ready" if model.is_loaded else "unavailable", version=__version__,
        model_loaded=model.is_loaded, model_repo_id=config.model.repo_id,
        device=config.model.device, precision=config.model.precision,
        cpu_offload=config.model.cpu_offload, loaded_lora=model.loaded_lora,
        available_loras=[p.name for p in config.loras if p.path and p.path.is_file()],
        vram=vram_snapshot(config.model.device), queue_length=jobs.queue.qsize(),
        running_jobs=jobs.running_count,
        max_concurrent_jobs=config.runtime.max_concurrent_jobs,
    )


@router.get("/loras", response_model=LoraListResponse)
def loras(request: Request):
    config = request.app.state.config
    return LoraListResponse(
        default=config.generation.default_preset,
        loaded=request.app.state.model.loaded_lora,
        profiles=[LoraProfileInfo(
            name=p.name, description=p.description,
            mode="lora" if p.path else "base", default_weight=p.default_weight,
            num_inference_steps=p.num_inference_steps, guidance_scale=p.guidance_scale,
            multi_stage=p.multi_stage,
            available=p.path is None or p.path.is_file(),
            is_default=p.name == config.generation.default_preset,
        ) for p in config.loras],
    )