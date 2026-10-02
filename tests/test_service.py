from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
import subprocess
import sys
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError
from imageio_ffmpeg import get_ffmpeg_exe

from app.config import ConfigError, PromptEnhancerConfig, load_config
from app.main import create_app
from app.models.requests import GenerateRequest
from app.models.responses import JobResult, JobStatus, JobStatusResponse
from app.services.jobs import Job, JobBusyError, JobManager, format_elapsed, utcnow, validate_request
from app.services.model_manager import ModelManager
from app.services.model_manager import GenerationOutput
from app.services.prompt_enhancer import enhance_prompt
from app.utils.paths import OutputPathError, resolve_output_path
from app.utils.videos import make_faststart


ROOT = Path(__file__).resolve().parents[1]


def test_load_example_config():
    config = load_config(ROOT / "config.yaml")
    assert config.server.host == "0.0.0.0"
    assert config.prompt_enhancer.model == "ministral-3-8b-instruct-2512"
    assert config.prompt_enhancer.base_url == "http://192.168.2.117:1234/v1"
    assert config.model.repo_id == "Lightricks/LTX-2.5-Diffusers"
    assert config.profile("fast").path is None
    assert config.storage.default_output_dir == ROOT / "outputs"


def test_config_loads_without_api_key(monkeypatch):
    monkeypatch.delenv("LTX_API_KEY", raising=False)
    assert load_config(ROOT / "config.yaml").server.port == 8000


def test_config_rejects_outside_output_dir(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "storage:\n  default_output_dir: ../outside\n"
        "  allowed_output_dirs: [./outputs]\n"
    )
    with pytest.raises(ConfigError, match="allowed_output_dirs"):
        load_config(config_file)


@pytest.mark.parametrize("payload", [
    {"prompt": ""}, {"prompt": "cat", "fps": 30},
    {"prompt": "cat", "resolution": "1080p"},
    {"prompt": "cat", "duration": 21},
    {"prompt": "cat", "output_dir": "/tmp"},
    {"prompt": "cat", "output_filename": "../escape.mp4"},
    {"prompt": "cat", "start_image_path": "/etc/passwd"},
    {"prompt": "cat", "end_image_path": "end.jpg"},
])
def test_invalid_generation_requests(payload):
    with pytest.raises(ValidationError):
        GenerateRequest.model_validate(payload)


def test_config_boundaries():
    config = load_config(ROOT / "config.yaml")
    validate_request(GenerateRequest(prompt="cat", duration=20, fps=25), config)
    with pytest.raises(ValueError, match="unknown preset"):
        validate_request(GenerateRequest(prompt="cat", lora={"name": "unknown"}), config)
    config.generation.allow_remote_images = False
    with pytest.raises(ValueError, match="remote conditioning"):
        validate_request(GenerateRequest(prompt="cat", start_image_url="https://example.org/a.png"), config)


def test_reject_when_busy_queued_and_cancelled():
    config = load_config(ROOT / "config.yaml")
    manager = JobManager(config, FakeModel())
    first = manager.submit(GenerateRequest(prompt="first"))
    with pytest.raises(JobBusyError, match="another generation is in progress"):
        manager.submit(GenerateRequest(prompt="second"))
    assert len(manager.jobs) == 1
    manager.cancel(first.job_id)
    manager.submit(GenerateRequest(prompt="third"))
    config.runtime.reject_when_busy = False
    manager.submit(GenerateRequest(prompt="fourth"))
    assert len(manager.jobs) == 3


def test_model_load_enables_temporal_vae_tiling(monkeypatch, tmp_path):
    config = load_config(ROOT / "config.yaml")
    config.model.enable_latent_upsampler = False
    vae = SimpleNamespace(use_framewise_decoding=False, enable_tiling=lambda: None, enable_slicing=lambda: None)
    pipeline = SimpleNamespace(vae=vae, enable_sequential_cpu_offload=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(LTX2Pipeline=SimpleNamespace(from_pretrained=lambda *args, **kwargs: pipeline)))
    monkeypatch.setattr("app.services.model_manager.resolve_dtype", lambda precision: precision)
    monkeypatch.setattr(ModelManager, "_ensure_checkpoint", lambda self: tmp_path)
    monkeypatch.setattr(ModelManager, "_load_sigma_schedules", staticmethod(lambda: {}))

    manager = ModelManager(config)
    manager.load()
    assert vae.use_framewise_decoding is True


def test_output_path_and_symlink_escape(tmp_path):
    from app.config import StorageConfig

    root = tmp_path / "outputs"
    root.mkdir()
    storage = StorageConfig(default_output_dir=root, allowed_output_dirs=[root], dated_subfolders=False)
    assert resolve_output_path(storage, filename="scene.mp4") == root / "scene.mp4"
    with pytest.raises(OutputPathError):
        resolve_output_path(storage, filename="scene.exe")
    (root / "link.mp4").symlink_to(tmp_path / "missing.mp4")
    with pytest.raises(OutputPathError):
        resolve_output_path(storage, filename="link.mp4")
    escape = tmp_path / "elsewhere"
    escape.mkdir()
    storage.default_output_dir = root / "redirect"
    (root / "redirect").symlink_to(escape, target_is_directory=True)
    with pytest.raises(OutputPathError):
        resolve_output_path(storage, filename="scene.mp4")


def test_faststart_mp4(tmp_path, monkeypatch):
    path = tmp_path / "sample.mp4"
    subprocess.run(
        [get_ffmpeg_exe(), "-loglevel", "error", "-f", "lavfi", "-i", "color=s=32x32:r=1", "-t", "1", "-c:v", "mpeg4", str(path)],
        check=True,
    )
    assert path.read_bytes().index(b"mdat") < path.read_bytes().index(b"moov")
    monkeypatch.setenv("PATH", "")
    make_faststart(path)
    assert path.read_bytes().index(b"moov") < path.read_bytes().index(b"mdat")


def test_remote_prompt_enhancement(monkeypatch):
    def fake_post(url, **kwargs):
        assert url == "http://192.168.2.117:1234/v1/chat/completions"
        assert "headers" not in kwargs
        assert kwargs["json"]["model"] == "ministral-3-8b-instruct-2512"
        return httpx.Response(200, json={"choices": [{"message": {"content": "  A cinematic cat at dusk  "}}]}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    config = PromptEnhancerConfig(enabled=True)
    assert enhance_prompt("cat", config) == "A cinematic cat at dusk"
    assert enhance_prompt("cat", PromptEnhancerConfig(enabled=False)) == "cat"


class FakeModel:
    is_loaded = True
    loaded_lora = None

    def load(self):
        pass

    def unload(self):
        pass

    def generate(self, *args, **kwargs):
        raise RuntimeError("test model has no inference")


def test_image_upload_for_generation(tmp_path):
    config = load_config(ROOT / "config.yaml")
    config.generation.input_image_dir = tmp_path / "inputs"
    config.generation.max_image_download_bytes = 1024
    config.logging.dir = tmp_path / "logs"
    image_data = BytesIO()
    Image.new("RGB", (2, 2), "red").save(image_data, format="PNG")
    with TestClient(create_app(config, model=FakeModel())) as client:
        response = client.post("/images", content=image_data.getvalue(), headers={"content-type": "image/png"})
        assert response.status_code == 201
        filename = response.json()["path"]
        assert (config.generation.input_image_dir / filename).read_bytes() == image_data.getvalue()
        assert client.post("/generate", json={"prompt": "cat", "start_image_path": filename}).status_code == 202
        assert client.post("/images", content=b"not an image").status_code == 422
        assert client.post("/images", content=b"x" * 1025).status_code == 413
        assert list(config.generation.input_image_dir.iterdir()) == [config.generation.input_image_dir / filename]


def test_endpoints_without_authentication(tmp_path):
    config = load_config(ROOT / "config.yaml")
    config.logging.dir = tmp_path / "logs"
    config.storage.default_output_dir = tmp_path / "outputs"
    config.storage.allowed_output_dirs = [tmp_path / "outputs"]
    with TestClient(create_app(config, model=FakeModel())) as client:
        assert client.get("/health").json()["model_loaded"] is True
        profiles = client.get("/loras").json()["profiles"]
        assert {p["name"] for p in profiles} == {"fast", "quality"}
        response = client.post("/generate", json={"prompt": "cat", "resolution": "720p"})
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        assert client.get(f"/jobs/{job_id}").status_code == 200
        assert client.get(f"/jobs/{job_id}/result").status_code == 409
        invalid = client.post("/generate", json={"prompt": "cat", "output_filename": "../escape.mp4"})
        assert invalid.status_code == 422


def test_generate_rejects_during_running_job(tmp_path):
    config = load_config(ROOT / "config.yaml")
    config.logging.dir = tmp_path / "logs"
    with TestClient(create_app(config, model=FakeModel())) as client:
        manager = client.app.state.jobs
        running = JobStatusResponse(job_id="existing", status=JobStatus.RUNNING, progress=0.5, created_at=utcnow())
        manager.jobs[running.job_id] = Job(GenerateRequest(prompt="first"), running)
        response = client.post("/generate", json={"prompt": "second"})
        assert response.status_code == 429
        assert response.json()["detail"] == "another generation is in progress"
        assert len(manager.jobs) == 1
        running.status = JobStatus.COMPLETED
        assert client.post("/generate", json={"prompt": "third"}).status_code == 202


def test_failed_job_logs_elapsed_time(tmp_path, caplog):
    config = load_config(ROOT / "config.yaml")
    config.logging.dir = tmp_path / "logs"
    config.storage.default_output_dir = tmp_path / "outputs"
    config.storage.allowed_output_dirs = [tmp_path / "outputs"]
    state = JobStatusResponse(job_id="timed", status=JobStatus.RUNNING, progress=0, created_at=utcnow())
    manager = JobManager(config, FakeModel())

    manager._execute(Job(GenerateRequest(prompt="cat"), state))

    assert state.status == JobStatus.FAILED
    assert "job failed: GENERATION_FAILED elapsed=" in caplog.text
    assert "elapsed=" in (config.logging.dir / "jobs/timed.log").read_text()


def test_cuda_context_lost_requests_restart(tmp_path, monkeypatch):
    from app.services import jobs as jobs_module

    config = load_config(ROOT / "config.yaml")
    config.logging.dir = tmp_path / "logs"
    model = FakeModel()

    def fail(*args):
        raise RuntimeError("CUDA error: the launch timed out and was terminated")

    model.generate = fail
    signals = []
    monkeypatch.setattr(jobs_module, "restart_requested", False)
    monkeypatch.setattr(jobs_module.signal, "raise_signal", signals.append)
    with TestClient(create_app(config, model=model)) as client:
        job_id = client.post("/generate", json={"prompt": "cat"}).json()["job_id"]
        deadline = time.monotonic() + 5
        while not signals and time.monotonic() < deadline:
            time.sleep(0.05)
        assert client.get(f"/jobs/{job_id}").json()["error"]["code"] == "CUDA_CONTEXT_LOST"
        assert client.get("/health").json()["status"] == "cuda_context_lost"
        assert client.post("/generate", json={"prompt": "cat"}).status_code == 503
    assert signals == [jobs_module.signal.SIGTERM]
    assert jobs_module.restart_requested is True


RESTART_PROBE = """
import signal, sys, threading, time
import uvicorn
from app.services import jobs
import run

async def app(scope, receive, send):
    while True:
        msg = await receive()
        if msg["type"] == "lifespan.startup":
            def trigger():
                time.sleep(0.3)
                jobs.restart_requested = True
                signal.raise_signal(signal.SIGTERM)
            threading.Thread(target=trigger, daemon=True).start()
            await send({"type": "lifespan.startup.complete"})
        else:
            await send({"type": "lifespan.shutdown.complete"})
            return

run.install_restart_handler(jobs)
uvicorn.run(app, host="127.0.0.1", port=0, log_level="warning")
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
def test_restart_request_exits_with_restart_code_under_uvicorn():
    from app.services.jobs import RESTART_EXIT_CODE

    # uvicorn re-raises the captured SIGTERM after shutdown; it must become exit 75.
    result = subprocess.run(
        [sys.executable, "-c", RESTART_PROBE], cwd=ROOT, timeout=30, capture_output=True
    )
    assert result.returncode == RESTART_EXIT_CODE, result.stderr.decode()


def test_completed_job_logs_result_duration(tmp_path, monkeypatch, caplog):
    config = load_config(ROOT / "config.yaml")
    config.logging.dir = tmp_path / "logs"
    config.storage.default_output_dir = tmp_path / "outputs"
    config.storage.allowed_output_dirs = [tmp_path / "outputs"]
    model = FakeModel()
    model.generate = lambda *args: GenerationOutput(
        video=[], audio=None, audio_sample_rate=None, num_frames=25, width=832, height=480
    )
    diffusers = ModuleType("diffusers")
    diffusers.__path__ = []
    utils = ModuleType("diffusers.utils")
    utils.encode_video = lambda video, fps, output_path, **kwargs: Path(output_path).write_bytes(b"mp4")
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.utils", utils)
    monkeypatch.setattr("app.utils.videos.make_faststart", lambda path: None)
    state = JobStatusResponse(job_id="timed", status=JobStatus.RUNNING, progress=0, created_at=utcnow())

    with caplog.at_level("INFO", logger="ltx.job.timed"):
        JobManager(config, model)._execute(Job(GenerateRequest(prompt="cat"), state))

    assert state.status == JobStatus.COMPLETED
    assert state.result is not None
    assert f"completed in {format_elapsed(state.result.generation_time_seconds)}" in (
        config.logging.dir / "jobs/timed.log"
    ).read_text()


def test_elapsed_log_format():
    assert format_elapsed(269.081) == "4m 29.081s"
    assert format_elapsed(9.5) == "0m 09.500s"


def test_completed_result_streams_inline(tmp_path):
    config = load_config(ROOT / "config.yaml")
    config.logging.dir = tmp_path / "logs"
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0123456789")
    with TestClient(create_app(config, model=FakeModel())) as client:
        client.app.state.jobs.get = lambda job_id: SimpleNamespace(
            status=JobStatus.COMPLETED, stage=None,
            result=SimpleNamespace(output_path=str(video), filename=video.name)
        )
        response = client.get("/jobs/example/result", headers={"Range": "bytes=0-3"})
        assert response.status_code == 206
        assert response.content == b"0123"
        assert response.headers["content-type"] == "video/mp4"
        assert response.headers["content-disposition"].startswith("inline;")


def test_delete_result_after_download(tmp_path):
    config = load_config(ROOT / "config.yaml")
    config.logging.dir = tmp_path / "logs"
    config.storage.default_output_dir = tmp_path / "outputs"
    config.storage.allowed_output_dirs = [tmp_path / "outputs"]
    config.storage.default_output_dir.mkdir()
    video = config.storage.default_output_dir / "clip.mp4"
    video.write_bytes(b"video")
    with TestClient(create_app(config, model=FakeModel())) as client:
        state = JobStatusResponse(job_id="done", status=JobStatus.COMPLETED, progress=1, created_at=utcnow())
        state.result = JobResult(
            output_path=str(video), filename=video.name, format="mp4", file_size_bytes=5,
            resolution="480p", width=832, height=480, aspect_ratio="16:9", duration_seconds=1,
            num_frames=25, fps=24, seed=1, lora=None, lora_weight=None,
            num_inference_steps=8, guidance_scale=1, multi_stage=False, audio=False,
            generation_time_seconds=1,
        )
        client.app.state.jobs.jobs[state.job_id] = Job(GenerateRequest(prompt="first"), state)
        state.status = JobStatus.RUNNING
        assert client.delete("/jobs/done/result").status_code == 409
        state.status = JobStatus.COMPLETED
        outside = tmp_path / "outside.mp4"
        outside.write_bytes(b"keep")
        state.result.output_path = str(outside)
        assert client.delete("/jobs/done/result").status_code == 409
        assert outside.read_bytes() == b"keep"
        link = config.storage.default_output_dir / "link.mp4"
        link.symlink_to(outside)
        state.result.output_path, state.result.filename = str(link), link.name
        assert client.delete("/jobs/done/result").status_code == 409
        assert link.is_symlink()
        state.result.output_path, state.result.filename = str(video), video.name
        assert client.get("/jobs/done/result").content == b"video"
        assert client.delete("/jobs/done/result").status_code == 204
        assert not video.exists()
        assert client.delete("/jobs/done/result").status_code == 204
        assert client.get("/jobs/done/result").status_code == 410
        assert client.get("/jobs/done").json()["stage"] == "result_deleted"
        assert client.get("/jobs/done").json()["result"] is None