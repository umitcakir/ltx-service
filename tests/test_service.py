from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess
import sys

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from app.config import ConfigError, PromptEnhancerConfig, load_config
from app.main import create_app
from app.models.requests import GenerateRequest
from app.models.responses import JobStatus
from app.services.jobs import validate_request
from app.services.model_manager import ModelManager
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


def test_faststart_mp4(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for video encoding")
    path = tmp_path / "sample.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=s=32x32:r=1", "-t", "1", "-c:v", "mpeg4", str(path)],
        check=True,
    )
    assert path.read_bytes().index(b"mdat") < path.read_bytes().index(b"moov")
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


def test_completed_result_streams_inline(tmp_path):
    config = load_config(ROOT / "config.yaml")
    config.logging.dir = tmp_path / "logs"
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0123456789")
    with TestClient(create_app(config, model=FakeModel())) as client:
        client.app.state.jobs.get = lambda job_id: SimpleNamespace(
            status=JobStatus.COMPLETED, result=SimpleNamespace(output_path=str(video), filename=video.name)
        )
        response = client.get("/jobs/example/result", headers={"Range": "bytes=0-3"})
        assert response.status_code == 206
        assert response.content == b"0123"
        assert response.headers["content-type"] == "video/mp4"
        assert response.headers["content-disposition"].startswith("inline;")