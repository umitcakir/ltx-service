from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import ConfigError, PromptEnhancerConfig, load_config
from app.main import create_app
from app.models.requests import GenerateRequest
from app.services.jobs import validate_request
from app.services.prompt_enhancer import enhance_prompt
from app.utils.paths import OutputPathError, resolve_output_path


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