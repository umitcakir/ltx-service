# LTX-2.5 Native REST Service

FastAPI wrapper around the **Diffusers** LTX-2.5 pack. No containers. One Uvicorn worker holds one copy of the model; jobs run in the background. Text-to-video, start-image video, first/last-frame video and optional joint audio are exposed through `/generate`. A separate OpenAI-compatible LAN LLM may enhance prompts before video inference.

## Requirements

- `python3` with `venv` and pip to bootstrap a project-local **Python 3.12** on Linux/macOS, Git, enough free disk for the Diffusers model pack (tens of GB). The `imageio-ffmpeg` dependency supplies the FFmpeg executable used for MP4 finalisation; a system `ffmpeg` on `PATH` is not required. An NVIDIA CUDA GPU is recommended; macOS can use `model.device: mps` and `model.precision: fp16` instead.
- The model is **gated**. Accept the license for [LTX-2.5-Diffusers](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) and authenticate with `hf auth login`. Startup downloads missing model files automatically if `model.auto_download` is true. This can take a while and requires substantial disk space. To fetch ahead of time, run `python -m scripts.download_models` after login. This excludes the unused `transformer_full/` and `diffusion_decoder/` directories.
- The `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` file you already have is a **transformer**, not a LoRA and not a complete Diffusers pipeline. The service downloads the compatible Diffusers pack (`transformer/` + Gemma-4 encoder + VAE + audio components); it cannot load that split checkpoint alone. Do not put the transformer file in `loras[].path`.
- A bf16 22B transformer plus text encoder does **not** fit wholly in 16 GB of VRAM. `cpu_offload: sequential` streams weights from RAM; tiling/slicing is enabled. Long 720p jobs may be slow and can exceed 32 GB RAM. Set `max_concurrent_jobs: 2` only after measuring peak RAM/VRAM locally. The service does not perform custom VRAM budget checks.

If a job reaches 100% denoising but reports `CUDA_OOM`, the VAE may still be decoding the video. Spatial and temporal VAE tiling are enabled when `model.vae_tiling` is true, but they cannot guarantee a long clip will fit alongside other GPU processes. Check `nvidia-smi` on the inference host before retrying, stop other GPU workloads you control, or use a shorter clip. The logged `peak_vram` is this process's PyTorch measurement, not total GPU usage.

## Setup

Windows PowerShell (Python 3.12 installed):

```powershell
.\setup.ps1
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\python.exe run.py
```

Linux or macOS (edit `model.device` to `mps` and `model.precision` to `fp16` for macOS):

```sh
sh setup.sh --start
```

The Unix setup script checks that the `app/models` source package exists, installs `uv` into `.bootstrap/`, managed Python 3.12 into `.python/`, and dependencies into `.venv/` on a fresh project. `--start` checks Hugging Face authentication, prompts for interactive login only if needed, then starts the server. Accept the gated model license in your browser first. Missing source files must be included when deploying the project; setup cannot recreate them. If `.venv/` already holds another Python version, the script leaves it intact and uses `.venv-3.12/` instead. Plain `sh setup.sh` installs dependencies without launching the server; `sh setup.sh --python-only` installs only Python 3.12 and creates the venv, without the large CUDA dependencies or the `hf` CLI. Run full setup on a capable inference host before logging in or starting the server. Windows `setup.ps1` still requires an existing Python 3.12 installation. Use the venv's Python with `run.py --config /path/to/config.yaml` to select another config. Paths in YAML are relative to its location. The supplied config binds to `0.0.0.0:8000`; clients on the LAN connect to `http://<server-lan-ip>:8000/health`, not to `0.0.0.0`. There is no API authentication: anyone who can reach the server can queue GPU jobs, inspect job metadata and download results. Bind to `127.0.0.1` in the config to restrict access to this machine. Do not expose the service to untrusted networks. For remote access, use a private VPN or an authenticating reverse proxy; setting `server.host` to `0.0.0.0` alone is not safe. No Docker runtime is involved. For persistent Linux operation, point a systemd unit's `ExecStart` at the venv's Python plus `run.py` and set `WorkingDirectory` to the project. On Windows use Task Scheduler or NSSM with the venv executable. Hugging Face login is only for downloading the gated checkpoint, not for API requests.

## Examples

```sh
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A slow dolly-in toward a rain-soaked street at dusk, warm lights reflected on the pavement","duration":5,"resolution":"720p","fps":24,"generate_audio":true,"lora":{"name":"fast"}}'

curl http://127.0.0.1:8000/jobs/JOB_ID
curl http://127.0.0.1:8000/jobs/JOB_ID/result -o clip.mp4
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/loras
```

For image conditioning, upload image bytes from a REST client and pass the returned `path` in the generation request:

```sh
curl -X POST http://127.0.0.1:8000/images -H 'Content-Type: image/png' --data-binary @newscaster_03.png
# Response: {"path":"uploaded-<id>.png"}
curl -X POST http://127.0.0.1:8000/generate -H 'Content-Type: application/json' \
  -d '{"prompt":"A presenter speaks to camera","start_image_path":"uploaded-<id>.png"}'
```

The upload endpoint accepts PNG, JPEG, WebP and BMP up to `generation.max_image_download_bytes` and stores files in `generation.input_image_dir`; uploaded images persist until removed. Alternatively put `first.jpg` in that directory yourself and send `"start_image_path": "first.jpg"`, or use `"start_image_url": "https://..."` for a publicly accessible image (private/LAN URLs are rejected). Add an `end_image_path` or `end_image_url` for first/last-frame video. `generation.allow_remote_images` controls URL fetching, not uploads. `duration: null` (or omitted) lets the LTX-2.5 duration head choose within the configured 1–20 second bounds. Only 24/25 fps and 480p (832x480) / 720p (1280x704) landscape and portrait equivalents are supported; frame count is snapped to `8k+1`. `quality` selects the two-stage distilled pass at 720p. At 480p it falls back to the single-stage distilled pass because 832x480 cannot be halved onto the model's 32-pixel grid. This is **not** the full/SFT transformer quality mode; that requires downloading and loading another large transformer.

`POST /generate` immediately returns a queued job ID. `GET /jobs/{id}` reports status, progress, error or finished file path and metadata; `GET /jobs/{id}/result` streams the MP4. `DELETE /jobs/{id}` cancels queued jobs only. Jobs and queue live in memory, so a restart loses job records (completed files remain). Inference timeouts are checked during denoising and before encoding; a stalled CUDA kernel cannot safely be killed from Python. Shutdown waits for active jobs to finish.

Generated MP4s are losslessly finalised with the index at the front for progressive browser playback. The result endpoint serves `video/mp4` inline and supports HTTP byte ranges. If another service copies files into a separate video directory, it must also serve them with the correct MIME type and byte-range support; that service's headers and caching are independent of this API.

## LAN Prompt Enhancer

To use your `ministral-3-8b-instruct-2512` server on `192.168.2.117:1234`, set `prompt_enhancer.enabled: true` in config. It expects the OpenAI-style endpoint `http://192.168.2.117:1234/v1/chat/completions`. A failed or timed-out enhancer call marks that job failed; it never downloads an LLM or loads one onto the video machine. LTX-2.5's text encoder is still required for video inference; the LAN LLM only rewrites the caption. If your server's API differs, adjust the endpoint and model ID in config or adapt `app/services/prompt_enhancer.py`.

## Presets, Files And Security

`loras` is a named preset registry: `fast` and `quality` use the Diffusers pack's distilled base transformer (no adapter). Add real adapter files under a configured local path, e.g. `./models/loras/style.safetensors`, with a new name and inference settings. Only actual **LoRA** weights can be hot-swapped; a distilled transformer cannot be unloaded like an adapter. Presets define steps, guidance and sigma schedule. Avoid mixing a distilled sigma schedule with a full/SFT transformer.

Output paths are config-only so API clients cannot choose arbitrary write locations or traverse directories. `output_filename` is a bare filename with a configured extension; the server validates it, resolves symlinks under `storage.allowed_output_dirs`, and adds dated folders automatically. Input image paths must be relative to `generation.input_image_dir`. Per-job logs are written to `logs/jobs/`; do not expose that directory publicly because prompts can be sensitive.

## Tests

Install `pytest` and run `python -m pytest -q`. Tests exercise config, requests, storage containment, unauthenticated queue endpoints and a mocked LAN enhancer without downloading weights or invoking CUDA. End-to-end inference still needs a real GPU and licensed checkpoint.