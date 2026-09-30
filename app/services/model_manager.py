"""Base model + LoRA lifecycle for the Diffusers LTX-2.5 backend.

The base pipeline is loaded once at startup and kept alive for the process.
LoRA adapters are hot-swapped between requests without reloading the base.
Heavy imports (torch, diffusers) are deferred so the API surface and tests can
be imported on machines without CUDA.
"""

from __future__ import annotations

import inspect
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.config import AppConfig, LoraProfile
from app.utils.gpu import empty_cache, resolve_dtype

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


class ModelNotLoadedError(RuntimeError):
    pass


class ModelLoadError(RuntimeError):
    pass


@dataclass(slots=True)
class GenerationSpec:
    prompt: str
    negative_prompt: str
    width: int
    height: int
    num_frames: int | None
    fps: int
    seed: int
    guidance_scale: float
    num_inference_steps: int
    sigma_schedule: str
    multi_stage: bool
    stage2_num_inference_steps: int
    generate_audio: bool
    start_image: Any | None = None
    end_image: Any | None = None


@dataclass(slots=True)
class GenerationOutput:
    video: Any
    audio: Any | None
    audio_sample_rate: int | None
    num_frames: int
    width: int
    height: int
    extras: dict[str, Any] = field(default_factory=dict)


class ModelManager:
    """Owns the LTX-2.5 pipeline and the currently-applied LoRA adapter."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._pipeline: Any | None = None
        self._t2v_pipeline: Any | None = None
        self._condition_pipeline: Any | None = None
        self._upsample_pipeline: Any | None = None
        self._sigmas: dict[str, Any] = {}
        self._loaded_adapter: str | None = None
        self._loaded_adapter_weight: float | None = None
        # Serialises adapter swaps and the denoising call. ``max_concurrent_jobs``
        # still overlaps image fetching, VAE decode-to-disk and encoding.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ state

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def loaded_lora(self) -> str | None:
        return self._loaded_adapter

    # ----------------------------------------------------------------- loading

    def load(self) -> None:
        """Blocking load of the base pipeline. Call from a worker thread."""
        if self._config.model.dry_run:
            logger.warning("model.dry_run is enabled — no weights will be loaded")
            return
        if self._pipeline is not None:
            return

        model_cfg = self._config.model
        checkpoint = self._ensure_checkpoint()

        import torch  # noqa: PLC0415

        try:
            from diffusers import LTX2Pipeline  # noqa: PLC0415
        except ImportError as exc:
            raise ModelLoadError(
                "diffusers with LTX-2.5 support is required. Install it with:\n"
                "  pip install 'git+https://github.com/huggingface/diffusers'"
            ) from exc

        dtype = resolve_dtype(model_cfg.precision)
        logger.info(
            "Loading LTX-2.5 from %s (dtype=%s, quantization=%s)",
            checkpoint,
            model_cfg.precision,
            model_cfg.quantization,
        )

        quant_config = self._build_quantization_config()
        from_pretrained_kwargs: dict[str, Any] = {"dtype": dtype}
        if quant_config is not None:
            from_pretrained_kwargs["quantization_config"] = quant_config

        pipeline = LTX2Pipeline.from_pretrained(str(checkpoint), **from_pretrained_kwargs)

        if model_cfg.cpu_offload == "model":
            pipeline.enable_model_cpu_offload()
        elif model_cfg.cpu_offload == "sequential":
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline.to(model_cfg.device)

        if model_cfg.vae_tiling and hasattr(pipeline, "vae"):
            pipeline.vae.enable_tiling()
            if hasattr(pipeline.vae, "use_framewise_decoding"):
                pipeline.vae.use_framewise_decoding = True
        if model_cfg.vae_slicing and hasattr(pipeline.vae, "enable_slicing"):
            pipeline.vae.enable_slicing()

        self._pipeline = pipeline
        self._t2v_pipeline = pipeline
        self._sigmas = self._load_sigma_schedules()

        if model_cfg.enable_latent_upsampler:
            self._upsample_pipeline = self._build_upsampler(checkpoint, pipeline, dtype)
            if self._upsample_pipeline is None:
                raise ModelLoadError("Configured latent upsampler could not be loaded")

        logger.info(
            "LTX-2.5 ready (device=%s, offload=%s, quantization=%s, upsampler=%s)",
            model_cfg.device,
            model_cfg.cpu_offload,
            model_cfg.quantization,
            self._upsample_pipeline is not None,
        )
        del torch

    def _build_quantization_config(self) -> Any | None:
        model_cfg = self._config.model
        if model_cfg.quantization == "none":
            return None

        try:
            from diffusers import PipelineQuantizationConfig  # noqa: PLC0415
        except ImportError as exc:
            raise ModelLoadError(
                "model.quantization requires a diffusers build exposing "
                "PipelineQuantizationConfig. Reinstall diffusers from main."
            ) from exc

        try:
            import torchao  # noqa: F401, PLC0415
        except ImportError as exc:
            raise ModelLoadError(
                "model.quantization='int8' requires torchao. Install it with:\n"
                "  pip install torchao"
            ) from exc

        return PipelineQuantizationConfig(
            quant_backend="torchao",
            quant_kwargs={"quant_type": "int8_weight_only"},
            components_to_quantize=list(model_cfg.quantize_components),
        )

    def _ensure_checkpoint(self) -> Path:
        model_cfg = self._config.model
        checkpoint = model_cfg.checkpoint_path
        if (checkpoint / "model_index.json").is_file():
            return checkpoint
        if not model_cfg.auto_download:
            raise ModelLoadError(
                f"No Diffusers pack at {checkpoint} and model.auto_download is false. "
                f"Run: python scripts/download_models.py"
            )

        logger.info("Checkpoint missing — downloading %s", model_cfg.repo_id)
        from scripts.download_models import download_base_model  # noqa: PLC0415

        return download_base_model(model_cfg.repo_id, checkpoint)

    def _build_upsampler(self, checkpoint: Path, pipeline: Any, dtype: Any) -> Any | None:
        try:
            from diffusers import LTX2LatentUpsamplePipeline  # noqa: PLC0415
            from diffusers.pipelines.ltx2.latent_upsampler import (  # noqa: PLC0415
                LTX2LatentUpsamplerModel,
            )
        except ImportError:
            logger.warning("Latent upsampler classes unavailable; multi_stage presets disabled")
            return None

        try:
            upsampler = LTX2LatentUpsamplerModel.from_pretrained(
                str(checkpoint),
                subfolder=self._config.model.latent_upsampler_subfolder,
                dtype=dtype,
            )
            upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipeline.vae, latent_upsampler=upsampler)
            if self._config.model.cpu_offload == "none":
                upsample_pipe.to(self._config.model.device)
            else:
                upsample_pipe.enable_model_cpu_offload(device=self._config.model.device)
            return upsample_pipe
        except Exception as exc:
            logger.warning("Could not load the latent upsampler: %s", exc)
            return None

    @staticmethod
    def _load_sigma_schedules() -> dict[str, Any]:
        try:
            from diffusers.pipelines.ltx2.utils import (  # noqa: PLC0415
                DISTILLED_SIGMA_VALUES,
                STAGE_2_DISTILLED_SIGMA_VALUES,
            )
        except ImportError:
            logger.warning("LTX-2.5 sigma constants unavailable; presets will use step counts")
            return {}
        return {
            "distilled": DISTILLED_SIGMA_VALUES,
            "distilled_stage2": STAGE_2_DISTILLED_SIGMA_VALUES,
        }

    def unload(self) -> None:
        with self._lock:
            self._unload_adapter_locked()
            self._pipeline = None
            self._t2v_pipeline = None
            self._condition_pipeline = None
            self._upsample_pipeline = None
            empty_cache()

    # -------------------------------------------------------------------- LoRA

    def apply_profile(self, profile: LoraProfile, weight: float) -> None:
        """Hot-swap to ``profile``'s adapter; base-only profiles unload any adapter."""
        if self._config.model.dry_run:
            self._loaded_adapter = profile.name if profile.path else None
            self._loaded_adapter_weight = weight if profile.path else None
            return

        with self._lock:
            if profile.path is None:
                self._unload_adapter_locked()
                return
            if self._loaded_adapter == profile.name and self._loaded_adapter_weight == weight:
                return

            self._unload_adapter_locked()
            pipeline = self._require_pipeline()
            logger.info("Loading LoRA '%s' (weight=%.3f)", profile.name, weight)
            pipeline.load_lora_weights(str(profile.path), adapter_name=profile.name)
            pipeline.set_adapters([profile.name], adapter_weights=[weight])
            self._loaded_adapter = profile.name
            self._loaded_adapter_weight = weight

    def _unload_adapter_locked(self) -> None:
        if self._loaded_adapter is None or self._pipeline is None:
            self._loaded_adapter = None
            self._loaded_adapter_weight = None
            return
        try:
            self._pipeline.unload_lora_weights()
        except Exception as exc:  # pragma: no cover - backend specific
            logger.warning("Failed to unload LoRA '%s': %s", self._loaded_adapter, exc)
        finally:
            self._loaded_adapter = None
            self._loaded_adapter_weight = None

    # -------------------------------------------------------------- generation

    def _require_pipeline(self) -> Any:
        if self._pipeline is None:
            raise ModelNotLoadedError("LTX-2.5 pipeline is not loaded")
        return self._pipeline

    def generate(
        self, spec: GenerationSpec, profile: LoraProfile, weight: float,
        progress: ProgressCallback | None = None
    ) -> GenerationOutput:
        """Run inference. Blocking — call from a worker thread."""
        if self._config.model.dry_run:
            raise ModelNotLoadedError(
                "model.dry_run is enabled; generation is disabled in this mode"
            )

        import torch  # noqa: PLC0415

        with self._lock:
            self.apply_profile(profile, weight)
            pipeline = self._require_pipeline()
            if spec.start_image is not None:
                if self._condition_pipeline is None:
                    from diffusers import LTX2ConditionPipeline
                    self._condition_pipeline = LTX2ConditionPipeline(**pipeline.components)
                pipeline = self._condition_pipeline

            device = self._config.model.device
            generator = torch.Generator(device="cpu" if device == "mps" else device)
            generator.manual_seed(spec.seed)

            shared = self._build_kwargs(pipeline, spec, generator, progress)

            if spec.multi_stage and self._upsample_pipeline is not None:
                return self._run_multi_stage(pipeline, spec, shared)
            if spec.multi_stage:
                raise ModelLoadError("Multi-stage preset requires the latent upsampler")
            return self._run_single_stage(pipeline, spec, shared)

    def _build_kwargs(
        self,
        pipeline: Any,
        spec: GenerationSpec,
        generator: Any,
        progress: ProgressCallback | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "prompt": spec.prompt,
            "negative_prompt": spec.negative_prompt,
            "frame_rate": float(spec.fps),
            "guidance_scale": spec.guidance_scale,
            "generator": generator,
            "return_dict": False,
        }

        if spec.start_image is not None:
            from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
            conditions = [LTX2VideoCondition(frames=spec.start_image, index=0, strength=1.0)]
            if spec.end_image is not None:
                conditions.append(LTX2VideoCondition(frames=spec.end_image, index=-1, strength=1.0))
            kwargs["conditions"] = conditions
        kwargs.update(audio_guidance_scale=spec.guidance_scale, stg_scale=0.0,
                      audio_stg_scale=0.0, modality_scale=1.0, audio_modality_scale=1.0,
                      min_seconds=1.0, max_seconds=float(self._config.generation.max_duration_seconds))

        if progress is not None:
            total = spec.num_inference_steps

            def _callback(_pipe: Any, step: int, _timestep: Any, callback_kwargs: dict) -> dict:
                progress(step + 1, total, "denoise")
                return callback_kwargs

            kwargs["callback_on_step_end"] = _callback

        return self._filter_supported(pipeline, kwargs)

    @staticmethod
    def _filter_supported(pipeline: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop kwargs this diffusers build's ``__call__`` does not accept."""
        try:
            signature = inspect.signature(pipeline.__call__)
        except (TypeError, ValueError):  # pragma: no cover
            return kwargs
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            return kwargs
        accepted = set(signature.parameters)
        dropped = sorted(set(kwargs) - accepted)
        if dropped:
            logger.debug("Dropping unsupported pipeline kwargs: %s", dropped)
        return {key: value for key, value in kwargs.items() if key in accepted}

    def _sigmas_for(self, name: str) -> Any | None:
        if name == "none":
            return None
        return self._sigmas.get(name)

    def _run_single_stage(
        self, pipeline: Any, spec: GenerationSpec, shared: dict[str, Any]
    ) -> GenerationOutput:
        call_kwargs = dict(shared)
        call_kwargs.update(
            {
                "height": spec.height,
                "width": spec.width,
                "output_type": "np",
            }
        )
        if spec.num_frames is not None:
            call_kwargs["num_frames"] = spec.num_frames

        sigmas = self._sigmas_for(spec.sigma_schedule)
        if spec.sigma_schedule != "none" and sigmas is None:
            raise ModelLoadError("Distilled sigma schedule unavailable in installed diffusers")
        if sigmas is not None:
            call_kwargs["sigmas"] = sigmas
        else:
            call_kwargs["num_inference_steps"] = spec.num_inference_steps

        result = pipeline(**self._filter_supported(pipeline, call_kwargs))
        return self._to_output(result, spec)

    def _run_multi_stage(
        self, pipeline: Any, spec: GenerationSpec, shared: dict[str, Any]
    ) -> GenerationOutput:
        """Stage 1 at half resolution, x2 latent upsample, then a short refine pass."""
        if spec.width % 64 or spec.height % 64:
            raise ValueError("Two-stage generation requires width and height divisible by 64")
        stage1_width = spec.width // 2
        stage1_height = spec.height // 2

        stage1_kwargs = dict(shared)
        stage1_kwargs.update(
            {
                "height": stage1_height,
                "width": stage1_width,
                "output_type": "latent",
            }
        )
        if spec.num_frames is not None:
            stage1_kwargs["num_frames"] = spec.num_frames

        sigmas = self._sigmas_for(spec.sigma_schedule)
        if spec.sigma_schedule != "none" and sigmas is None:
            raise ModelLoadError("Distilled sigma schedule unavailable in installed diffusers")
        if sigmas is not None:
            stage1_kwargs["sigmas"] = sigmas
        else:
            stage1_kwargs["num_inference_steps"] = spec.num_inference_steps

        stage1 = pipeline(**self._filter_supported(pipeline, stage1_kwargs))
        video_latents, audio_latents = self._split_result(stage1)

        upsampled = self._upsample_pipeline(
            latents=video_latents, latents_normalized=False, output_type="latent", return_dict=False
        )[0]

        stage2_kwargs = dict(shared)
        stage2_kwargs.update(
            {
                "latents": upsampled,
                "audio_latents": audio_latents,
                "width": spec.width,
                "height": spec.height,
                "output_type": "np",
            }
        )
        stage2_kwargs["num_frames"] = spec.num_frames or (
            (video_latents.shape[2] - 1) * pipeline.vae_temporal_compression_ratio + 1
        )

        stage2_sigmas = self._sigmas.get("distilled_stage2")
        if stage2_sigmas is not None:
            stage2_kwargs["sigmas"] = stage2_sigmas
            stage2_kwargs["noise_scale"] = stage2_sigmas[0]
        else:
            stage2_kwargs["num_inference_steps"] = spec.stage2_num_inference_steps

        result = pipeline(**self._filter_supported(pipeline, stage2_kwargs))
        return self._to_output(result, spec)

    @staticmethod
    def _split_result(result: Any) -> tuple[Any, Any | None]:
        if isinstance(result, tuple):
            if len(result) >= 2:
                return result[0], result[1]
            return result[0], None
        frames = getattr(result, "frames", result)
        return frames, getattr(result, "audio", None)

    def _to_output(self, result: Any, spec: GenerationSpec) -> GenerationOutput:
        video, audio = self._split_result(result)
        frames = video[0] if len(video) else video

        sample_rate: int | None = None
        if audio is not None and self._pipeline is not None:
            vocoder = getattr(self._pipeline, "vocoder", None)
            sample_rate = getattr(getattr(vocoder, "config", None), "output_sampling_rate", None)

        audio_track = audio[0] if (audio is not None and spec.generate_audio) else None
        num_frames = int(frames.shape[0]) if hasattr(frames, "shape") else len(frames)

        return GenerationOutput(
            video=frames,
            audio=audio_track,
            audio_sample_rate=sample_rate,
            num_frames=num_frames,
            width=spec.width,
            height=spec.height,
        )
