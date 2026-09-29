"""FastAPI lifespan; model load happens exactly once in the single server process."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.config import AppConfig, get_config
from app.routes.api import router
from app.services.jobs import JobManager
from app.services.model_manager import ModelManager
from app.utils.logging_setup import configure_logging


def create_app(config: AppConfig | None = None, *, model: ModelManager | None = None) -> FastAPI:
    settings = config or get_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.logging)
        instance = model or ModelManager(settings)
        await asyncio.to_thread(instance.load)
        jobs = JobManager(settings, instance)
        app.state.config = settings
        app.state.model = instance
        app.state.jobs = jobs
        await jobs.start()
        try:
            yield
        finally:
            await jobs.stop()
            # A poisoned CUDA context fails every call; the process exit frees the GPU.
            if not jobs.cuda_context_lost:
                await asyncio.to_thread(instance.unload)

    app = FastAPI(title="LTX-2.5 Local API", version=__version__, lifespan=lifespan)
    app.include_router(router)
    return app