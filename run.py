"""Native entrypoint: python run.py [--config path/to/config.yaml]."""

import argparse
import os
import sys


def check_runtime() -> None:
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Python 3.12 is required for inference. Run sh setup.sh to install a project-local Python, then use the venv path printed by setup (not this interpreter).")
    try:
        import huggingface_hub
        import torch
        import diffusers
    except ImportError as exc:
        raise SystemExit(f"Missing inference dependency ({exc.name}). Run sh setup.sh to install the complete environment before starting the server.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve local LTX-2.5 generation via FastAPI")
    parser.add_argument("--config", default="config.yaml", help="YAML config file")
    args = parser.parse_args()
    check_runtime()
    import uvicorn
    from app.config import load_config

    config = load_config(args.config)
    os.environ["LTX_SERVICE_CONFIG"] = str(config.source_path)
    uvicorn.run(
        "app.main:create_app", host=config.server.host, port=config.server.port,
        workers=config.server.workers, factory=True,
        log_level=config.server.log_level,
        timeout_graceful_shutdown=int(config.server.shutdown_grace_seconds),
    )
    from app.services import jobs

    if jobs.restart_requested:
        sys.exit(jobs.RESTART_EXIT_CODE)


if __name__ == "__main__":
    main()