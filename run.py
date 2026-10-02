"""Native entrypoint: python run.py [--config path/to/config.yaml]."""

import argparse
import os
import signal
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

    from app.services import jobs

    config = load_config(args.config)
    os.environ["LTX_SERVICE_CONFIG"] = str(config.source_path)
    install_restart_handler(jobs)
    uvicorn.run(
        "app.main:create_app", host=config.server.host, port=config.server.port,
        workers=config.server.workers, factory=True,
        log_level=config.server.log_level,
        timeout_graceful_shutdown=int(config.server.shutdown_grace_seconds),
    )
    if jobs.restart_requested:
        sys.exit(jobs.RESTART_EXIT_CODE)


def install_restart_handler(jobs) -> None:
    """Map uvicorn's post-shutdown SIGTERM re-raise to the restart exit code.

    uvicorn restores this handler after a graceful shutdown and re-raises the
    captured signal; without it the process dies by SIGTERM (status 143) and the
    start scripts never see exit code 75.
    """

    def handle_sigterm(signum, frame) -> None:
        if jobs.restart_requested:
            sys.exit(jobs.RESTART_EXIT_CODE)
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)

    signal.signal(signal.SIGTERM, handle_sigterm)


if __name__ == "__main__":
    main()