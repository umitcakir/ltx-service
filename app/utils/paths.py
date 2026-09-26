"""Filename sanitisation and output-path containment.

Clients never supply a directory. They may supply a bare filename, which is
sanitised and then resolved strictly inside a configured whitelist directory.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path

from app.config import StorageConfig

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_COLLAPSE = re.compile(r"[-_.]{2,}")
_RESERVED_WINDOWS_NAMES = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


class OutputPathError(ValueError):
    """Raised when a requested output location is not permitted."""


def sanitize_filename(filename: str, storage: StorageConfig) -> str:
    """Reduce a client string to a safe ``stem.ext`` inside the allowed formats."""
    candidate = filename.strip().replace("\\", "/")
    # Discard any directory component the client tried to sneak in.
    candidate = candidate.rsplit("/", 1)[-1]

    stem, _, suffix = candidate.rpartition(".")
    if not stem or not suffix:
        raise OutputPathError("output_filename must be of the form 'name.ext'")

    suffix = _UNSAFE_CHARS.sub("", suffix).lower()
    if suffix not in storage.allowed_formats:
        raise OutputPathError(
            f"output format '{suffix}' is not allowed; permitted: {storage.allowed_formats}"
        )

    stem = _UNSAFE_CHARS.sub("_", stem)
    stem = _COLLAPSE.sub("_", stem).strip("._-")
    if not stem:
        raise OutputPathError("output_filename has no usable characters")
    if stem.lower() in _RESERVED_WINDOWS_NAMES:
        stem = f"{stem}_file"
    stem = stem[: storage.max_filename_length]

    return f"{stem}.{suffix}"


def build_filename(
    storage: StorageConfig,
    *,
    resolution: str,
    job_id: str,
    now: datetime | None = None,
) -> str:
    """Render ``storage.filename_pattern`` for an auto-named output."""
    moment = now or datetime.now()
    rendered = storage.filename_pattern.format(
        date=moment.strftime("%Y%m%d"),
        time=moment.strftime("%H%M%S"),
        datetime=moment.strftime("%Y%m%d-%H%M%S"),
        uuid=job_id or uuid.uuid4().hex,
        resolution=resolution,
    )
    return sanitize_filename(rendered, storage)


def _containing_root(path: Path, roots: list[Path]) -> Path | None:
    for root in roots:
        try:
            resolved_root = root.resolve()
        except OSError:  # pragma: no cover - unreadable root
            continue
        if path == resolved_root or resolved_root in path.parents:
            return resolved_root
    return None


def resolve_output_path(
    storage: StorageConfig,
    *,
    filename: str,
    now: datetime | None = None,
    create_parents: bool = True,
) -> Path:
    """Return an absolute output path guaranteed to sit inside the whitelist."""
    safe_name = sanitize_filename(filename, storage)

    target_dir = storage.default_output_dir
    if storage.dated_subfolders:
        moment = now or datetime.now()
        target_dir = target_dir / moment.strftime(storage.dated_subfolder_pattern)

    if create_parents:
        target_dir.mkdir(parents=True, exist_ok=True)

    # Resolve after mkdir so symlinked roots normalise consistently.
    resolved_dir = target_dir.resolve() if target_dir.exists() else target_dir.absolute()
    candidate = (resolved_dir / safe_name).absolute()

    if _containing_root(candidate, storage.allowed_output_dirs) is None:
        raise OutputPathError(
            "resolved output path escapes the configured allowed_output_dirs"
        )
    if candidate.is_symlink():
        raise OutputPathError("output filename cannot be a symlink")
    if candidate.is_dir():
        raise OutputPathError(f"output path '{safe_name}' is an existing directory")

    return candidate


def resolve_input_image(relative_path: str, input_dir: Path) -> Path:
    """Resolve a client-supplied relative image path inside ``input_dir``."""
    cleaned = relative_path.strip().replace("\\", "/").lstrip("/")
    if not cleaned or ".." in cleaned.split("/"):
        raise OutputPathError("invalid image path")

    root = input_dir.resolve() if input_dir.exists() else input_dir.absolute()
    candidate = (root / cleaned).absolute()
    try:
        candidate = candidate.resolve(strict=True)
    except (OSError, FileNotFoundError) as exc:
        raise OutputPathError(f"image not found: {cleaned}") from exc

    if candidate != root and root not in candidate.parents:
        raise OutputPathError("image path escapes the configured input directory")
    if not candidate.is_file():
        raise OutputPathError(f"image is not a regular file: {cleaned}")
    return candidate
