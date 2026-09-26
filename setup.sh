#!/usr/bin/env sh
set -eu
if [ "${1:-}" != "" ] && [ "$1" != "--python-only" ]; then
  printf '%s\n' 'Usage: sh setup.sh [--python-only]' >&2
  exit 2
fi
cd "$(dirname "$0")"
project_dir=$(pwd -P)
bootstrap="$project_dir/.bootstrap"
export UV_PYTHON_INSTALL_DIR="$project_dir/.python"
if [ ! -x "$bootstrap/bin/uv" ]; then
  python3 -m venv "$bootstrap"
  "$bootstrap/bin/python" -m pip install 'uv>=0.8,<1'
fi
"$bootstrap/bin/uv" python install 3.12
interpreter=$("$bootstrap/bin/uv" python find --managed-python 3.12)
venv="$project_dir/.venv"
if [ -e "$venv/pyvenv.cfg" ] && ! "$venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12)' 2>/dev/null; then
  venv="$project_dir/.venv-3.12"
fi
if [ ! -e "$venv/pyvenv.cfg" ]; then
  "$bootstrap/bin/uv" venv --python "$interpreter" "$venv"
fi
if [ "${1:-}" = "--python-only" ]; then
  printf 'Python 3.12 ready at %s/bin/python (dependencies not installed).\n' "$venv"
  exit 0
fi
"$bootstrap/bin/uv" pip install --python "$venv/bin/python" pip
if [ "$(uname -s)" = "Darwin" ]; then
  "$bootstrap/bin/uv" pip install --python "$venv/bin/python" torch==2.7.1 torchvision==0.22.1
else
  "$bootstrap/bin/uv" pip install --python "$venv/bin/python" torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
fi
"$bootstrap/bin/uv" pip install --python "$venv/bin/python" -r requirements.txt
printf 'Accept the LTX-2.5 HF license, then run %s/bin/hf auth login and %s/bin/python run.py.\n' "$venv" "$venv"