#!/usr/bin/env sh
set -eu
if [ "$#" -gt 1 ] || { [ "${1:-}" != "" ] && [ "$1" != "--python-only" ] && [ "$1" != "--start" ]; }; then
  printf '%s\n' 'Usage: sh setup.sh [--python-only|--start]' >&2
  exit 2
fi
cd "$(dirname "$0")"
project_dir=$(pwd -P)
for source in app/models/__init__.py app/models/requests.py app/models/responses.py; do
  if [ ! -f "$source" ]; then
    printf 'Missing %s. Deploy the complete app/models source package before running setup.\n' "$source" >&2
    exit 1
  fi
done
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
if [ "${1:-}" = "--start" ]; then
  if ! "$venv/bin/hf" auth whoami >/dev/null 2>&1; then
    printf '%s\n' 'Accept the LTX-2.5 Hugging Face license in your browser, then complete the login prompt.'
    "$venv/bin/hf" auth login
  fi
  exec "$project_dir/start.sh"
fi
printf 'Setup complete. Run sh setup.sh --start to log in if necessary and launch the service (or %s/bin/python run.py to launch directly).\n' "$venv"