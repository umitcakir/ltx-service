#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
project_dir=$(pwd -P)

venv="$project_dir/.venv"
if [ ! -e "$venv/pyvenv.cfg" ] || ! "$venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12)' 2>/dev/null; then
  venv="$project_dir/.venv-3.12"
fi
if [ ! -x "$venv/bin/python" ]; then
  printf 'No Python 3.12 environment found. Run sh setup.sh first.\n' >&2
  exit 1
fi

if [ -x "$venv/bin/hf" ] && ! "$venv/bin/hf" auth whoami >/dev/null 2>&1; then
  printf 'Not logged in to Hugging Face; run %s/bin/hf auth login if model weights still need downloading.\n' "$venv" >&2
fi

exec "$venv/bin/python" run.py "$@"
