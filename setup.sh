#!/bin/sh
# Creates .venv with runtime + build deps. Used by the Viam cloud build (Linux) and the Windows
# GitHub Actions job (Git Bash). Override the interpreter with PYTHON=/path/to/python3.x.
set -e
cd "$(dirname "$0")"
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) PY="${PYTHON:-python}";  VPY=.venv/Scripts/python.exe ;;
    *)                    PY="${PYTHON:-python3}"; VPY=.venv/bin/python ;;
esac
$PY -m venv .venv
$VPY -m pip install -q -r requirements.txt "pyinstaller>=6.0"
