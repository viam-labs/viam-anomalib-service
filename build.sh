#!/bin/sh
# Freezes the module with PyInstaller and packages it for the registry. Run ./setup.sh first.
set -e
cd "$(dirname "$0")"
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) VPY=.venv/Scripts/python.exe; EXE=dist/main.exe ;;
    *)                    VPY=.venv/bin/python;         EXE=dist/main ;;
esac
$VPY -m PyInstaller --clean --noconfirm main.spec
# meta.json's entrypoint stays "dist/main" everywhere; on Windows viam-server resolves it to dist/main.exe.
tar -czf dist/archive.tar.gz meta.json "$EXE"
