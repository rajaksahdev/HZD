#!/usr/bin/env bash
# HazardLens: one-command setup. Run from the project root (Git Bash or WSL2 on Windows).
set -euo pipefail
cd "$(dirname "$0")"

echo "1/7 Creating folders"
mkdir -p pipelines models api assistant web tests docs scripts data/raw data/processed secrets

echo "2/7 Creating .env (edit it afterwards)"
[ -f .env ] || cp .env.example .env

echo "3/7 Python environment (3.11)"
if command -v uv >/dev/null 2>&1; then UV="uv"; else
  python -m uv --version >/dev/null 2>&1 || python -m pip install --quiet uv
  UV="python -m uv"
fi
[ -d .venv ] || $UV venv --python 3.11
# Install before activating: "python -m uv" must run from the system Python, not the venv.
if [ -f requirements.lock ]; then REQS=requirements.lock; else REQS=requirements.txt; fi
$UV pip install --python .venv -r "$REQS"
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; else source .venv/Scripts/activate; fi

echo "4/7 Code-style hooks"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || git init -q
pre-commit install

echo "5/7 Database and cache"
docker compose up -d --wait db redis

echo "6/7 Tables and ward boundaries"
python -m scripts.load_wards

echo "7/7 Map (needs Node.js)"
if command -v npm >/dev/null 2>&1; then
  (cd web && npm ci --no-audit --no-fund && npm run build)
else
  echo "npm not found: install Node.js 20+ and run: cd web && npm ci && npm run build"
fi

echo "Done. Next: activate .venv, then run: python -m scripts.demo --serve   (and pytest)"
