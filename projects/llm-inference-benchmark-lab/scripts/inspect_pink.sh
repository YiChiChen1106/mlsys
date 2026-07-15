#!/usr/bin/env bash
set -euo pipefail

echo "## Host"
hostname

echo "## GPU"
nvidia-smi

echo "## Docker"
if command -v docker >/dev/null 2>&1; then
  docker --version
  docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
else
  echo "docker: not found"
fi

echo "## Python"
if command -v python >/dev/null 2>&1; then
  python --version
else
  echo "python: not found"
fi

echo "## Disk"
df -h .
