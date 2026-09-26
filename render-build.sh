#!/usr/bin/env bash
# Render Build Script for MAUSAM Backend
# Exit immediately if a command exits with a non-zero status
set -o errexit

echo "=========================================="
echo "  MAUSAM BACKEND: RENDER BUILD START"
echo "=========================================="

echo "--> [1/3] Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "--> [2/3] Installing lightweight CPU-only PyTorch (prevents 2.5GB CUDA memory crash)..."
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "--> [3/3] Installing remaining dependencies from requirements-full.txt..."
python -m pip install -r requirements-full.txt

echo "=========================================="
echo "  MAUSAM BACKEND: RENDER BUILD COMPLETE!"
echo "=========================================="
