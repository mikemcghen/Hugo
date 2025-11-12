#!/bin/bash
# Hugo Quick Start Script for Unix/Mac
# =====================================

set -e

echo ""
echo "================================================"
echo "Hugo Local AI Assistant - Quick Start"
echo "================================================"
echo ""

# Check if Ollama is running
echo "[1/3] Checking Ollama connection..."
if ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo "[!] Ollama is not running!"
    echo ""
    echo "Please start Ollama first:"
    echo "  1. Open a new terminal"
    echo "  2. Run: ollama serve"
    echo "  3. Then run this script again"
    echo ""
    exit 1
fi
echo "[OK] Ollama is running"

# Check if model is available
echo ""
echo "[2/3] Checking for llama3:8b model..."
if ! ollama list | grep -q "llama3:8b"; then
    echo "[!] Model not found. Pulling llama3:8b..."
    ollama pull llama3:8b
fi
echo "[OK] Model available"

# Start Hugo
echo ""
echo "[3/3] Starting Hugo interactive shell..."
echo ""
echo "================================================"
echo ""

python -m runtime.cli shell

echo ""
echo "Hugo session ended."
