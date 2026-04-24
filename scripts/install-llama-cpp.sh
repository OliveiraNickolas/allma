#!/usr/bin/env bash
# Build llama-server from source with CUDA support.
# Run from anywhere — installs llama-server to ~/.local/bin/
set -e

INSTALL_DIR="$HOME/.local/bin"
BUILD_DIR="$HOME/.local/share/llama.cpp"

echo ""
echo "  llama.cpp builder"
echo "  Builds llama-server with CUDA GPU support"
echo ""

# ── Check prerequisites ─────────────────────────────────────────────────────────
missing=()
for cmd in git cmake g++ make; do
    command -v "$cmd" &>/dev/null || missing+=("$cmd")
done

if [ ${#missing[@]} -gt 0 ]; then
    echo "  ERROR: Missing required tools: ${missing[*]}"
    echo ""
    echo "  Install them with:"
    echo "    sudo apt install -y git cmake build-essential"
    exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
    echo "  WARNING: NVIDIA GPU not detected. Building CPU-only version."
    CUDA_FLAG=""
else
    CUDA_FLAG="-DGGML_CUDA=ON"
    echo "  NVIDIA GPU detected — building with CUDA support"
fi

# ── Clone or update ─────────────────────────────────────────────────────────────
if [ -d "$BUILD_DIR/.git" ]; then
    echo "  Updating existing llama.cpp source..."
    git -C "$BUILD_DIR" pull --ff-only 2>&1 | tail -1
else
    echo "  Cloning llama.cpp..."
    git clone --depth=1 https://github.com/ggerganov/llama.cpp "$BUILD_DIR"
fi

# ── Build ────────────────────────────────────────────────────────────────────────
echo "  Configuring (cmake)..."
cmake -B "$BUILD_DIR/build" "$BUILD_DIR" $CUDA_FLAG -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON 2>&1 | tail -3

JOBS=$(nproc 2>/dev/null || echo 4)
echo "  Building with $JOBS jobs (this takes 2-5 minutes)..."
cmake --build "$BUILD_DIR/build" --config Release -j"$JOBS" --target llama-server 2>&1 | tail -5

# ── Install ──────────────────────────────────────────────────────────────────────
mkdir -p "$INSTALL_DIR"
cp "$BUILD_DIR/build/bin/llama-server" "$INSTALL_DIR/llama-server"
chmod +x "$INSTALL_DIR/llama-server"

echo ""
echo "  llama-server installed → $INSTALL_DIR/llama-server"
echo "  Version: $("$INSTALL_DIR/llama-server" --version 2>&1 | head -1)"
echo ""

if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo "  NOTE: Add $INSTALL_DIR to your PATH if it isn't already:"
    echo '    export PATH="$HOME/.local/bin:$PATH"'
    echo ""
fi

echo "  Done. Run 'allma serve' to start."
echo ""
