#!/usr/bin/env bash
# Allma installer
set -e

ALLMA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ALLMA_DIR/venv"
BIN="$HOME/.local/bin"

echo ""
echo "  ██████╗     Allma installer"
echo "  ██╔══██╗    Local LLM Manager"
echo "  ███████║    "
echo "  ██╔══██║    "
echo "  ██║  ██║    "
echo "  ╚═╝  ╚═╝    "
echo ""

# ── Prerequisites ───────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python 3.10+ required (found $PY_VERSION)."
    exit 1
fi

echo "  Python $PY_VERSION — OK"

if ! command -v nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi not found. Allma requires an NVIDIA GPU with CUDA drivers."
fi

# ── Virtual environment ─────────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv "$VENV"
else
    echo "  Virtual environment exists — skipping"
fi

echo "  Installing Python dependencies..."
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r "$ALLMA_DIR/requirements.txt"
echo "  Dependencies installed — OK"

# ── .env ────────────────────────────────────────────────────────────────────────
if [ ! -f "$ALLMA_DIR/.env" ]; then
    cp "$ALLMA_DIR/.env.example" "$ALLMA_DIR/.env"
    echo "  Created .env from .env.example — review and adjust if needed"
else
    echo "  .env exists — skipping"
fi

# ── allma command ───────────────────────────────────────────────────────────────
mkdir -p "$BIN"
ALLMA_BIN="$BIN/allma"

cat > "$ALLMA_BIN" <<EOF
#!/usr/bin/env bash
exec "$VENV/bin/python" "$ALLMA_DIR/allma_cli.py" "\$@"
EOF
chmod +x "$ALLMA_BIN"
echo "  Installed allma command → $ALLMA_BIN"

# Check PATH
if [[ ":$PATH:" != *":$BIN:"* ]]; then
    echo ""
    echo "  NOTE: $BIN is not in your PATH."
    echo "  Add this to your ~/.bashrc or ~/.zshrc:"
    echo ""
    echo '    export PATH="$HOME/.local/bin:$PATH"'
    echo ""
    echo "  Then run:  source ~/.bashrc"
fi

# ── Detect backends ─────────────────────────────────────────────────────────────
echo ""
echo "  Checking backends..."

# llama.cpp
LLAMA_OK=false
LLAMA_WHERE=""
for candidate in \
    "$(command -v llama-server 2>/dev/null)" \
    "$HOME/.local/bin/llama-server" \
    "$HOME/llama.cpp/build/bin/llama-server" \
    "$HOME/AI/llama.cpp/build/bin/llama-server" \
    "/usr/local/bin/llama-server"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        LLAMA_WHERE="$candidate"
        LLAMA_OK=true
        break
    fi
done

if $LLAMA_OK; then
    LLAMA_VER=$("$LLAMA_WHERE" --version 2>&1 | head -1 | grep -oP 'version \K\S+' || echo "")
    echo "  llama-server ${LLAMA_VER:+$LLAMA_VER }— $LLAMA_WHERE"
elif "$VENV/bin/python" -c "import llama_cpp" &>/dev/null 2>&1; then
    LLAMA_PYVER=$("$VENV/bin/python" -c "import llama_cpp; print(llama_cpp.__version__)" 2>/dev/null)
    echo "  llama-cpp-python $LLAMA_PYVER — found (limited features, no vision/flash-attn)"
    LLAMA_OK=true
else
    echo "  llama-server — not found"
    echo "  llama-cpp-python — not found"
    LLAMA_OK=false
fi

# vLLM — check venv, PATH, and common conda/venv locations
VLLM_OK=false
VLLM_WHERE=""
for vllm_candidate in \
    "$VENV/bin/vllm" \
    "$(command -v vllm 2>/dev/null)"; do
    if [ -n "$vllm_candidate" ] && [ -x "$vllm_candidate" ]; then
        VLLM_WHERE="$vllm_candidate"
        VLLM_OK=true
        break
    fi
done
# Also check if importable in venv
if ! $VLLM_OK && "$VENV/bin/python" -c "import vllm" &>/dev/null 2>&1; then
    VLLM_OK=true
    VLLM_WHERE="(Python module in venv)"
fi

if $VLLM_OK; then
    VLLM_VER=$("$VENV/bin/python" -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "")
    echo "  vllm ${VLLM_VER:+$VLLM_VER }— ${VLLM_WHERE:-found}"
else
    echo "  vllm — not found"
fi

# ── Summary ─────────────────────────────────────────────────────────────────────
echo ""
echo "  ─────────────────────────────────────────"
echo "  Installation complete!"
echo ""
echo "  Next steps:"
echo ""

STEP=1
if [ "$VLLM_OK" = false ] || [ "$LLAMA_OK" = false ]; then
    echo "  $STEP. Install the backend(s) you need:"
    STEP=$((STEP+1))
    echo ""
    if [ "$VLLM_OK" = false ]; then
        echo "     ── vLLM (for safetensors / FP8 models) ──"
        echo "     $VENV/bin/pip install vllm"
        echo ""
    fi
    if [ "$LLAMA_OK" = false ]; then
        echo "     ── llama.cpp (for GGUF models) ── choose one:"
        echo ""
        echo "     Option A — automated build script (recommended):"
        echo "       bash scripts/install-llama-cpp.sh"
        echo ""
        echo "     Option B — pip (easier, limited features, no vision):"
        # Detect CUDA version for the right wheel URL
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+" | head -1)
        if [ -n "$CUDA_VER" ]; then
            echo "       $VENV/bin/pip install \"llama-cpp-python[server]\" \\"
            echo "         --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu${CUDA_VER}"
        else
            echo "       $VENV/bin/pip install \"llama-cpp-python[server]\""
        fi
        echo ""
    fi
fi

echo "  $STEP. Create your model configs:"
STEP=$((STEP+1))
echo "     cp configs/base/Qwen3.6-27B-FP8.allm.example configs/base/MyModel.allm"
echo "     # Edit the file and set the correct model path"
echo "     # Or use the interactive wizard:"
echo "     allma wizard"
echo ""
echo "  $STEP. Start:"
echo "     allma serve"
echo "     allma list"
echo "     allma run <profile-name>"
echo ""
echo "  Full docs: README.md"
echo "  ─────────────────────────────────────────"
echo ""
