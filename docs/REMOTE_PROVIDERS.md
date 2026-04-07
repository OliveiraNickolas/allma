# 🌐 Using Remote Providers with Allama

Allama now supports running models from remote providers (OpenCode, OpenClaw) directly without needing to download and configure them locally. Perfect for testing models or using proprietary APIs.

## Supported Providers

| Provider | Docs | Models |
|----------|------|--------|
| **OpenCode** | https://opencode.ai/docs | 75+ LLM providers (OpenAI, Anthropic, Google, etc.) |
| **OpenClaw** | https://docs.openclaw.ai | Multi-provider orchestrator |
| More coming... | | |

## Quick Start

### 1. Set Up Credentials

Create `~/.allama/.env` with your API keys:

```env
# For OpenCode (https://opencode.ai)
OPENCODE_API_KEY=sk-...
OPENCODE_BASE_URL=https://api.opencode.io/v1

# For OpenClaw (https://docs.openclaw.ai)
OPENCLAW_API_KEY=...
OPENCLAW_BASE_URL=https://api.openclaw.ai/v1
```

**Never commit `.env` to git!** It's in `.gitignore`.

### 2. Run a Model

Use the new syntax: `allama run <model> on <provider>`

```bash
# Run OpenAI's GPT-4 through OpenCode
allama run gpt-4 on opencode

# Run Claude Sonnet through OpenClaw
allama run claude-sonnet-4 on openclaw

# Save model for future use (no "on" needed next time)
allama run gpt-4 on opencode --persist
```

### 3. Chat

Once the REPL opens, chat normally:

```
>>> Hello! What's 2+2?
(streaming response from remote API...)
4
>>> /bye
Bye!
```

## How It Works

1. **Validation:** Allama checks if the model exists on the provider
2. **Metadata:** Fetches context window, max tokens, and other specs
3. **Caching:** Optionally saves the model config for future use (`--persist`)
4. **Routing:** Routes all `/v1/chat/completions` requests to the provider's API
5. **Auth:** Automatically adds your API key to requests

## Managing Cached Models

Allama caches remote models in `~/.allama/dynamic_models.json`:

```json
{
  "opencode/gpt-4": {
    "backend": "opencode",
    "model_id": "gpt-4",
    "context_window": 8192,
    "max_tokens": 4096
  }
}
```

**To update the cache:**
- Delete `~/.allama/dynamic_models.json` and Allama will re-fetch on next use
- Or manually edit the file (advanced users only)

## Examples

### Example 1: Compare Models Across Providers

```bash
# Test GPT-4 from OpenAI (via OpenCode)
allama run gpt-4 on opencode

# Test Claude Sonnet (via OpenClaw)
allama run claude-sonnet-4 on openclaw

# Test Gemini (via OpenCode)
allama run gemini-2.0-flash on opencode
```

### Example 2: Use with CI/CD

```bash
# Batch test (non-interactive)
export OPENCODE_API_KEY=sk-...
echo "What is AI?" | allama run gpt-4 on opencode < /dev/stdin

# Or via API:
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "opencode/gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

### Example 3: Persistent Config

```bash
# First time: slower (fetches metadata, saves config)
allama run gpt-4 on opencode --persist

# Next time: instant (uses cached config)
allama run gpt-4 on opencode
```

## Troubleshooting

### "Model not found" Error

```bash
❌ Model 'gpt-4-unknown' not found on opencode
```

**Solution:** Check the provider's docs for the exact model ID.
- OpenCode models: https://opencode.ai/docs/models
- OpenClaw models: https://docs.openclaw.ai/cli/models

---

### "OPENCODE_API_KEY not set" Error

```bash
⚠️  OPENCODE_API_KEY not set; skipping model validation
```

**Solution:** Create `~/.allama/.env`:

```bash
mkdir -p ~/.allama
cat > ~/.allama/.env << EOF
OPENCODE_API_KEY=sk-...
EOF
chmod 600 ~/.allama/.env  # Secure the file
```

---

### Network Timeout

```bash
❌ Connection failed: timed out
```

**Solutions:**
- Check your internet connection
- Verify the provider is up: `curl https://api.opencode.io/v1/models`
- Try `allama serve --verbose` for more logs

---

## Comparison: Local vs Remote Models

| Feature | Local (vLLM/llama.cpp) | Remote (OpenCode/OpenClaw) |
|---------|----------------------|---------------------------|
| Speed | Very fast (GPU) | Medium (network) |
| Cost | GPU hardware | Pay per API call |
| Privacy | Local only | Sent to provider |
| Models | Limited to downloaded | Access to 100+ models |
| Setup | Complex | Just add API key |
| Latency | <100ms | 500ms-2s |
| Concurrency | Limited by VRAM | Provider limits |

**Use remote for:** Testing, proprietary models, low hardware, API-first workflows.
**Use local for:** Privacy, speed, unlimited requests, cost control.

## Advanced: Custom Provider Setup

Currently, Allama supports OpenCode and OpenClaw directly. To add support for another OpenAI-compatible API:

**File:** `core/provider_resolver.py`

```python
class MyCustomProviderResolver(ProviderResolver):
    def __init__(self):
        super().__init__("myprovider")
        self.base_url = "https://api.myprovider.com/v1"
        self.api_key = ENV_CREDENTIALS.get("MYPROVIDER_API_KEY")

    def validate_model(self, model_id: str) -> bool:
        # Implement model validation
        pass

    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        # Fetch model specs
        pass
```

Then register in `get_resolver()` and update docs.

---

## FAQ

**Q: Can I use local and remote models together?**
A: Yes! `allama list` shows all models (local + remote cached). Mix freely.

**Q: What if my provider goes down?**
A: Requests fail with a 503 error. No fallback (yet). Switch to a cached model or different provider.

**Q: Do you store my API key?**
A: No. It's only in `~/.allama/.env` (file on your computer) and never sent anywhere except the provider's API.

**Q: How much does it cost?**
A: Depends on the provider. OpenAI charges per token; Anthropic has different pricing. Check their rate cards.

**Q: Can I use this in production?**
A: Yes, but be aware of:
- Rate limits (some providers have strict limits)
- Downtime (no local fallback)
- Costs (especially for high-volume traffic)
- Latency (network is slower than local GPU)

For production, consider hybrid: use remote for backup, local for primary.

---

**Documentation:** docs/CONFIGURATION_ENCYCLOPEDIA.md
**Config Examples:** configs/physical/ and configs/logical/ (for reference)
**API Docs:** https://docs.vllm.ai/en/stable/serving/openai_compatible_server/
