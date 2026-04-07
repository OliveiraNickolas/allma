# 🧪 Testing Guide: Remote Providers

This guide helps you test the new OpenCode/OpenClaw integration.

## Prerequisites

1. **API Key:** Obtain from OpenCode or OpenClaw
2. **Budget:** Each test API call costs ~$0.0001-0.001
3. **Time:** 5-10 minutes per test

## Test 1: Basic Setup (No API Call)

**Goal:** Verify code loads without errors.

```bash
cd /home/nick/AI/allama

# Test Python syntax
python3 -m py_compile core/config.py core/provider_resolver.py allama_cli.py

# Should output nothing (no errors)
```

✅ **Expected:** No errors
❌ **If failed:** Check Python version (3.10+)

---

## Test 2: Configuration Loading

**Goal:** Verify `.env` file and dynamic models load correctly.

```bash
python3 << 'EOF'
from core.config import ENV_CREDENTIALS, DYNAMIC_MODELS, load_dynamic_models

print("Credentials loaded:", bool(ENV_CREDENTIALS))
print("Dynamic models:", DYNAMIC_MODELS)

# Should print: Credentials loaded: False (empty) or True (if .env exists)
# Should print: Dynamic models: {} (empty dict)
EOF
```

✅ **Expected:** No errors, dicts printed
❌ **If failed:** Check `.env` file exists and is readable

---

## Test 3: Provider Resolver (Mock)

**Goal:** Verify resolver class structure without API calls.

```bash
python3 << 'EOF'
from core.provider_resolver import OpenCodeResolver, OpenClawResolver, get_resolver

# Test class instantiation (no API calls yet)
oc = get_resolver("opencode")
oclaw = get_resolver("openclaw")
unknown = get_resolver("unknown")

assert oc is not None, "OpenCode resolver failed"
assert oclaw is not None, "OpenClaw resolver failed"
assert unknown is None, "Unknown provider should return None"

print("✅ All provider resolvers instantiated correctly")
EOF
```

✅ **Expected:** "✅ All provider resolvers..."
❌ **If failed:** Check `core/provider_resolver.py` syntax

---

## Test 4: CLI Parser (New Syntax)

**Goal:** Verify new `allama run <model> on <provider>` parsing.

**Note:** This test requires the server to be running. Start it first:

```bash
# In one terminal
cd /home/nick/AI/allama
allama serve --verbose
```

**In another terminal:**

```bash
# Test the new CLI syntax (will fail without API key, but parser should work)
cd /home/nick/AI/allama
python3 allama_cli.py run gpt-4 on opencode 2>&1 | head -20

# Should output an error like:
# ❌ Model 'gpt-4' not found on opencode
# (This is expected! Shows parser worked, resolver tried API.)
```

✅ **Expected:** Parser error (not syntax error)
❌ **If failed:** Check argparse definition in `allama_cli.py`

---

## Test 5: Full Integration (With API Key)

**Goal:** End-to-end test with real API call.

**Prerequisites:**
- OpenCode API key (from https://opencode.ai)
- $1+ in account balance

**Setup:**

```bash
# Create ~/.allama/.env
mkdir -p ~/.allama
cat > ~/.allama/.env << 'EOF'
OPENCODE_API_KEY=sk-<your-key-here>
EOF

chmod 600 ~/.allama/.env  # Secure the file
```

**Test:**

```bash
# Terminal 1: Start server
cd /home/nick/AI/allama
allama serve --verbose

# Terminal 2: List models
allama list

# Should show error or models (depending on server state)

# Terminal 3: Run a model
allama run gpt-4 on opencode

# Should prompt for input or error if API key invalid
```

✅ **Expected:** Either chat prompt or clear error message
❌ **If failed:** Check API key is correct and has balance

---

## Test 6: Dynamic Model Persistence

**Goal:** Verify `--persist` flag saves model config.

```bash
# With API key set up:
allama run gpt-4 on opencode --persist

# Check if saved
cat ~/.allama/dynamic_models.json | python3 -m json.tool

# Should show:
# {
#   "opencode/gpt-4": {
#     "backend": "opencode",
#     "model_id": "gpt-4",
#     ...
#   }
# }
```

✅ **Expected:** JSON file contains model config
❌ **If failed:** Check file permissions on `~/.allama/`

---

## Test 7: Server Model List

**Goal:** Verify remote models appear in `/v1/models` endpoint.

```bash
# With server running:
curl http://localhost:9000/v1/models 2>/dev/null | python3 -m json.tool

# Should list both:
# - Static models (from configs/)
# - Dynamic models (from ~/.allama/dynamic_models.json)
```

✅ **Expected:** JSON response with model list
❌ **If failed:** Server not running or models not loading

---

## Test 8: Documentation

**Goal:** Verify guides are complete and readable.

```bash
ls -lh /home/nick/AI/allama/docs/

# Should show:
# CONFIGURATION_ENCYCLOPEDIA.md (3000+ lines)
# REMOTE_PROVIDERS.md (250+ lines)
# TESTING_GUIDE.md (this file)
```

✅ **Expected:** All files exist and are readable
❌ **If failed:** Files not created or corrupted

---

## Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'core'"

**Solution:** Run from the allama directory:
```bash
cd /home/nick/AI/allama
python3 ...
```

---

### Issue: "OPENCODE_API_KEY not set"

**Solution:** Create `~/.allama/.env`:
```bash
mkdir -p ~/.allama
echo "OPENCODE_API_KEY=sk-..." > ~/.allama/.env
```

---

### Issue: "Connection failed" or timeout

**Solution:** Verify provider API is up:
```bash
curl -I https://api.opencode.io/v1/models -w "%{http_code}\n"
# Should output: 200 or 401 (not 0 or timeout)
```

---

### Issue: "Model not found" even with valid API key

**Solution:** Check exact model ID on provider's docs:
- OpenCode: https://opencode.ai/docs/models
- OpenClaw: https://docs.openclaw.ai/cli/models

---

## Performance Benchmarks (Reference)

On a typical internet connection:
- Model lookup: 100-500ms
- Single message: 500ms-2s (depends on model)
- Streaming: Real-time (100+ tokens/sec)

Expected from provider:
- First API call: Warm up (1-2s)
- Subsequent calls: Faster (500ms-1s)

---

## Budget Tracking

| Test | Cost | Notes |
|------|------|-------|
| 1-4 | $0 | No API calls |
| 5-8 | ~$0.001-0.01 | Small API calls |
| **Total** | **<$0.05** | Very cheap testing |

---

## Sign-Off Checklist

Once all tests pass:

- [ ] Test 1: Syntax validation passes
- [ ] Test 2: Config loading works
- [ ] Test 3: Provider resolvers instantiate
- [ ] Test 4: CLI parser handles new syntax
- [ ] Test 5: Full integration works (with API)
- [ ] Test 6: Dynamic model persistence works
- [ ] Test 7: Server endpoint includes remote models
- [ ] Test 8: Documentation complete

**Once all checked:** ✅ **Remote provider support is production-ready!**

---

## Next Steps

1. **Commit:** Save your work
   ```bash
   cd /home/nick/AI/allama
   git add -A
   git commit -m "feat: OpenCode/OpenClaw remote provider support + CLI syntax + docs"
   ```

2. **Document:** Update main README with new feature
3. **Monitor:** Watch for API costs and rate limits
4. **Expand:** Add more providers as needed

---

Questions? Check:
- `docs/REMOTE_PROVIDERS.md` — User guide
- `docs/CONFIGURATION_ENCYCLOPEDIA.md` — Parameter reference
- `core/provider_resolver.py` — Implementation details
