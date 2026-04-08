# 🚀 Allama + OpenCode IDE Integration

Use Allama as your AI coding assistant directly in the **OpenCode IDE** (https://github.com/anomalyco/opencode).

---

## Quick Setup (3 Steps)

### Step 1: Start Allama
```bash
allama serve
```

Allama will start on `http://localhost:9000` with OpenAI-compatible API at `/v1`.

### Step 2: Configure OpenCode

Edit your OpenCode config file at `~/.local/share/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "allama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Allama Local",
      "options": {
        "baseURL": "http://localhost:9000/v1"
      },
      "models": {
        "Qwen3.5:27b": {
          "name": "Qwen 3.5 (27B) - Balanced"
        },
        "Qwen3.5:35b-Code": {
          "name": "Qwen 3.5 (35B) - Code"
        },
        "Qwen3.5:9b": {
          "name": "Qwen 3.5 (9B) - Fast"
        },
        "Qwen3vl:8b": {
          "name": "Qwen Vision (8B) - Vision & OCR"
        }
      }
    }
  }
}
```

### Step 3: Use in OpenCode

1. Open OpenCode
2. Select **Provider:** "Allama Local"
3. Select **Model:** Choose from dropdown
4. Start coding with AI assistance! 💻

---

## Full Configuration Guide

### File Location

OpenCode config is stored at:
```
~/.local/share/opencode/opencode.json
```

On Windows:
```
%APPDATA%\opencode\opencode.json
```

### Configuration Breakdown

#### Basic Fields

```json
{
  "provider": {
    "allama": {                    // Provider ID (you'll select this in OpenCode UI)
      "npm": "@ai-sdk/openai-compatible",  // Use OpenAI-compatible module
      "name": "Allama Local",      // Display name in OpenCode UI
      "options": {
        "baseURL": "http://localhost:9000/v1"  // Allama API endpoint
      },
      "models": { ... }
    }
  }
}
```

#### Adding Models

For each model, add an entry:

```json
"models": {
  "Qwen3.5:27b": {
    "name": "Qwen 3.5 (27B) - Balanced",
    "limit": {
      "context": 32768,      // Max context window (use lower for faster responses)
      "output": 8192         // Max output tokens
    }
  }
}
```

### Model Configuration Examples

#### For Code Generation
```json
"Qwen3.5:35b-Code": {
  "name": "Qwen 3.5 (35B) - Best for Code",
  "limit": {
    "context": 32768,
    "output": 8192
  }
}
```

#### For Fast Responses
```json
"Qwen3.5:9b": {
  "name": "Qwen 3.5 (9B) - Fast/Lightweight",
  "limit": {
    "context": 16384,
    "output": 4096
  }
}
```

#### For Vision/Analysis
```json
"Qwen3vl:8b": {
  "name": "Qwen Vision (8B) - Image/Code Analysis",
  "limit": {
    "context": 16384,
    "output": 4096
  }
}
```

### Advanced: Custom Headers

If you add authentication to Allama later:

```json
{
  "provider": {
    "allama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Allama Local (Secured)",
      "options": {
        "baseURL": "http://localhost:9000/v1",
        "apiKey": "{env:ALLAMA_API_KEY}",
        "headers": {
          "X-Custom-Header": "value"
        }
      },
      "models": { ... }
    }
  }
}
```

Then set environment variable:
```bash
export ALLAMA_API_KEY="your-key"
```

---

## Complete Config Example

Here's a full config with all your models optimized:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "allama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Allama (Local AI)",
      "options": {
        "baseURL": "http://localhost:9000/v1"
      },
      "models": {
        "Qwen3.5:27b": {
          "name": "🧠 Balanced (27B) - General purpose",
          "limit": { "context": 32768, "output": 8192 }
        },
        "Qwen3.5:35b-Code": {
          "name": "💻 Code Master (35B) - Best for programming",
          "limit": { "context": 32768, "output": 8192 }
        },
        "Qwen3.5:35b-Instruct": {
          "name": "📚 Instructor (35B) - Teaching & explanation",
          "limit": { "context": 32768, "output": 8192 }
        },
        "Qwen3.5:35b-Instruct-Reasoning": {
          "name": "🎯 Reasoning (35B) - Complex problems",
          "limit": { "context": 32768, "output": 8192 }
        },
        "Qwen3.5:9b": {
          "name": "⚡ Quick (9B) - Fast responses",
          "limit": { "context": 16384, "output": 4096 }
        },
        "Qwen3vl:8b": {
          "name": "👁️ Vision (8B) - Image analysis & OCR",
          "limit": { "context": 8192, "output": 4096 }
        }
      }
    }
  }
}
```

---

## Usage Patterns

### Pattern 1: Quick Questions (Fast Model)
```
Provider: Allama Local
Model: Quick (9B)
Task: "Explain this code snippet"
→ Returns in 1-2 seconds
```

### Pattern 2: Code Generation (Powerful Model)
```
Provider: Allama Local
Model: Code Master (35B)
Task: "Write a Python function to parse JSON"
→ Returns in 2-4 seconds
```

### Pattern 3: Complex Analysis (Reasoning Model)
```
Provider: Allama Local
Model: Reasoning (35B)
Task: "Debug this algorithm - why is it slow?"
→ Returns in 3-5 seconds with detailed analysis
```

### Pattern 4: Image/Document Analysis (Vision)
```
Provider: Allama Local
Model: Vision (8B)
Task: Upload code screenshot → "Extract and format this code"
→ Returns OCR'd code in 2-3 seconds
```

---

## Troubleshooting

### Problem: "Connection refused" in OpenCode

**Check Allama is running:**
```bash
allama status
```

If not running:
```bash
allama serve
```

**Verify API is accessible:**
```bash
curl http://localhost:9000/v1/models
```

### Problem: Models don't appear in OpenCode dropdown

**Solution:**
1. Restart OpenCode (close and reopen)
2. Verify JSON syntax in `opencode.json` is valid
3. Check that Allama is serving those models:
   ```bash
   allama list
   ```

### Problem: OpenCode can't connect to Allama

**Possible causes & solutions:**

1. **Wrong port:**
   - Check Allama port: `allama status` (default 9000)
   - Update baseURL: `http://localhost:XXXX/v1`

2. **Wrong hostname:**
   - Same machine: `localhost`
   - Different machine: Use IP: `http://192.168.1.100:9000/v1`

3. **Firewall blocking:**
   - Allow port 9000 in your firewall
   - Or run both on same machine

### Problem: Responses are slow

**Solutions:**
1. Use smaller model (`9b` instead of `35b`)
2. Reduce `context` window in config
3. Check if Allama is busy: `allama ps`
4. Ensure only one model is loaded at a time

---

## Performance Tips

### For Coding Assistance

**Recommended setup:**
```json
"Qwen3.5:35b-Code": {
  "name": "Code Master",
  "limit": {
    "context": 8192,      // Keep smaller for speed
    "output": 4096
  }
}
```

**Settings in OpenCode:**
- Temperature: 0.3 (factual)
- Max tokens: 2048 (quick responses)

### For Long Code Files

**Recommended setup:**
```json
"Qwen3.5:27b": {
  "name": "Balanced",
  "limit": {
    "context": 32768,     // Large context for full files
    "output": 8192
  }
}
```

### For Quick Suggestions

**Use the fast model:**
```json
"Qwen3.5:9b": {
  "name": "Quick",
  "limit": {
    "context": 4096,      // Very small
    "output": 1024        // Quick responses
  }
}
```

---

## Environment Variables

You can reference environment variables in config:

```json
{
  "options": {
    "baseURL": "{env:ALLAMA_API_URL}",
    "apiKey": "{env:ALLAMA_API_KEY}"
  }
}
```

Set environment variables:
```bash
export ALLAMA_API_URL="http://localhost:9000/v1"
export ALLAMA_API_KEY="your-key"
```

---

## Multiple Machines

### Scenario: Allama on Server, OpenCode on Laptop

1. **Start Allama on server (192.168.1.100):**
   ```bash
   ALLAMA_HOST=0.0.0.0 allama serve
   ```

2. **Configure OpenCode with server IP:**
   ```json
   {
     "baseURL": "http://192.168.1.100:9000/v1"
   }
   ```

3. **Use normally in OpenCode UI**

---

## API Compatibility

Allama uses the **OpenAI-compatible API**, so any `@ai-sdk/openai-compatible` client works:
- ✅ OpenCode (this guide)
- ✅ Vercel AI SDK
- ✅ Custom applications
- ✅ LiteLLM proxies

---

## Next Steps

- 📖 [Allama Configuration](CONFIGURATION_ENCYCLOPEDIA.md)
- 🎯 [Your Setup Guide](YOUR_SETUP_GUIDE.md)
- 🐛 [Troubleshooting](OPENCODE_INTEGRATION.md)

---

**Last Updated:** April 2026
**Tested with:** OpenCode 1.4.0, Allama latest
**Config Format:** OpenCode opencode.json v1.0
