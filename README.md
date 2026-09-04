# fixerr

> Local-first, provider-agnostic error/fix memory with semantic search and smart deduplication


[![PyPI](https://img.shields.io/pypi/v/fixerr.svg)](https://pypi.org/project/fixerr/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://pypi.org/project/fixerr/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 The Problem

Every developer repeatedly solves the same errors:
- `port already in use` after switching branches
- `ModuleNotFoundError` for a package that *was* installed
- Docker permission errors in CI/CD
- Environment-specific configuration issues

fixerr captures these errors and their fixes, then surfaces the solution automatically when the same failure reoccurs — saving you from reinventing the wheel.

## ✨ Key Features

### 🧠 Smart Error Deduplication
- Automatically detects recurring errors using deterministic fingerprints
- Tracks occurrence frequency (`seen 5x`) and timestamps (first/last seen)
- Upgrades lightweight Tier 1 captures to full Tier 2 records seamlessly

### 🔍 Hybrid Search
- Semantic search via embeddings (when AI backend available)
- Falls back to exact/error-code matching when offline
- Combines both for maximum recall and precision

### 🛡️ Privacy-First by Design
- 100% local: SQLite database on your machine, no telemetry
- Automatic redaction of secrets, API keys, and volatile tokens
- No account required, no server to maintain

### 🤖 Provider-Agnostic AI
- Default: Ollama (fully offline)
- Swap to OpenAI, Anthropic, Gemini, or OpenAI-compatible with one config
- Graceful degradation to text-only search when no AI backend available

### 💻 Beautiful Terminal Dashboard
- Interactive TUI built with Textual
- Search, filter, resolve/dismiss errors, copy fixes
- Recurrence badges (`×<count>`) and occurrence metrics
- Semantic clustering to spot your most costly error patterns

## 🚀 Installation

```bash
# Core (uses local Ollama by default)
pip install fixerr

# With specific AI providers
pip install fixerr[openai]        # OpenAI / OpenAI-compatible
pip install fixerr[anthropic]     # Anthropic (Claude)
pip install fixerr[gemini]        # Google Gemini
pip install fixerr[all-providers] # Everything
```

Requires Python 3.9+. Works without any AI provider configured (offline text matching only).

## ⚡ Quick Start

### Manual Capture

```bash
# Capture a failing command
some-command-that-fails 2>&1 | fixerr capture -c "some-command-that-fails"

# Record the fix
fixerr resolve 1 "increased timeout in docker-compose.yml"

# Search for similar errors later
fixerr search "connection refused on startup"
```

### Automatic Capture (Recommended)

**Zsh / Bash**
Add to your `~/.zshrc` or `~/.bashrc`:

```bash
eval "$(fixerr shell-init zsh)"   # zsh
eval "$(fixerr shell-init bash)"  # bash
```

**PowerShell**
Add to your PowerShell profile:

```powershell
fixerr shell-init pwsh | Out-String | Invoke-Expression
```

Now failed commands are captured automatically — no extra step needed.

### Explore Your Error History

```bash
fixerr dashboard    # Interactive TUI
fixerr doctor       # Check AI provider status
fixerr explain 5    # Get AI explanation for error #5
```


## 🔧 How It Works

1. **Capture**: Errors are captured via `fixerr capture`, `fixerr run`, or automatic shell hook
2. **Redact**: Secrets, API keys, emails, and volatile tokens are stripped/normalized
3. **Fingerprint**: Deterministic hash identifies recurring errors (ignoring timestamps, paths, etc.)
4. **Store**: SQLite database stores one record per unique error, with occurrence count
5. **Search**: 
   - Semantic: Cosine similarity over embeddings (when AI backend available)
   - Fallback: Jaccard token overlap over normalized text
   - Hybrid: Combines both (Phase 2)
6. **Surface**: When a similar error occurs, fixerr shows the most relevant past fix

## 📊 The Dashboard (`fixerr dashboard`)

**Left Pane**:
- Filterable, searchable error list with recurrence badges (`×<count>`)

**Right Pane**: 
- Metadata (first/last seen, occurrence count)
- Full error output and applied fix
- AI explanation (when available)
- Top 3 semantically similar past errors

**Patterns tab**:
- Clusters errors via k-means over embeddings to spot your most time-consuming issues

**Keybindings**:
| Key | Action |
|-----|--------|
| `j`/`k` or `↑`/`↓` | Navigate error list |
| `r` | Mark selected error resolved |
| `d` | Mark selected error won't-fix |
| `u` | Reopen resolved/won't-fix error |
| `e` | Generate/show AI explanation |
| `c` | Copy fix to clipboard |
| `s` | Focus search |
| `/` | Focus filter chips |
| `p` or `1`-`4` | Switch tabs (All/Unresolved/Resolved/Patterns) |
| `q`/`Ctrl-C` | Quit |

## 🤖 AI Backends

Configure in `~/.fixerr/config.toml`:
```toml
[ai]
provider = "ollama"          # ollama | openai | openai-compatible | anthropic | gemini
embed_fallback = "ollama"    # Used when provider can't embed (e.g. anthropic)

[ai.ollama]
host = "http://localhost:11434"
embed_model = "nomic-embed-text"
gen_model = "llama3"

[ai.anthropic]
api_key_env = "ANTHROPIC_API_KEY"
gen_model = "claude-haiku-4-5"
```

Switch providers via CLI:
```bash
fixerr config set ai.provider anthropic
fixerr config show  # See effective config (secrets masked)
```

## 💡 Embedding in Other Tools

```python
from fixerr import ErnestClient

client = ErnestClient()

# Surface a fix for stderr text (returns None if no confident match)
fix = client.surface(stderr_text)

# Search for similar errors in a project context
recent = client.search(
    "build failed", 
    cwd="/path/to/project", 
    top_k=3
)
```

Wrap in `try/except ImportError` to make fixerr an optional dependency.

## ⚙️ Configuration

Environment Variables:
| Variable | Purpose | Default |
|----------|---------|---------|
| `fixerr_CONFIG` | Path to config file | `~/.fixerr/config.toml` |
| `fixerr_DB` | Path to SQLite store | `~/.fixerr/errors.db` |
| `fixerr_SIM_THRESHOLD` | Similarity cutoff for auto-surfacing | `0.7` |
| `OLLAMA_HOST` | Ollama daemon URL | `http://localhost:11434` |

Tune Tier 1 auto-capture via `[capture]` in `~/.fixerr/config.toml`:
- `auto_capture`: Enable/disable
- `min_exit_code`: Minimum exit code to consider failure
- `ignore_commands`: List of commands to ignore (e.g. `["cd", "git status"]`)
- `ignore_patterns`: Regex patterns to ignore
- `surface_threshold`: Similarity threshold for auto-surfacing
- `quiet`: Suppress non-essential output

Full reference: [USAGE.md](USAGE.md)

## 🧪 Contributing

Issues and PRs welcome! To run the test suite:

```bash
# Install dev dependencies
pip install -e ".[dev,all-providers]"

# Run tests
pytest

# Check code style
ruff check .
```

Provider tests use mocks — no real API calls or Ollama instance needed.

## 📜 License

[MIT](LICENSE) © 2024 ucmani