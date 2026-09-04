# Fixerr User Guide

Fixerr helps you stop solving the same errors twice by capturing failures and their fixes, then surfacing solutions automatically when similar issues reoccur.

## Installation

Install fixerr with pip:

```bash
pip install fixerr                    # Core functionality with Ollama backend
pip install fixerr[openai]            # Add OpenAI/OpenAI-compatible support
pip install fixerr[anthropic]         # Add Anthropic (Claude) support
pip install fixerr[gemini]            # Add Google Gemini support
pip install fixerr[all-providers]     # Install all available AI backends
```

Verify your installation:

```bash
fixerr --help
fixerr doctor
```

## Configuration

Fixerr uses a TOML configuration file at `~/.fixerr/config.toml` (or path specified by `fixerr_CONFIG` environment variable).

Key configuration sections:

### AI Provider Settings
```toml
[ai]
provider = "ollama"           # Choose: ollama, openai, openai-compatible, anthropic, gemini
embed_fallback = "ollama"     # Used when provider doesn't support embeddings (e.g., anthropic)
```

### Provider-Specific Configuration
```toml
[ai.ollama]
host = "http://localhost:11434"
embed_model = "nomic-embed-text"
gen_model = "llama3"
embed_timeout = 30
gen_timeout = 180

[ai.openai]
api_key_env = "OPENAI_API_KEY"
embed_model = "text-embedding-3-small"
gen_model = "gpt-4o-mini"

[ai.anthropic]
api_key_env = "ANTHROPIC_API_KEY"
gen_model = "claude-haiku-4-5"
# Uses embed_fallback for embeddings

[ai.gemini]
api_key_env = "GEMINI_API_KEY"
embed_model = "models/text-embedding-004"
gen_model = "gemini-1.5-flash"
```

### Capture Behavior
```toml
[capture]
auto_capture = true           # Enable/disable shell hook
min_exit_code = 1             # Minimum exit code to treat as failure
surface_threshold = 1         # Minimum resolved matches to show fix box
quiet = false                 # Silent capture mode
ignore_commands = ["cd", "ls", "cat", "echo", "man", "git log", "git status", "git diff", "clear", "exit"]

[capture.ignore_patterns]
patterns = ["^vim ", "^nano ", "^less "]  # Regex patterns to ignore
```

### Environment Variables
Set these before running fixerr:

| Variable | Purpose | Default |
|----------|---------|---------|
| `fixerr_CONFIG` | Path to config file | `~/.fixerr/config.toml` |
| `fixerr_DB` | Path to SQLite database | `~/.fixerr/errors.db` |
| `fixerr_SIM_THRESHOLD` | Similarity threshold for auto-surfacing | `0.7` |
| `OLLAMA_HOST` | Ollama daemon URL | `http://localhost:11434` |
| `ANTHROPIC_API_KEY` | API key for Anthropic | *(required if using anthropic)* |
| `OPENAI_API_KEY` | API key for OpenAI | *(required if using openai)* |
| `GEMINI_API_KEY` | API key for Gemini | *(required if using gemini)* |

## Usage

### Basic Workflow

1. **Capture an error**
   ```bash
   # From a failing command:
   failing-command 2>&1 | fixerr capture -c "failing-command"
   
   # Interactive capture:
   fixerr capture -c "my-command"
   # Paste error output, then Ctrl-D
   ```

2. **Record the fix**
   ```bash
   fixerr resolve 1 "Applied the solution here"
   ```

3. **Find similar errors later**
   ```bash
   fixerr search "error description"
   ```

4. **View full details**
   ```bash
   fixerr show 1
   ```

### Automatic Error Detection

Enable automatic capture with shell integration:

**Zsh:**
```bash
eval "$(fixerr shell-init zsh)"
```

**Bash:**
```bash
eval "$(fixerr shell-init bash)"
```

**PowerShell:**
```powershell
fixerr shell-init pwsh | Out-String | Invoke-Expression
```

### Advanced Commands

- `fixerr run <command>` - Execute command, capture stderr on failure with semantic search
- `fixerr explain <id>` - Generate AI explanation for error (background process)
- `fixerr edit <id>` - Edit fix text in your preferred editor
- `fixerr dismiss <id>` - Mark error as won't-fix (excluded from auto-surfacing)
- `fixerr reopen <id>` - Reopen resolved/won't-fix error
- `fixerr dashboard` - Launch interactive TUI interface
- `fixerr config set <key> <value>` - Modify configuration
- `fixerr doctor` - Check AI provider health and connectivity

### Programmatic Access

```python
from fixerr import ErnestClient

client = ErnestClient()

# Check if we've seen this error before
fix = client.surface(stderr_text)
if fix:
    print(f"Found fix: {fix}")

# Search for similar errors
results = client.search("database connection failed", top_k=5)

# Record new error programmatically
error_id = client.record("my-command", "Error output here")
client.resolve(error_id, "Fixed by doing X")
```

## AI Provider Setup

### Ollama (Default - Zero Configuration)
```bash
ollama serve
ollama pull nomic-embed-text
ollama pull llama3
# fixerr works immediately with defaults
```

### OpenAI
```bash
pip install fixerr[openai]
export OPENAI_API_KEY=your-key-here
fixerr config set ai.provider openai
fixerr doctor
```

### Anthropic (Claude)
```bash
pip install fixerr[anthropic]
export ANTHROPIC_API_KEY=your-key-here
fixerr config set ai.provider anthropic
# Ensure Ollama is running for embeddings fallback
fixerr doctor
```

### Gemini
```bash
pip install fixerr[gemini]
export GEMINI_API_KEY=your-key-here
fixerr config set ai.provider gemini
fixerr doctor
```

## How It Works

1. **Capture** - Error output and failing command are received
2. **Redact** - Secrets, API keys, and volatile tokens are removed/normalized
3. **Fingerprint** - Deterministic hash identifies recurring errors (ignores variable data)
4. **Store** - SQLite database maintains one record per unique error with occurrence tracking
5. **Search** - Uses semantic embeddings (when AI available) or text overlap (fallback)
6. **Surface** - When similar error occurs, displays most relevant past fix

The dashboard provides visual interaction with your error history, showing recurrence counts, timestamps, and semantic clustering of similar issues.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `fixerr doctor` shows unreachable models | Verify API keys, provider availability, and correct configuration |
| No semantic search matches | Check embedding backend is running and accessible |
| Shell hook not capturing commands | Re-run `fixerr shell-init <shell>` after config changes |
| Missing provider SDK | Install with appropriate extra: `pip install fixerr[<provider>]` |
| Configuration changes not taking effect | Restart your shell or application to reload config |

For detailed command references, run `fixerr <command> --help` on any subcommand.