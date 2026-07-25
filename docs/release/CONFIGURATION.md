# Configuration Conventions

Configuration templates use repository-relative paths. Inputs belong under a
local `data/` directory and generated output belongs under `outputs/`; both
locations are excluded from version control.

When a workflow uses an OpenAI-compatible service, set the endpoint and
credential in the environment before running it:

```bash
export BENCHMARK_API_BASE_URL="http://localhost:8000/v1"
export BENCHMARK_API_KEY="replace-with-local-credential"
```

For local model loading, provide a path through the relevant command-line
option or configuration value. The repository intentionally does not prescribe
a filesystem location, remote endpoint, credential, or model inventory.
