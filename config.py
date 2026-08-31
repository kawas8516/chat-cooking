import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "all-MiniLM-L6-v2")
TOP_K = int(os.environ.get("TOP_K", "3"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.3"))
DATA_DIR = os.environ.get("DATA_DIR", "data")
INDEX_PATH = os.environ.get("INDEX_PATH", "data/recipes.index")
CSV_PATH = os.environ.get("CSV_PATH", "data/recipes.csv")
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "2"))
LLM_RETRY_BACKOFF_BASE = float(os.environ.get("LLM_RETRY_BACKOFF_BASE", "1.0"))
LLM_RETRY_BACKOFF_MAX = float(os.environ.get("LLM_RETRY_BACKOFF_MAX", "8.0"))
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "10"))


class ConfigError(ValueError):
    """Raised when configuration values are invalid or missing."""


def validate() -> None:
    """Check config values are usable. Not run at import time — call explicitly
    from entry points that are about to hit the real API (app.py, eval.py
    --faithfulness) so plain `import config` stays safe in tests/CI."""
    errors = []
    if not HF_TOKEN.strip():
        errors.append("HF_TOKEN is not set. Provide it via environment variable or .env file.")
    if TOP_K <= 0:
        errors.append(f"TOP_K must be > 0, got {TOP_K}")
    if not (0 < TEMPERATURE <= 2.0):
        errors.append(f"TEMPERATURE must be in (0, 2.0], got {TEMPERATURE}")
    if not (0.0 <= SCORE_THRESHOLD <= 1.0):
        errors.append(f"SCORE_THRESHOLD must be in [0.0, 1.0], got {SCORE_THRESHOLD}")
    if MAX_NEW_TOKENS <= 0:
        errors.append(f"MAX_NEW_TOKENS must be > 0, got {MAX_NEW_TOKENS}")
    if LLM_MAX_RETRIES < 0:
        errors.append(f"LLM_MAX_RETRIES must be >= 0, got {LLM_MAX_RETRIES}")
    if MAX_HISTORY_TURNS < 0:
        errors.append(f"MAX_HISTORY_TURNS must be >= 0, got {MAX_HISTORY_TURNS}")
    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))
