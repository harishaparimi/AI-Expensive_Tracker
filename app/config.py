"""App settings, read from the .env file in the project root."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Tried in order: the first is the main model, the rest are fallbacks.
LLM_MODELS = [m.strip() for m in os.getenv("LLM_MODELS", "").split(",") if m.strip()]

# Databases stay outside the project, so an uploaded statement cannot sync to OneDrive or reach git.
DATA_DIR = Path(os.getenv("DATA_DIR") or Path.home() / ".expense-tracker").expanduser()
