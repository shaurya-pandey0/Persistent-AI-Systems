# core/api_client.py
"""
OpenRouter API client (v2) – handles auth, errors and .env loading.
"""

from __future__ import annotations
import os, json, requests, sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# -----------------------------------------------------------
# 1. Load environment variables from .env automatically
# -----------------------------------------------------------
try:
    from dotenv import load_dotenv                       # pip install python-dotenv
    load_dotenv(Path.cwd() / ".env", override=False)
except ModuleNotFoundError:
    # fallback – read .env manually if python-dotenv isn’t installed
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# -----------------------------------------------------------
# 2. Constants
# -----------------------------------------------------------
BASE_URL          = "https://openrouter.ai/api/v1"
CHAT_ENDPOINT     = f"{BASE_URL}/chat/completions"
DEFAULT_MODEL     = "tngtech/deepseek-r1t2-chimera:free"
TIMEOUT           = 60          # seconds
REFERRER_HEADER   = "http://localhost:8501"  # Streamlit default
APP_TITLE_HEADER  = "Memory-Based AI Agent"

# -----------------------------------------------------------
# 3. Custom error
# -----------------------------------------------------------
class OpenRouterError(Exception):
    def __init__(self, message: str, status_code: int | None = None,
                 response_data: dict | None = None):
        super().__init__(message)
        self.status_code   = status_code
        self.response_data = response_data or {}

# -----------------------------------------------------------
# 4. API client
# -----------------------------------------------------------
class OpenRouterClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY not found in environment variables or .env file.\n"
                "Get one at https://openrouter.ai/keys and set it like:\n"
                "   export OPENROUTER_API_KEY=sk-or-v1-xxxxx"
            )

        if not self.api_key.startswith("sk-or-v1-"):
            print(f"[OpenRouter] Warning: API-key should start with 'sk-or-v1-'. "
                  f"Your key starts with: {self.api_key[:10]}…", file=sys.stderr)

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": REFERRER_HEADER,
            "X-Title": APP_TITLE_HEADER,
        }

    # ------------- main request wrapper -------------
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> Dict:
        if not messages:
            raise ValueError("messages array may not be empty")

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        print(f"[OpenRouter] POST {CHAT_ENDPOINT}  – {len(messages)} msgs, "
              f"model={model}, temp={temperature}")

        try:
            r = requests.post(CHAT_ENDPOINT,
                              headers=self.headers,
                              json=payload,
                              timeout=TIMEOUT)
        except requests.exceptions.RequestException as e:
            raise OpenRouterError(f"Network error: {e}")

        if r.status_code == 200:
            return r.json()

        # --- translate common HTTP errors ---
        try:
            err_json = r.json()
            err_msg  = err_json.get("error", {}).get("message", r.text)
        except Exception:
            err_json, err_msg = {}, r.text

        match r.status_code:
            case 401:
                raise OpenRouterError(
                    "401 Unauthorized – check OPENROUTER_API_KEY.", 401, err_json)
            case 402:
                raise OpenRouterError(
                    "402 Payment Required – add / renew credits.", 402, err_json)
            case 429:
                raise OpenRouterError(
                    "429 Rate-limit exceeded – slow down.", 429, err_json)
            case _:
                raise OpenRouterError(
                    f"HTTP {r.status_code}: {err_msg}", r.status_code, err_json)

# -----------------------------------------------------------
# 5. Helper – single completion wrapper
# -----------------------------------------------------------
_client: Optional[OpenRouterClient] = None

def get_client() -> OpenRouterClient:
    global _client
    if _client is None:
        _client = OpenRouterClient()
    return _client

def get_completion(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    **kwargs,
) -> str:
    data = get_client().chat_completion(messages, model, temperature,
                                        max_tokens, **kwargs)
    # Extract first choice text
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise OpenRouterError("Malformed response from OpenRouter", response_data=data)

__all__ = ["OpenRouterError", "get_completion", "OpenRouterClient"]
