"""Shared local Ollama client (no API key, data stays on the machine).

Used by the dashboard assistant (plain-English summary) and the weekly
automation digest, so there's a single implementation of the local-LLM call.
Configure with OLLAMA_MODEL (default qwen2.5:32b — a strong mid-size model that
fits comfortably in memory; llama3.1:latest is a faster, lighter option) and
OLLAMA_HOST.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Optional

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:32b")
# Mid-size models are slower than 8B; give generous headroom (still under the
# patched Gradio queue timeout of 600s used by the dashboard).
OLLAMA_TIMEOUT_S = int(os.environ.get("OLLAMA_TIMEOUT_S", "600"))
# Keep the model resident between calls so a fresh analysis doesn't pay the
# (large, ~20GB for 32B) model-reload cost each time. "-1" = keep loaded
# indefinitely; e.g. "30m" to free memory after 30 idle minutes.
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")


def ollama_available() -> bool:
    """Quick reachability check so the UI can fail gracefully if Ollama is off."""
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def ollama_chat(
    system: str,
    user: str,
    *,
    model: Optional[str] = None,
    history: Optional[list[dict]] = None,
    temperature: float = 0.3,
    timeout: Optional[int] = None,
) -> Optional[str]:
    """Call the local Ollama chat API. Returns text, or None on any failure.

    `history` is an optional list of prior {"role": "user"|"assistant", "content"}
    turns (for the conversational assistant).
    """
    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})
    body = json.dumps({
        "model": model or OLLAMA_MODEL,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"temperature": temperature},
        "messages": messages,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout or OLLAMA_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("message", {}).get("content") or "").strip() or None
    except Exception:
        return None
