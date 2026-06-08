"""Shared local Ollama client (no API key, data stays on the machine).

Used by both the dashboard assistant (plain-English summary + chat) and the
weekly automation digest, so there's a single implementation of the local-LLM
call. Configure with OLLAMA_MODEL (default llama3.1:latest; llama3.3:latest for
higher quality) and OLLAMA_HOST.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Optional

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")
OLLAMA_TIMEOUT_S = int(os.environ.get("OLLAMA_TIMEOUT_S", "300"))


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
