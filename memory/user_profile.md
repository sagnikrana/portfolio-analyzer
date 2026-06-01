---
name: user-profile
description: Who the user is — hardware, goals, collaboration style, and domain knowledge
metadata:
  type: user
---

Self-directed retail investor building a personal portfolio intelligence tool. Strong product instincts — pushes for explainability, trustworthy metrics, and clean UX over feature count.

**Hardware:** Mac Studio M4 Max, 64 GB RAM — can run quantized 30B-class local models comfortably.

**LLM choice:** Ollama with llama3.3:latest (70B, 128K context). Chose it over Hugging Face because HF model loading inside Jupyter didn't work. Ollama served locally at http://127.0.0.1:11434.

**Domain knowledge:** Has a real Robinhood portfolio (CSV: mantis_invest.csv). Understands investing concepts well enough to challenge the system when metrics feel wrong. Wants the app to be honest and credible, not a "stock oracle."

**Goals:** Eventually wants the app to become a weekly monitoring system with user accounts, recommendation tracking, and "cost of ignoring the app" measurement.
