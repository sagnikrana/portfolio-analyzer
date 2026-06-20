#!/bin/bash
# launchd entrypoint for the always-on Gradio dashboard.
# Sources secrets (so the Send-Portfolio-Summary email button works), disables
# the Gradio share tunnel (the public URL is provided by Tailscale Funnel /
# Cloudflare Tunnel), and runs the app bound to 127.0.0.1:7863.
set -euo pipefail

ENV_FILE="$HOME/.portfolio-analyzer/secrets.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

export PA_DISABLE_SHARE=1
cd /Users/sagnikrana/Documents/GitHub/portfolio-analyzer
exec .venv/bin/python -m portfolio_analyzer
