#!/bin/bash
# launchd entrypoint for the weekly portfolio agent.
# Sources secrets (SMTP creds, OLLAMA_MODEL, EMAIL_TO) from a gitignored env file,
# then runs the requested weekly_job subcommand inside the project venv.
set -euo pipefail

ENV_FILE="$HOME/.portfolio-analyzer/secrets.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

cd /Users/sagnikrana/Documents/GitHub/portfolio-analyzer
exec .venv/bin/python -m automation.weekly_job "$@"
