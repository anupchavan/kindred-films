#!/bin/zsh
# Codex bakeoff matrix: 5 configs x 50 movies. Sequential so a quota hit stops cleanly.
cd "$(dirname "$0")/.."
for cfg in "gpt-5.5 medium" "gpt-5.6-sol none" "gpt-5.6-sol low" "gpt-5.6-terra low" "gpt-5.6-luna low"; do
  set -- ${(z)cfg}
  echo "=== $1 @ $2 ==="
  python experiments/25_bakeoff_codex.py "$1" "$2" 2>&1 | grep -vE "Loading|Warning" | tail -3
  [[ ${pipestatus[1]} -eq 2 ]] && { echo "quota exhausted - stopping matrix"; exit 1; }
done
echo ALL-CONFIGS-DONE
