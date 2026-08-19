#!/bin/zsh
# Generate profile cards for all cleaned subtitles via codex (gpt-5.5, low reasoning).
# Idempotent: skips movies that already have a non-empty profile. Run again to retry failures.
cd "$(dirname "$0")"
mkdir -p data/profiles logs
PROMPT=$(cat profile_prompt.txt)

profile_one() {
  local txt=$1
  local id=$(basename "$txt" .txt)
  local out="data/profiles/${id}.json"
  [[ -s "$out" ]] && return 0
  codex exec --skip-git-repo-check --sandbox read-only \
    -m gpt-5.5 -c model_reasoning_effort=low \
    -o "$out" "$PROMPT $(pwd)/$txt" </dev/null >"logs/${id}.log" 2>&1
  [[ -s "$out" ]] || echo "EMPTY $id" >> logs/failures.txt
}

export -f profile_one 2>/dev/null || true

N=0
for txt in data/clean/*.txt; do
  profile_one "$txt" &
  (( ++N % 8 == 0 )) && wait
done
wait
echo "done: $(ls data/profiles | wc -l) profiles"
