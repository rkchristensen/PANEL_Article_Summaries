#!/bin/zsh
set -e
cd /Users/robertchristensen/Documents/GitHubRepos/PANEL_Article_Summaries

STAMP=".last_local_refresh_date"
TODAY="$(date +%F)"

if [[ -f "$STAMP" ]] && [[ "$(cat "$STAMP")" == "$TODAY" ]]; then
  exit 0
fi

git pull --rebase origin main
export OLLAMA_MODEL=qwen2.5:3b-instruct
export ENABLE_OLLAMA=1
python3 scripts/update_stories.py
git add data/stories.json data/summaries_cache.json
git diff --cached --quiet || { git commit -m "chore: daily local refresh with Ollama summaries"; git push; }

echo "$TODAY" > "$STAMP"
