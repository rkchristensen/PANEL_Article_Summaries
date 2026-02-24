#!/bin/zsh
set -e
cd ~/Documents/GitHubRepos/PANEL_Article_Summaries
git pull --rebase origin main
export OLLAMA_MODEL=qwen2.5:3b-instruct
export ENABLE_OLLAMA=1
python3 scripts/update_stories.py
git add data/stories.json data/summaries_cache.json
git diff --cached --quiet && exit 0
git commit -m "chore: daily local refresh with Ollama summaries"
git push
