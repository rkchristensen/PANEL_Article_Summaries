# Nonmarket Ethics Scholarship Feed

This project builds a daily-updated nonmarket ethics scholarship board with two columns:

- Left column: Government
- Right column: Nonprofit

Each story appears as a small tile:

- Light green background for positive stories
- Light red background for negative stories
- Clickable tile linking to the source article
- Newest stories first

If a story matches both Government and Nonprofit, it is shown in both columns.

## Files

- `index.html`: page structure
- `styles.css`: page styling
- `app.js`: client-side rendering from JSON data
- `data/stories.json`: generated story data
- `scripts/update_stories.py`: daily fetch/classify/generate script
- `.github/workflows/update-stories.yml`: GitHub Action for daily updates

## Local run

Generate fresh story data:

```bash
python3 scripts/update_stories.py
```

Open the page:

```bash
open index.html
```

## Auto-update schedule

GitHub Actions runs once per day and commits updated `data/stories.json`.
