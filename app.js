function formatDate(value) {
  if (!value) return "Unknown date";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "Unknown date";
  return parsed.toLocaleString();
}

function buildTile(story) {
  const tile = document.createElement("details");
  tile.className = `tile ${story.sentiment || "negative"}`;

  const header = document.createElement("summary");
  header.className = "tile-header";

  const title = document.createElement("div");
  title.className = "tile-title";
  title.textContent = story.short_title || story.title || "Untitled story";
  header.appendChild(title);

  const link = document.createElement("a");
  link.href = story.url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = "Read full article";
  link.className = "full-link";

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = `${story.source || "Unknown source"} | ${formatDate(story.published_at)}`;
  header.appendChild(meta);

  tile.appendChild(header);

  const body = document.createElement("div");
  body.className = "tile-body";

  const summary = document.createElement("p");
  summary.className = "summary";
  summary.textContent =
    story.plain_summary ||
    "Plain-language summary not available yet. Open the full article for details.";
  body.appendChild(summary);
  body.appendChild(link);
  tile.appendChild(body);

  return tile;
}

function renderColumn(elementId, stories) {
  const list = document.getElementById(elementId);
  list.innerHTML = "";

  if (!stories || !stories.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No matching stories yet.";
    list.appendChild(empty);
    return;
  }

  stories.forEach((story) => {
    list.appendChild(buildTile(story));
  });
}

async function loadStories() {
  const response = await fetch("data/stories.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load stories (${response.status})`);
  return response.json();
}

async function init() {
  try {
    const data = await loadStories();
    renderColumn("government-list", data.government || []);
    renderColumn("nonprofit-list", data.nonprofit || []);

    const updated = document.getElementById("last-updated");
    updated.textContent = data.updated_at
      ? `Last updated: ${formatDate(data.updated_at)}`
      : "Last updated: unknown";
  } catch (error) {
    const message = `Could not load stories: ${error.message}`;
    renderColumn("government-list", []);
    renderColumn("nonprofit-list", []);
    document.getElementById("last-updated").textContent = message;
  }
}

init();
