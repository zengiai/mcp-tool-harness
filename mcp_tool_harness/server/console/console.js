const state = { tab: "tool" };

const titles = {
  tool: ["Tool", "Current instance tool registry"],
  chain: ["Chain", "Tool call chains derived from audit events"],
  metrics: ["Metrics", "Tool call metric events and latency summary"],
};

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => selectTab(button.dataset.tab));
});

document.getElementById("refresh").addEventListener("click", () => load());

function selectTab(tab) {
  state.tab = tab;
  document.querySelectorAll(".tab").forEach((button) => {
    button.setAttribute("aria-selected", String(button.dataset.tab === tab));
  });
  document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
  document.getElementById(`view-${tab}`).classList.add("active");
  document.getElementById("title").textContent = titles[tab][0];
  document.getElementById("subtitle").textContent = titles[tab][1];
  load();
}

async function load() {
  try {
    if (state.tab === "tool") return await loadTools();
    if (state.tab === "chain") return await loadChains();
    return await loadMetrics();
  } catch (error) {
    renderError(error);
    return undefined;
  }
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

async function loadTools() {
  const data = await getJson("/console/api/tools");
  const rows = data.tools.map((tool) => (
    `<tr><td><code>${esc(tool.name)}</code></td><td>${esc(tool.description)}</td><td>${esc((tool.tags || []).join(", "))}</td><td>${esc(tool.timeout_ms || "")}</td></tr>`
  )).join("");
  document.getElementById("tools").innerHTML = `<table><thead><tr><th>Name</th><th>Description</th><th>Tags</th><th>Timeout ms</th></tr></thead><tbody>${rows || '<tr><td colspan="4" class="muted">No tools registered.</td></tr>'}</tbody></table>`;
}

async function loadChains() {
  const data = await getJson("/console/api/chains");
  const rows = data.chains.map((chain) => (
    `<tr class="clickable" data-id="${esc(chain.chain_id)}"><td><code>${esc(chain.chain_id)}</code></td><td>${esc(chain.event_count)}</td><td>${esc((chain.tools || []).join(", "))}</td><td>${statusBadges(chain.status_counts)}</td></tr>`
  )).join("");
  document.getElementById("chains").innerHTML = `<table><thead><tr><th>Chain</th><th>Events</th><th>Tools</th><th>Status</th></tr></thead><tbody>${rows || '<tr><td colspan="4" class="muted">No chains found.</td></tr>'}</tbody></table>`;
  document.querySelectorAll("#chains tr[data-id]").forEach((row) => {
    row.addEventListener("click", () => loadChainDetail(row.dataset.id));
  });
}

async function loadChainDetail(chainId) {
  const data = await getJson(`/console/api/chains/${encodeURIComponent(chainId)}`);
  document.getElementById("chain-detail").textContent = JSON.stringify(data, null, 2);
}

async function loadMetrics() {
  const data = await getJson("/console/api/metrics");
  const summary = data.summary || {};
  const latency = summary.latency_ms || {};
  document.getElementById("metric-summary").innerHTML = [
    metricBox("Total Calls", summary.total_calls || 0),
    metricBox("Avg Latency", latency.avg == null ? "-" : `${latency.avg.toFixed(1)} ms`),
    metricBox("Max Latency", latency.max == null ? "-" : `${latency.max.toFixed(1)} ms`),
    metricBox("Events", data.count || 0),
  ].join("");
  const rows = (summary.by_tool || []).map((tool) => (
    `<tr><td><code>${esc(tool.tool_name)}</code></td><td>${esc(tool.count)}</td><td>${statusBadges(tool.status_counts)}</td><td>${esc(tool.latency_ms.avg == null ? "-" : tool.latency_ms.avg.toFixed(1))}</td></tr>`
  )).join("");
  document.getElementById("metrics-table").innerHTML = `<table><thead><tr><th>Tool</th><th>Calls</th><th>Status</th><th>Avg latency ms</th></tr></thead><tbody>${rows || '<tr><td colspan="4" class="muted">No metrics found.</td></tr>'}</tbody></table>`;
}

function metricBox(label, value) {
  return `<div class="metric"><div class="muted">${esc(label)}</div><div class="value">${esc(value)}</div></div>`;
}

function statusBadges(counts) {
  return Object.entries(counts || {}).map(([key, value]) => (
    `<span class="status ${esc(key)}">${esc(key)} ${esc(value)}</span>`
  )).join(" ");
}

function renderError(error) {
  const target = state.tab === "tool"
    ? document.getElementById("tools")
    : state.tab === "chain"
      ? document.getElementById("chains")
      : document.getElementById("metrics-table");
  target.innerHTML = `<div class="error">${esc(error.message || error)}</div>`;
}

load();
