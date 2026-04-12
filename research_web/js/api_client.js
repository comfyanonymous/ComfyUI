/** Research API client - talks to ComfyUI's built-in research routes via aiohttp. */

const API_BASE = "/research";

export async function listProjects() {
    const res = await fetch(`${API_BASE}/projects/`);
    return res.json();
}

export async function createProject(data) {
    const res = await fetch(`${API_BASE}/projects/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
    });
    return res.json();
}

export async function getProject(projectId) {
    const res = await fetch(`${API_BASE}/projects/${projectId}`);
    return res.json();
}

export async function updateProject(projectId, data) {
    const res = await fetch(`${API_BASE}/projects/${projectId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
    });
    return res.json();
}

export async function listIntents(projectId) {
    const res = await fetch(`${API_BASE}/projects/${projectId}/intents`);
    return res.json();
}

export async function createIntent(projectId, data) {
    const res = await fetch(`${API_BASE}/projects/${projectId}/intents`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
    });
    return res.json();
}

export async function updateIntent(intentId, data) {
    const res = await fetch(`${API_BASE}/projects/intents/${intentId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
    });
    return res.json();
}

export async function listPapers(params = {}) {
    const qs = new URLSearchParams(params).toString();
    const res = await fetch(`${API_BASE}/papers/${qs ? "?" + qs : ""}`);
    return res.json();
}

export async function createPaper(data) {
    const res = await fetch(`${API_BASE}/papers/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
    });
    return res.json();
}

export async function updatePaper(paperId, data) {
    const res = await fetch(`${API_BASE}/papers/${paperId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
    });
    return res.json();
}

export async function listClaims(params = {}) {
    const qs = new URLSearchParams(params).toString();
    const res = await fetch(`${API_BASE}/claims/${qs ? "?" + qs : ""}`);
    return res.json();
}

export async function createClaim(data) {
    const res = await fetch(`${API_BASE}/claims/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
    });
    return res.json();
}

export async function updateClaim(claimId, data) {
    const res = await fetch(`${API_BASE}/claims/${claimId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
    });
    return res.json();
}

export async function getTodayFeed() {
    const res = await fetch(`${API_BASE}/feed/today`);
    return res.json();
}

export async function listSources() {
    const res = await fetch(`${API_BASE}/sources/`);
    return res.json();
}

export async function createSource(data) {
    const res = await fetch(`${API_BASE}/sources/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
    });
    return res.json();
}
