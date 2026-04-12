/** Main Research Workbench app - Home and Project Workspace panels. */
import * as api from "./api_client.js";

const mainContent = document.getElementById("main-content");
const navButtons = document.querySelectorAll(".nav-btn");

let currentPanel = "home";
let currentProjectId = null;

function navigateTo(panel, projectId = null) {
    currentPanel = panel;
    currentProjectId = projectId;
    navButtons.forEach(btn => btn.classList.remove("active"));
    document.getElementById(`nav-${panel}`)?.classList.add("active");
    render();
}

async function render() {
    if (currentPanel === "home") {
        try {
            const [projects, feedItems, papers] = await Promise.all([
                api.listProjects(),
                api.getTodayFeed(),
                api.listPapers({ library_status: "library" }),
            ]);
            mainContent.innerHTML = "";
            renderHomePanel(mainContent, { projects, feedItems, papers }, navigateTo);
        } catch (e) {
            mainContent.innerHTML = `<p class="error">Failed to load: ${e.message}</p>`;
        }
    } else if (currentPanel === "projects") {
        try {
            const projects = await api.listProjects();
            mainContent.innerHTML = "";
            renderProjectPanel(mainContent, { projects }, navigateTo);
        } catch (e) {
            mainContent.innerHTML = `<p class="error">Failed to load: ${e.message}</p>`;
        }
    } else if (currentPanel === "canvas") {
        window.location.href = "/";
    }
}

function renderHomePanel(container, { projects, feedItems, papers }, navigateTo) {
    const projectCount = projects.length || 0;
    const activeProjects = projects.filter(p => p.status === "active").length;

    container.innerHTML = `
        <div class="home-panel">
            <section class="top-summary">
                <div class="summary-card">
                    <h3>${projectCount}</h3>
                    <p>Total Projects</p>
                </div>
                <div class="summary-card">
                    <h3>${activeProjects}</h3>
                    <p>Active</p>
                </div>
                <div class="summary-card">
                    <h3>${papers.length}</h3>
                    <p>Papers in Library</p>
                </div>
                <div class="summary-card">
                    <h3>${feedItems.length}</h3>
                    <p>Today's Feed</p>
                </div>
            </section>

            <section class="today-feed">
                <h2>Today Feed</h2>
                ${feedItems.length === 0 ? '<p class="empty-state">No papers in feed yet.</p>' :
                    feedItems.map(item => `
                    <div class="feed-card">
                        <h4>${item.title || "Untitled"}</h4>
                        <p class="feed-meta">${item.authors_text || ""} · ${item.published_at || ""}</p>
                        <p class="feed-abstract">${item.abstract || ""}</p>
                        <div class="feed-actions">
                            <button class="btn-save" data-id="${item.id}">Save to Library</button>
                        </div>
                    </div>
                `).join("")}
            </section>

            <section class="project-focus">
                <h2>Project Focus</h2>
                ${projects.length === 0 ? '<p class="empty-state">No projects yet.</p>' :
                    projects.slice(0, 5).map(p => `
                    <div class="project-card" data-id="${p.id}">
                        <h4>${p.title}</h4>
                        <p>${p.goal || "No goal set"}</p>
                        <span class="status-badge ${p.status}">${p.status}</span>
                    </div>
                `).join("")}
            </section>

            <section class="quick-entry">
                <h2>Quick Entry</h2>
                <button id="btn-new-project" class="btn-primary">+ New Project</button>
            </section>
        </div>
    `;

    document.getElementById("btn-new-project")?.addEventListener("click", async () => {
        const title = prompt("Project title:");
        if (title) {
            await api.createProject({ title, goal: "" });
            navigateTo("projects");
        }
    });

    container.querySelectorAll(".project-card").forEach(card => {
        card.addEventListener("click", () => {
            navigateTo("projects");
        });
    });
}

function renderProjectPanel(container, { projects }, navigateTo) {
    container.innerHTML = `
        <div class="project-panel">
            <div class="project-header">
                <h2>Projects</h2>
                <button id="btn-create-project" class="btn-primary">+ New Project</button>
            </div>

            <div class="project-list">
                ${projects.length === 0 ? '<p class="empty-state">No projects. Create your first project.</p>' :
                    projects.map(p => `
                    <div class="project-item" data-id="${p.id}">
                        <div class="project-info">
                            <h3>${p.title}</h3>
                            <p>${p.goal || "No goal set"}</p>
                            <span class="status-badge ${p.status}">${p.status}</span>
                        </div>
                        <div class="project-actions">
                            <button class="btn-open-canvas" data-id="${p.id}">Open Canvas</button>
                        </div>
                    </div>
                `).join("")}
            </div>
        </div>
    `;

    document.getElementById("btn-create-project")?.addEventListener("click", async () => {
        const title = prompt("Project title:");
        if (title) {
            await api.createProject({ title, goal: "" });
            const projects = await api.listProjects();
            renderProjectPanel(container, { projects }, navigateTo);
        }
    });
}

// Wire up navigation
document.getElementById("nav-home").addEventListener("click", () => navigateTo("home"));
document.getElementById("nav-projects").addEventListener("click", () => navigateTo("projects"));
document.getElementById("nav-assets").addEventListener("click", () => navigateTo("assets"));
document.getElementById("nav-canvas").addEventListener("click", () => navigateTo("canvas"));

// Initial render
render();
