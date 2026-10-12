// "Model Compatibility" sidebar tab: will a model run on this machine? Hugging
// Face repos are judged from their file list alone (/model_compat/hf, /model_compat/org),
// nothing is downloaded; installed models from their size on disk (/model_compat/local).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STYLE = `
.mc { --bg:#0f1115; --panel:#161a21; --raised:#1d222b; --line:#2a303b; --text:#e6e9ef; --muted:#8d96a5;
  --faint:#5c6573; --accent:#d97757; --accent-hover:#e48a6c; --ok:#3fb950; --warn:#e3b341; --err:#f85149;
  --mono:"JetBrains Mono","Fira Code",ui-monospace,Menlo,Consolas,monospace;
  display:flex; flex-direction:column; height:100%; background:var(--bg); color:var(--text);
  font:13px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
.mc button { font:inherit; cursor:pointer; border:none; }
.mc button:disabled { opacity:.5; cursor:default; }
.mc button:focus-visible, .mc input:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.mc-head { padding:12px 12px 10px; border-bottom:1px solid var(--line); }
.mc-title { font-weight:600; font-size:14px; margin-bottom:10px; }
.mc-hw { display:flex; gap:10px; align-items:center; padding:10px; border:1px solid var(--line); border-radius:10px; background:var(--panel); }
.mc-hw > .pi { font-size:20px; color:var(--accent); }
.mc-hw-name { font-weight:600; font-size:12.5px; }
.mc-chips { display:flex; flex-wrap:wrap; gap:4px; margin-top:4px; }
.mc-chip { font-size:11px; padding:1px 7px; border-radius:999px; background:var(--raised); color:var(--muted); white-space:nowrap; }
.mc-chip.yes { color:#7ee787; } .mc-chip.no { color:var(--faint); text-decoration:line-through; }
.mc-seg { display:flex; margin-top:10px; padding:3px; border-radius:9px; background:var(--panel); border:1px solid var(--line); }
.mc-seg button { flex:1; padding:5px 8px; border-radius:7px; background:transparent; color:var(--muted); font-weight:500; }
.mc-seg button.on { background:var(--raised); color:var(--text); box-shadow:0 1px 2px rgba(0,0,0,.3); }
.mc-search { display:flex; gap:6px; padding:10px 12px; border-bottom:1px solid var(--line); }
.mc-search[hidden] { display:none; }
.mc-field { flex:1; min-width:0; display:flex; align-items:center; gap:7px; padding:0 10px; border:1px solid var(--line);
  border-radius:9px; background:var(--panel); }
.mc-field:focus-within { border-color:#4a5262; }
.mc-field .pi { color:var(--faint); font-size:12px; }
.mc-field input { flex:1; min-width:0; padding:7px 0; background:transparent; border:none; outline:none !important; color:var(--text); font:inherit; }
.mc-btn { display:inline-flex; align-items:center; gap:6px; padding:7px 13px; border-radius:9px; font-weight:500; white-space:nowrap;
  background:var(--accent); color:#fff; transition:background .15s, transform .05s; }
.mc-btn:hover:not(:disabled) { background:var(--accent-hover); }
.mc-btn:active:not(:disabled) { transform:scale(.97); }
.mc-body { flex:1; overflow-y:auto; padding:12px; }
.mc-summary { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:12px; }
.mc-summary a { color:var(--muted); font-size:12px; text-decoration:none; margin-right:auto; display:flex; align-items:center; gap:5px; }
.mc-summary a:hover { color:var(--text); }
.mc-group { font-size:11px; font-weight:600; letter-spacing:.04em; text-transform:uppercase; color:var(--faint); margin:14px 0 6px; }
.mc-group:first-of-type { margin-top:0; }
.mc-card { padding:9px 10px; margin-bottom:6px; border:1px solid var(--line); border-radius:10px; background:var(--panel); }
.mc-card.rec { border-color:rgba(63,185,80,.55); }
.mc-row { display:flex; align-items:center; gap:8px; }
.mc-name { flex:1; min-width:0; font:12px var(--mono); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.mc-badge { display:inline-flex; align-items:center; gap:4px; font-size:11px; font-weight:600; padding:2px 8px; border-radius:999px; white-space:nowrap; }
.mc-badge.fits { color:#7ee787; background:rgba(63,185,80,.14); }
.mc-badge.offload { color:#f2cc60; background:rgba(227,179,65,.14); }
.mc-badge.no { color:#ffa198; background:rgba(248,81,73,.14); }
.mc-meta { display:flex; align-items:center; gap:6px; margin-top:6px; font-size:11.5px; color:var(--muted); }
.mc-tag { font:10.5px var(--mono); padding:0 5px; border-radius:4px; background:var(--raised); color:var(--text); }
.mc-tag.rec { background:rgba(63,185,80,.18); color:#7ee787; font-family:inherit; font-weight:600; }
.mc-bar { position:relative; height:6px; margin-top:7px; border-radius:3px; background:var(--raised); overflow:hidden; }
.mc-bar i { position:absolute; inset:0 auto 0 0; border-radius:3px; }
.mc-bar i.fits { background:var(--ok); } .mc-bar i.offload { background:var(--warn); } .mc-bar i.no { background:var(--err); }
.mc-notes { margin:6px 0 0; padding-left:16px; font-size:11.5px; color:var(--muted); }
.mc-empty { display:flex; flex-direction:column; align-items:center; gap:8px; text-align:center; color:var(--muted); padding:30px 12px; }
.mc-empty .pi { font-size:26px; color:var(--faint); }
.mc-error { color:#ffa198; background:rgba(248,81,73,.08); border:1px solid rgba(248,81,73,.3); border-radius:10px; padding:10px; }
.mc-spin { width:16px; height:16px; border:2px solid var(--line); border-top-color:var(--accent); border-radius:50%; animation:mc-spin .8s linear infinite; }
.mc-legend { padding:10px 12px; border-top:1px solid var(--line); font-size:11px; color:var(--faint); display:flex; flex-wrap:wrap; gap:4px 12px; }
.mc-legend b { font-weight:600; }
.mc-filters { display:flex; gap:5px; flex-wrap:wrap; margin-bottom:10px; }
.mc-filter { padding:4px 10px; border-radius:999px; background:var(--panel) !important; border:1px solid var(--line) !important;
  color:var(--muted); font-size:12px !important; }
.mc-filter.on { background:var(--text) !important; color:var(--bg); border-color:var(--text) !important; font-weight:600; }
.mc-repo { width:100%; text-align:left; display:block; padding:9px 10px; margin-bottom:6px; border:1px solid var(--line) !important;
  border-radius:10px; background:var(--panel); color:var(--text); transition:background .15s, border-color .15s; }
.mc-repo:hover { background:var(--raised); border-color:#3a4250 !important; }
.mc-repo .mc-name { font-family:inherit; font-weight:600; font-size:12.5px; }
.mc-repo .mc-best { margin-top:4px; font-size:11.5px; color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.mc-repo .mc-best code { font:11px var(--mono); color:var(--text); }
.mc-counts { display:flex; gap:8px; margin-top:5px; font-size:11px; }
.mc-counts .fits { color:#7ee787; } .mc-counts .offload { color:#f2cc60; } .mc-counts .no { color:#ffa198; }
.mc-back { display:inline-flex; align-items:center; gap:6px; margin-bottom:10px; padding:4px 10px 4px 6px; border-radius:8px;
  background:var(--raised); color:var(--text); font-size:12px; }
.mc-back:hover { background:var(--line); }
@keyframes mc-spin { to { transform:rotate(360deg); } }
`;

const LEVELS = {
  fits: { icon: "pi-check-circle", text: "Fits on GPU" },
  offload: { icon: "pi-exclamation-triangle", text: "Needs offload" },
  no: { icon: "pi-times-circle", text: "Too large" },
};

const COUNT_LABELS = { fits: "fit", offload: "offload", no: "too large" };
const ROLES = {
  diffusion: "Diffusion", checkpoint: "Checkpoint", model: "Model", text_encoder: "Text encoder", vae: "VAE",
  lora: "LoRA", controlnet: "ControlNet", upscale: "Upscale", embedding: "Embedding",
};

function esc(text) {
  return String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

function gb(bytes) {
  const value = bytes / 1024 ** 3;
  return value >= 0.1 ? `${value.toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 1024 ** 2))} MB`;
}

function badge(level) {
  return `<span class="mc-badge ${level}"><i class="pi ${LEVELS[level].icon}"></i>${LEVELS[level].text}</span>`;
}

function renderHardware(hw) {
  const chip = (ok, label) => `<span class="mc-chip ${ok ? "yes" : "no"}">${label}</span>`;
  const gpu = hw.type === "cuda";
  return `<div class="mc-hw"><i class="pi pi-microchip"></i><div>
    <div class="mc-hw-name">${esc(hw.name.replace(/^cuda:\d+\s*/, ""))}</div>
    <div class="mc-chips">
      ${gpu ? `<span class="mc-chip">${gb(hw.vram_total)} VRAM</span>` : `<span class="mc-chip">No GPU</span>`}
      <span class="mc-chip">${gb(hw.ram_total)} RAM</span>
      ${gpu ? chip(hw.supports.bf16, "bf16") + chip(hw.supports.fp8, "fp8") + chip(hw.supports.fp4, "fp4") : ""}
    </div></div></div>`;
}

function renderCard(file, hw) {
  const share = Math.min(100, (file.size / hw.vram_total) * 100);
  const notes = file.notes.length ? `<ul class="mc-notes">${file.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : "";
  return `<div class="mc-card${file.recommended ? " rec" : ""}">
    <div class="mc-row"><span class="mc-name" title="${esc(file.path)}">${esc(file.name)}</span>${badge(file.level)}</div>
    <div class="mc-meta">
      <span>${gb(file.size)}</span>
      ${file.precision ? `<span class="mc-tag">${esc(file.precision)}</span>` : ""}
      ${file.recommended ? `<span class="mc-tag rec">★ Recommended</span>` : ""}
      <span style="margin-left:auto">${hw.type === "cuda" ? `${Math.round((file.size / hw.vram_total) * 100)}% of VRAM` : ""}</span>
    </div>
    <div class="mc-bar"><i class="${file.level}" style="width:${share}%"></i></div>
    ${notes}
  </div>`;
}

function renderFiles(files, hw, groupKey) {
  if (!files.length) return `<div class="mc-empty"><i class="pi pi-inbox"></i>No model files found.</div>`;
  const count = (level) => files.filter((f) => f.level === level).length;
  let html = `<div class="mc-summary">${["fits", "offload", "no"]
    .filter(count)
    .map((level) => `<span class="mc-badge ${level}">${count(level)} · ${LEVELS[level].text}</span>`)
    .join("")}</div>`;
  let group = null;
  for (const file of files) {
    if (file[groupKey] !== group) {
      group = file[groupKey];
      html += `<div class="mc-group">${esc(ROLES[group] ?? group)}</div>`;
    }
    html += renderCard(file, hw);
  }
  return html;
}

function createPanel(el) {
  if (!document.getElementById("mc-style")) {
    const style = document.createElement("style");
    style.id = "mc-style";
    document.head.append(style);
  }
  document.getElementById("mc-style").textContent = STYLE;

  el.innerHTML = `
    <div class="mc">
      <div class="mc-head">
        <div class="mc-title">Model Compatibility</div>
        <div class="mc-hwbox"><div class="mc-hw"><div class="mc-spin"></div><span>Reading hardware…</span></div></div>
        <div class="mc-seg"><button class="on" data-tab="org">Comfy-Org</button><button data-tab="hf">Hugging Face</button><button data-tab="local">Installed</button></div>
      </div>
      <form class="mc-search" hidden>
        <label class="mc-field"><i class="pi pi-search"></i><input placeholder="Repo id or link: Comfy-Org/z_image_turbo" spellcheck="false"></label>
        <button class="mc-btn" type="submit">Check</button>
      </form>
      <div class="mc-body"><div class="mc-empty"><i class="pi pi-gauge"></i>
        Enter a Hugging Face repo. Nothing is downloaded: only file sizes are read and compared with this machine.</div></div>
      <div class="mc-legend">
        <span><b style="color:#7ee787">Fits</b>: whole on the GPU, full speed</span>
        <span><b style="color:#f2cc60">Offload</b>: partly in RAM, slower</span>
        <span><b style="color:#ffa198">Too large</b>: will not run</span>
        <span>Assumes ~1.5 GB headroom for image generation; video models need more.</span>
      </div>
    </div>`;
  const hwBox = el.querySelector(".mc-hwbox");
  const body = el.querySelector(".mc-body");
  const form = el.querySelector(".mc-search");
  const input = form.querySelector("input");
  const submit = form.querySelector("button");
  const tabs = el.querySelectorAll("[data-tab]");

  const loading = (text) => `<div class="mc-empty"><div class="mc-spin"></div>${esc(text)}</div>`;
  const failure = (text) => `<div class="mc-error">${esc(text)}</div>`;

  async function getJson(path) {
    const response = await api.fetchApi(path);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error ?? `Server error (${response.status})`);
    return data;
  }

  let orgRepos = null;
  let orgFilter = "all";

  function renderOrg() {
    const shown = orgRepos.filter((r) => orgFilter === "all" || r.level === orgFilter);
    const count = (level) => orgRepos.filter((r) => r.level === level).length;
    const filter = (key, label) => `<button class="mc-filter${orgFilter === key ? " on" : ""}" data-filter="${key}">${label}</button>`;
    body.innerHTML = `<div class="mc-filters">${filter("all", `All ${orgRepos.length}`)}${filter("fits", `Fits ${count("fits")}`)}${filter(
      "offload",
      `Offload ${count("offload")}`,
    )}${filter("no", `Too large ${count("no")}`)}</div>${shown
      .map((r) => {
        const name = r.repo.split("/")[1];
        if (!r.best) return `<div class="mc-repo"><div class="mc-row"><span class="mc-name">${esc(name)}</span></div>
          <div class="mc-best">${r.error ? "Could not be read" : "No model files"}</div></div>`;
        const counts = ["fits", "offload", "no"]
          .filter((l) => r.counts[l])
          .map((l) => `<span class="${l}">● ${r.counts[l]} ${COUNT_LABELS[l]}</span>`)
          .join("");
        const lead = r.level === "fits" ? "Best at full speed" : r.level === "offload" ? "Even the smallest needs offload" : "Smallest";
        return `<button class="mc-repo" data-repo="${esc(r.repo)}">
          <div class="mc-row"><span class="mc-name">${esc(name)}</span>${badge(r.level)}</div>
          <div class="mc-best">${lead} · ${gb(r.best.size)}${r.best.precision ? ` · ${esc(r.best.precision)}` : ""} · ${esc(ROLES[r.best.role] ?? r.best.role)}</div>
          <div class="mc-best"><code title="${esc(r.best.name)}">${esc(r.best.name)}</code></div>
          <div class="mc-counts">${counts}</div></button>`;
      })
      .join("")}`;
  }

  async function showOrg() {
    if (!orgRepos) {
      body.innerHTML = loading("Scanning Comfy-Org repos (takes a few seconds the first time)…");
      try {
        const data = await getJson("/model_compat/org?name=Comfy-Org");
        hwBox.innerHTML = renderHardware(data.hardware);
        orgRepos = data.repos;
      } catch (error) {
        body.innerHTML = failure(error.message);
        return;
      }
    }
    renderOrg();
  }

  async function checkRepo(fromOrg = false) {
    const repo = input.value.trim();
    if (!repo) return input.focus();
    submit.disabled = true;
    body.innerHTML = loading("Reading the file list from Hugging Face…");
    try {
      const data = await getJson(`/model_compat/hf?repo=${encodeURIComponent(repo)}`);
      hwBox.innerHTML = renderHardware(data.hardware);
      const back = fromOrg ? `<button class="mc-back" data-back><i class="pi pi-arrow-left"></i>Comfy-Org list</button>` : "";
      body.innerHTML = back + renderFiles(data.files, data.hardware, "role").replace(
        `<div class="mc-summary">`,
        `<div class="mc-summary"><a href="https://huggingface.co/${esc(data.repo)}" target="_blank" rel="noopener"><i class="pi pi-external-link"></i>${esc(data.repo)}</a>`,
      );
    } catch (error) {
      body.innerHTML = failure(error.message);
    } finally {
      submit.disabled = false;
    }
  }

  async function showLocal() {
    body.innerHTML = loading("Scanning installed models…");
    try {
      const data = await getJson("/model_compat/local");
      body.innerHTML = renderFiles(data.files, data.hardware, "folder");
    } catch (error) {
      body.innerHTML = failure(error.message);
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    checkRepo();
  });
  function selectTab(name) {
    tabs.forEach((t) => t.classList.toggle("on", t.dataset.tab === name));
    form.hidden = name !== "hf";
    if (name === "org") showOrg();
    else if (name === "local") showLocal();
    else body.innerHTML = `<div class="mc-empty"><i class="pi pi-gauge"></i>Enter a Hugging Face repo.</div>`;
  }
  tabs.forEach((tab) => tab.addEventListener("click", () => selectTab(tab.dataset.tab)));

  body.addEventListener("click", (e) => {
    const filter = e.target.closest("[data-filter]");
    if (filter) {
      orgFilter = filter.dataset.filter;
      return renderOrg();
    }
    const repo = e.target.closest("[data-repo]");
    if (repo) {
      input.value = repo.dataset.repo;
      return checkRepo(true);
    }
    if (e.target.closest("[data-back]")) showOrg();
  });
  showOrg();

  getJson("/model_compat/hardware")
    .then((hw) => (hwBox.innerHTML = renderHardware(hw)))
    .catch((error) => (hwBox.innerHTML = failure(error.message)));
}

app.registerExtension({
  name: "Comfy.ModelCompat",
  setup() {
    app.extensionManager.registerSidebarTab({
      id: "model-compat",
      icon: "pi pi-gauge",
      title: "Model Compatibility",
      tooltip: "Which models run on this machine?",
      type: "custom",
      render: createPanel,
    });
  },
});
