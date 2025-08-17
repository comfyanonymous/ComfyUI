import { SERVICE as PROMPT_SERVICE } from "../common/prompt_service.js";
import { createElement } from "./utils_dom.js";
export class RgthreeProgressBar extends HTMLElement {
    static create() {
        return document.createElement(RgthreeProgressBar.NAME);
    }
    get currentNodeId() {
        var _a, _b;
        const prompt = this.currentPromptExecution;
        const nodeId = ((_a = prompt === null || prompt === void 0 ? void 0 : prompt.errorDetails) === null || _a === void 0 ? void 0 : _a.node_id) || ((_b = prompt === null || prompt === void 0 ? void 0 : prompt.currentlyExecuting) === null || _b === void 0 ? void 0 : _b.nodeId);
        return nodeId || null;
    }
    constructor() {
        super();
        this.shadow = null;
        this.currentPromptExecution = null;
        this.onProgressUpdateBound = this.onProgressUpdate.bind(this);
        this.connected = false;
    }
    onProgressUpdate(e) {
        var _a, _b, _c, _d;
        if (!this.connected)
            return;
        const prompt = e.detail.prompt;
        this.currentPromptExecution = prompt;
        if (prompt === null || prompt === void 0 ? void 0 : prompt.errorDetails) {
            let progressText = `${(_a = prompt.errorDetails) === null || _a === void 0 ? void 0 : _a.exception_type} ${((_b = prompt.errorDetails) === null || _b === void 0 ? void 0 : _b.node_id) || ""} ${((_c = prompt.errorDetails) === null || _c === void 0 ? void 0 : _c.node_type) || ""}`;
            this.progressTextEl.innerText = progressText;
            this.progressNodesEl.classList.add("-error");
            this.progressStepsEl.classList.add("-error");
            return;
        }
        if (prompt === null || prompt === void 0 ? void 0 : prompt.currentlyExecuting) {
            this.progressNodesEl.classList.remove("-error");
            this.progressStepsEl.classList.remove("-error");
            const current = prompt === null || prompt === void 0 ? void 0 : prompt.currentlyExecuting;
            let progressText = `(${e.detail.queue}) `;
            if (!prompt.totalNodes) {
                progressText += `??%`;
                this.progressNodesEl.style.width = `0%`;
            }
            else {
                const percent = (prompt.executedNodeIds.length / prompt.totalNodes) * 100;
                this.progressNodesEl.style.width = `${Math.max(2, percent)}%`;
                progressText += `${Math.round(percent)}%`;
            }
            let nodeLabel = (_d = current.nodeLabel) === null || _d === void 0 ? void 0 : _d.trim();
            let stepsLabel = "";
            if (current.step != null && current.maxSteps) {
                const percent = (current.step / current.maxSteps) * 100;
                this.progressStepsEl.style.width = `${percent}%`;
                if (current.pass > 1 || current.maxPasses != null) {
                    stepsLabel += `#${current.pass}`;
                    if (current.maxPasses && current.maxPasses > 0) {
                        stepsLabel += `/${current.maxPasses}`;
                    }
                    stepsLabel += ` - `;
                }
                stepsLabel += `${Math.round(percent)}%`;
            }
            if (nodeLabel || stepsLabel) {
                progressText += ` - ${nodeLabel || "???"}${stepsLabel ? ` (${stepsLabel})` : ""}`;
            }
            if (!stepsLabel) {
                this.progressStepsEl.style.width = `0%`;
            }
            this.progressTextEl.innerText = progressText;
        }
        else {
            if (e === null || e === void 0 ? void 0 : e.detail.queue) {
                this.progressTextEl.innerText = `(${e.detail.queue}) Running... in another tab`;
            }
            else {
                this.progressTextEl.innerText = "Idle";
            }
            this.progressNodesEl.style.width = `0%`;
            this.progressStepsEl.style.width = `0%`;
        }
    }
    connectedCallback() {
        if (!this.connected) {
            PROMPT_SERVICE.addEventListener("progress-update", this.onProgressUpdateBound);
            this.connected = true;
        }
        if (this.shadow) {
            this.progressTextEl.innerText = "Idle";
            this.progressNodesEl.style.width = `0%`;
            this.progressStepsEl.style.width = `0%`;
            return;
        }
        this.shadow = this.attachShadow({ mode: "open" });
        const sheet = new CSSStyleSheet();
        sheet.replaceSync(`

      :host {
        position: relative;
        overflow: hidden;
        box-sizing: border-box;
        background: var(--rgthree-progress-bg-color);
        --rgthree-progress-bg-color: rgba(23, 23, 23, 0.9);
        --rgthree-progress-nodes-bg-color: rgb(0, 128, 0);
        --rgthree-progress-steps-bg-color: rgb(0, 128, 0);
        --rgthree-progress-error-bg-color: rgb(128, 0, 0);
        --rgthree-progress-text-color: #fff;
      }
      :host * {
        box-sizing: inherit;
      }

      :host > div.bar {
        background: var(--rgthree-progress-nodes-bg-color);
        position: absolute;
        left: 0;
        top: 0;
        width: 0%;
        height: 50%;
        z-index: 1;
        transition: width 50ms ease-in-out;
      }
      :host > div.bar + div.bar {
        background: var(--rgthree-progress-steps-bg-color);
        top: 50%;
        height: 50%;
        z-index: 2;
      }
      :host > div.bar.-error {
        background: var(--rgthree-progress-error-bg-color);
      }

      :host > .overlay {
        position: absolute;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        z-index: 5;
        background: linear-gradient(to bottom, rgba(255,255,255,0.25), rgba(0,0,0,0.25));
        mix-blend-mode: overlay;
      }

      :host > span {
        position: relative;
        z-index: 4;
        text-align: left;
        font-size: inherit;
        height: 100%;
        font-family: sans-serif;
        text-shadow: 1px 1px 0px #000;
        display: flex;
        flex-direction: row;
        padding: 0 6px;
        align-items: center;
        justify-content: start;
        color: var(--rgthree-progress-text-color);
        text-shadow: black 0px 0px 2px;
      }

      :host > div.bar[style*="width: 0%"]:first-child,
      :host > div.bar[style*="width:0%"]:first-child {
        height: 0%;
      }
      :host > div.bar[style*="width: 0%"]:first-child + div,
      :host > div.bar[style*="width:0%"]:first-child + div {
        bottom: 0%;
      }
    `);
        this.shadow.adoptedStyleSheets = [sheet];
        const overlayEl = createElement(`div.overlay[part="overlay"]`, { parent: this.shadow });
        this.progressNodesEl = createElement(`div.bar[part="progress-nodes"]`, { parent: this.shadow });
        this.progressStepsEl = createElement(`div.bar[part="progress-steps"]`, { parent: this.shadow });
        this.progressTextEl = createElement(`span[part="text"]`, { text: "Idle", parent: this.shadow });
    }
    disconnectedCallback() {
        this.connected = false;
        PROMPT_SERVICE.removeEventListener("progress-update", this.onProgressUpdateBound);
    }
}
RgthreeProgressBar.NAME = "rgthree-progress-bar";
customElements.define(RgthreeProgressBar.NAME, RgthreeProgressBar);
