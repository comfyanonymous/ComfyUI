/**
 * Progress bar web component.
 */

import { SERVICE as PROMPT_SERVICE, type PromptExecution } from "rgthree/common/prompt_service.js";
import { createElement } from "./utils_dom.js";

/**
 * The progress bar web component.
 */
export class RgthreeProgressBar extends HTMLElement {
  static NAME = "rgthree-progress-bar";

  static create(): RgthreeProgressBar {
    return document.createElement(RgthreeProgressBar.NAME) as RgthreeProgressBar;
  }

  private shadow: ShadowRoot | null = null;
  private progressNodesEl!: HTMLDivElement;
  private progressStepsEl!: HTMLDivElement;
  private progressTextEl!: HTMLSpanElement;

  private currentPromptExecution: PromptExecution | null = null;

  private readonly onProgressUpdateBound = this.onProgressUpdate.bind(this);

  private connected: boolean = false;

  /** The currentNodeId so outside callers can see what we're currently executing against. */
  get currentNodeId() {
    const prompt = this.currentPromptExecution;
    const nodeId = prompt?.errorDetails?.node_id || prompt?.currentlyExecuting?.nodeId;
    return nodeId || null;
  }

  constructor() {
    super();
  }

  private onProgressUpdate(e: CustomEvent<{ queue: number; prompt: PromptExecution }>) {
    if (!this.connected) return;

    const prompt = e.detail.prompt;
    this.currentPromptExecution = prompt;

    if (prompt?.errorDetails) {
      let progressText = `${prompt.errorDetails?.exception_type} ${
        prompt.errorDetails?.node_id || ""
      } ${prompt.errorDetails?.node_type || ""}`;
      this.progressTextEl.innerText = progressText;
      this.progressNodesEl.classList.add("-error");
      this.progressStepsEl.classList.add("-error");
      return;
    }
    if (prompt?.currentlyExecuting) {
      this.progressNodesEl.classList.remove("-error");
      this.progressStepsEl.classList.remove("-error");

      const current = prompt?.currentlyExecuting;

      let progressText = `(${e.detail.queue}) `;

      // Sometimes we may get status updates for a workflow that was already running. In that case
      // we don't know totalNodes.
      if (!prompt.totalNodes) {
        progressText += `??%`;
        this.progressNodesEl.style.width = `0%`;
      } else {
        const percent = (prompt.executedNodeIds.length / prompt.totalNodes) * 100;
        this.progressNodesEl.style.width = `${Math.max(2, percent)}%`;
        // progressText += `Node ${prompt.executedNodeIds.length + 1} of ${prompt.totalNodes || "?"}`;
        progressText += `${Math.round(percent)}%`;
      }

      let nodeLabel = current.nodeLabel?.trim();
      let stepsLabel = "";
      if (current.step != null && current.maxSteps) {
        const percent = (current.step / current.maxSteps) * 100;
        this.progressStepsEl.style.width = `${percent}%`;
        // stepsLabel += `Step ${current.step} of ${current.maxSteps}`;
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
    } else {
      if (e?.detail.queue) {
        this.progressTextEl.innerText = `(${e.detail.queue}) Running... in another tab`;
      } else {
        this.progressTextEl.innerText = "Idle";
      }
      this.progressNodesEl.style.width = `0%`;
      this.progressStepsEl.style.width = `0%`;
    }
  }

  connectedCallback() {
    if (!this.connected) {
      PROMPT_SERVICE.addEventListener(
        "progress-update",
        this.onProgressUpdateBound as EventListener,
      );
      this.connected = true;
    }
    // We were already connected, so we just need to reset.
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
    PROMPT_SERVICE.removeEventListener(
      "progress-update",
      this.onProgressUpdateBound as EventListener,
    );
  }
}

customElements.define(RgthreeProgressBar.NAME, RgthreeProgressBar);
