import { api } from "../../scripts/api.js";
import { getResolver } from "./shared_utils.js";
export class PromptExecution {
    constructor(id) {
        this.promptApi = null;
        this.executedNodeIds = [];
        this.totalNodes = 0;
        this.currentlyExecuting = null;
        this.errorDetails = null;
        this.apiPrompt = getResolver();
        this.id = id;
    }
    setPrompt(prompt) {
        this.promptApi = prompt.output;
        this.totalNodes = Object.keys(this.promptApi).length;
        this.apiPrompt.resolve(null);
    }
    getApiNode(nodeId) {
        var _a;
        return ((_a = this.promptApi) === null || _a === void 0 ? void 0 : _a[String(nodeId)]) || null;
    }
    getNodeLabel(nodeId) {
        var _a, _b;
        const apiNode = this.getApiNode(nodeId);
        let label = ((_a = apiNode === null || apiNode === void 0 ? void 0 : apiNode._meta) === null || _a === void 0 ? void 0 : _a.title) || (apiNode === null || apiNode === void 0 ? void 0 : apiNode.class_type) || undefined;
        if (!label) {
            const graphNode = (_b = this.maybeGetComfyGraph()) === null || _b === void 0 ? void 0 : _b.getNodeById(Number(nodeId));
            label = (graphNode === null || graphNode === void 0 ? void 0 : graphNode.title) || (graphNode === null || graphNode === void 0 ? void 0 : graphNode.type) || undefined;
        }
        return label;
    }
    executing(nodeId, step, maxSteps) {
        var _a;
        if (nodeId == null) {
            this.currentlyExecuting = null;
            return;
        }
        if (((_a = this.currentlyExecuting) === null || _a === void 0 ? void 0 : _a.nodeId) !== nodeId) {
            if (this.currentlyExecuting != null) {
                this.executedNodeIds.push(nodeId);
            }
            this.currentlyExecuting = { nodeId, nodeLabel: this.getNodeLabel(nodeId), pass: 0 };
            this.apiPrompt.promise.then(() => {
                var _a;
                if (this.currentlyExecuting == null) {
                    return;
                }
                const apiNode = this.getApiNode(nodeId);
                if (!this.currentlyExecuting.nodeLabel) {
                    this.currentlyExecuting.nodeLabel = this.getNodeLabel(nodeId);
                }
                if ((apiNode === null || apiNode === void 0 ? void 0 : apiNode.class_type) === "UltimateSDUpscale") {
                    this.currentlyExecuting.pass--;
                    this.currentlyExecuting.maxPasses = -1;
                }
                else if ((apiNode === null || apiNode === void 0 ? void 0 : apiNode.class_type) === "IterativeImageUpscale") {
                    this.currentlyExecuting.maxPasses = (_a = apiNode === null || apiNode === void 0 ? void 0 : apiNode.inputs["steps"]) !== null && _a !== void 0 ? _a : -1;
                }
            });
        }
        if (step != null) {
            if (!this.currentlyExecuting.step || step < this.currentlyExecuting.step) {
                this.currentlyExecuting.pass++;
            }
            this.currentlyExecuting.step = step;
            this.currentlyExecuting.maxSteps = maxSteps;
        }
    }
    error(details) {
        this.errorDetails = details;
    }
    maybeGetComfyGraph() {
        var _a;
        return ((_a = window === null || window === void 0 ? void 0 : window.app) === null || _a === void 0 ? void 0 : _a.graph) || null;
    }
}
class PromptService extends EventTarget {
    constructor(api) {
        super();
        this.promptsMap = new Map();
        this.currentExecution = null;
        this.lastQueueRemaining = 0;
        const that = this;
        const queuePrompt = api.queuePrompt;
        api.queuePrompt = async function (num, prompt) {
            let response;
            try {
                response = await queuePrompt.apply(api, [...arguments]);
            }
            catch (e) {
                const promptExecution = that.getOrMakePrompt("error");
                promptExecution.error({ exception_type: "Unknown." });
                throw e;
            }
            const promptExecution = that.getOrMakePrompt(response.prompt_id);
            promptExecution.setPrompt(prompt);
            if (!that.currentExecution) {
                that.currentExecution = promptExecution;
            }
            that.promptsMap.set(response.prompt_id, promptExecution);
            that.dispatchEvent(new CustomEvent("queue-prompt", {
                detail: {
                    prompt: promptExecution,
                },
            }));
            return response;
        };
        api.addEventListener("status", (e) => {
            var _a;
            if (!((_a = e.detail) === null || _a === void 0 ? void 0 : _a.exec_info))
                return;
            this.lastQueueRemaining = e.detail.exec_info.queue_remaining;
            this.dispatchProgressUpdate();
        });
        api.addEventListener("execution_start", (e) => {
            if (!this.promptsMap.has(e.detail.prompt_id)) {
                console.warn("'execution_start' fired before prompt was made.");
            }
            const prompt = this.getOrMakePrompt(e.detail.prompt_id);
            this.currentExecution = prompt;
            this.dispatchProgressUpdate();
        });
        api.addEventListener("executing", (e) => {
            if (!this.currentExecution) {
                this.currentExecution = this.getOrMakePrompt("unknown");
                console.warn("'executing' fired before prompt was made.");
            }
            this.currentExecution.executing(e.detail);
            this.dispatchProgressUpdate();
            if (e.detail == null) {
                this.currentExecution = null;
            }
        });
        api.addEventListener("progress", (e) => {
            if (!this.currentExecution) {
                this.currentExecution = this.getOrMakePrompt(e.detail.prompt_id);
                console.warn("'progress' fired before prompt was made.");
            }
            this.currentExecution.executing(e.detail.node, e.detail.value, e.detail.max);
            this.dispatchProgressUpdate();
        });
        api.addEventListener("execution_cached", (e) => {
            if (!this.currentExecution) {
                this.currentExecution = this.getOrMakePrompt(e.detail.prompt_id);
                console.warn("'execution_cached' fired before prompt was made.");
            }
            for (const cached of e.detail.nodes) {
                this.currentExecution.executing(cached);
            }
            this.dispatchProgressUpdate();
        });
        api.addEventListener("executed", (e) => {
            if (!this.currentExecution) {
                this.currentExecution = this.getOrMakePrompt(e.detail.prompt_id);
                console.warn("'executed' fired before prompt was made.");
            }
        });
        api.addEventListener("execution_error", (e) => {
            var _a;
            if (!this.currentExecution) {
                this.currentExecution = this.getOrMakePrompt(e.detail.prompt_id);
                console.warn("'execution_error' fired before prompt was made.");
            }
            (_a = this.currentExecution) === null || _a === void 0 ? void 0 : _a.error(e.detail);
            this.dispatchProgressUpdate();
        });
    }
    async queuePrompt(prompt) {
        return await api.queuePrompt(-1, prompt);
    }
    dispatchProgressUpdate() {
        this.dispatchEvent(new CustomEvent("progress-update", {
            detail: {
                queue: this.lastQueueRemaining,
                prompt: this.currentExecution,
            },
        }));
    }
    getOrMakePrompt(id) {
        let prompt = this.promptsMap.get(id);
        if (!prompt) {
            prompt = new PromptExecution(id);
            this.promptsMap.set(id, prompt);
        }
        return prompt;
    }
}
export const SERVICE = new PromptService(api);
