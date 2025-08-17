import type {
  ComfyApiEventDetailCached,
  ComfyApiEventDetailError,
  ComfyApiEventDetailExecuted,
  ComfyApiEventDetailExecuting,
  ComfyApiEventDetailExecutionStart,
  ComfyApiEventDetailProgress,
  ComfyApiEventDetailStatus,
  ComfyApiFormat,
  ComfyApiPrompt,
} from "typings/comfy.js";
import { api } from "scripts/api.js";
import type { LGraph as TLGraph, LGraphCanvas as TLGraphCanvas } from "@comfyorg/frontend";
import { Resolver, getResolver } from "./shared_utils.js";

/**
 * Wraps general data of a prompt's execution.
 */
export class PromptExecution {
  id: string;
  promptApi: ComfyApiFormat | null = null;
  executedNodeIds: string[] = [];
  totalNodes: number = 0;
  currentlyExecuting: {
    nodeId: string;
    nodeLabel?: string;
    step?: number;
    maxSteps?: number;
    /** The current pass, for nodes with multiple progress passes. */
    pass: number;
    /**
     * The max num of passes. Can be calculated for some nodes, or set to -1 when known there will
     * be multiple passes, but the number cannot be calculated.
     */
    maxPasses?: number;
  } | null = null;
  errorDetails: any | null = null;

  apiPrompt: Resolver<null> = getResolver();

  constructor(id: string) {
    this.id = id;
  }

  /**
   * Sets the prompt and prompt-related data. This can technically come in lazily, like if the web
   * socket fires the 'execution-start' event before we actually get a response back from the
   * initial prompt call.
   */
  setPrompt(prompt: ComfyApiPrompt) {
    this.promptApi = prompt.output;
    this.totalNodes = Object.keys(this.promptApi).length;
    this.apiPrompt.resolve(null);
  }

  getApiNode(nodeId: string | number) {
    return this.promptApi?.[String(nodeId)] || null;
  }

  private getNodeLabel(nodeId: string | number) {
    const apiNode = this.getApiNode(nodeId);
    let label = apiNode?._meta?.title || apiNode?.class_type || undefined;
    if (!label) {
      const graphNode = this.maybeGetComfyGraph()?.getNodeById(Number(nodeId));
      label = graphNode?.title || graphNode?.type || undefined;
    }
    return label;
  }

  /**
   * Updates the execution data depending on the passed data, fed from api events.
   */
  executing(nodeId: string | null, step?: number, maxSteps?: number) {
    if (nodeId == null) {
      // We're done, any left over nodes must be skipped...
      this.currentlyExecuting = null;
      return;
    }
    if (this.currentlyExecuting?.nodeId !== nodeId) {
      if (this.currentlyExecuting != null) {
        this.executedNodeIds.push(nodeId);
      }
      this.currentlyExecuting = { nodeId, nodeLabel: this.getNodeLabel(nodeId), pass: 0 };
      // We'll see if we're known node for multiple passes, that will come in as generic 'progress'
      // updates from the api. If we're known to have multiple passes, then we'll pre-set data to
      // allow the progress bar to handle intial rendering. If we're not, that's OK, the data will
      // be shown with the second pass.
      this.apiPrompt.promise.then(() => {
        // If we execute with a null node id and clear the currently executing, then we can just
        // move on. This seems to only happen with a super-fast execution (like, just seed node
        // and display any for testing).
        if (this.currentlyExecuting == null) {
          return;
        }
        const apiNode = this.getApiNode(nodeId);
        if (!this.currentlyExecuting.nodeLabel) {
          this.currentlyExecuting.nodeLabel = this.getNodeLabel(nodeId);
        }
        if (apiNode?.class_type === "UltimateSDUpscale") {
          // From what I can tell, UltimateSDUpscale, does an initial pass that isn't actually a
          // tile. It seems to always be 4 steps... We'll start our pass at -1, so this prepass is
          // "0" and "1" will start with the first tile. This way, a user knows they have 4 tiles,
          // know this pass counter will go to 4 (and not 5). Also, we cannot calculate maxPasses
          // for 'UltimateSDUpscale' :(
          this.currentlyExecuting.pass--;
          this.currentlyExecuting.maxPasses = -1;
        } else if (apiNode?.class_type === "IterativeImageUpscale") {
          this.currentlyExecuting.maxPasses = (apiNode?.inputs["steps"] as number) ?? -1;
        }
      });
    }
    if (step != null) {
      // If we haven't had any stpes before, or the passes step is lower than the previous, then
      // increase the passes.
      if (!this.currentlyExecuting!.step || step < this.currentlyExecuting!.step) {
        this.currentlyExecuting!.pass!++;
      }
      this.currentlyExecuting!.step = step;
      this.currentlyExecuting!.maxSteps = maxSteps;
    }
  }

  /**
   * If there's an error, we add the details.
   */
  error(details: any) {
    this.errorDetails = details;
  }

  private maybeGetComfyGraph(): TLGraph | null {
    return ((window as any)?.app?.graph as TLGraph) || null;
  }
}

/**
 * A singleton service that wraps the Comfy API and simplifies the event data being fired.
 */
class PromptService extends EventTarget {
  promptsMap: Map<string, PromptExecution> = new Map();
  currentExecution: PromptExecution | null = null;
  lastQueueRemaining = 0;

  constructor(api: any) {
    super();
    const that = this;

    // Patch the queuePrompt method so we can capture new data going through.
    const queuePrompt = api.queuePrompt;
    api.queuePrompt = async function (num: number, prompt: ComfyApiPrompt) {
      let response;
      try {
        response = await queuePrompt.apply(api, [...arguments]);
      } catch (e) {
        const promptExecution = that.getOrMakePrompt("error");
        promptExecution.error({ exception_type: "Unknown." });
        // console.log("ERROR QUEUE PROMPT", response, arguments);
        throw e;
      }
      // console.log("QUEUE PROMPT", response, arguments);
      const promptExecution = that.getOrMakePrompt(response.prompt_id);
      promptExecution.setPrompt(prompt);
      if (!that.currentExecution) {
        that.currentExecution = promptExecution;
      }
      that.promptsMap.set(response.prompt_id, promptExecution);
      that.dispatchEvent(
        new CustomEvent("queue-prompt", {
          detail: {
            prompt: promptExecution,
          },
        }),
      );
      return response;
    };

    api.addEventListener("status", (e: CustomEvent<ComfyApiEventDetailStatus>) => {
      // console.log("status", JSON.stringify(e.detail));
      // Sometimes a status message is fired when the app loades w/o any details.
      if (!e.detail?.exec_info) return;
      this.lastQueueRemaining = e.detail.exec_info.queue_remaining;
      this.dispatchProgressUpdate();
    });

    api.addEventListener("execution_start", (e: CustomEvent<ComfyApiEventDetailExecutionStart>) => {
      // console.log("execution_start", JSON.stringify(e.detail));
      if (!this.promptsMap.has(e.detail.prompt_id)) {
        console.warn("'execution_start' fired before prompt was made.");
      }
      const prompt = this.getOrMakePrompt(e.detail.prompt_id);
      this.currentExecution = prompt;
      this.dispatchProgressUpdate();
    });

    api.addEventListener("executing", (e: CustomEvent<ComfyApiEventDetailExecuting>) => {
      // console.log("executing", JSON.stringify(e.detail));
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

    api.addEventListener("progress", (e: CustomEvent<ComfyApiEventDetailProgress>) => {
      // console.log("progress", JSON.stringify(e.detail));
      if (!this.currentExecution) {
        this.currentExecution = this.getOrMakePrompt(e.detail.prompt_id);
        console.warn("'progress' fired before prompt was made.");
      }
      this.currentExecution.executing(e.detail.node, e.detail.value, e.detail.max);
      this.dispatchProgressUpdate();
    });

    api.addEventListener("execution_cached", (e: CustomEvent<ComfyApiEventDetailCached>) => {
      // console.log("execution_cached", JSON.stringify(e.detail));
      if (!this.currentExecution) {
        this.currentExecution = this.getOrMakePrompt(e.detail.prompt_id);
        console.warn("'execution_cached' fired before prompt was made.");
      }
      for (const cached of e.detail.nodes) {
        this.currentExecution.executing(cached);
      }
      this.dispatchProgressUpdate();
    });

    api.addEventListener("executed", (e: CustomEvent<ComfyApiEventDetailExecuted>) => {
      // console.log("executed", JSON.stringify(e.detail));
      if (!this.currentExecution) {
        this.currentExecution = this.getOrMakePrompt(e.detail.prompt_id);
        console.warn("'executed' fired before prompt was made.");
      }
    });

    api.addEventListener("execution_error", (e: CustomEvent<ComfyApiEventDetailError>) => {
      // console.log("execution_error", e.detail);
      if (!this.currentExecution) {
        this.currentExecution = this.getOrMakePrompt(e.detail.prompt_id);
        console.warn("'execution_error' fired before prompt was made.");
      }
      this.currentExecution?.error(e.detail);
      this.dispatchProgressUpdate();
    });
  }

  /** A helper method, since we extend/override api.queuePrompt above anyway. */
  async queuePrompt(prompt: ComfyApiPrompt) {
    return await api.queuePrompt(-1, prompt);
  }

  dispatchProgressUpdate() {
    this.dispatchEvent(
      new CustomEvent("progress-update", {
        detail: {
          queue: this.lastQueueRemaining,
          prompt: this.currentExecution,
        },
      }),
    );
  }

  getOrMakePrompt(id: string) {
    let prompt = this.promptsMap.get(id);
    if (!prompt) {
      prompt = new PromptExecution(id);
      this.promptsMap.set(id, prompt);
    }
    return prompt;
  }
}

export const SERVICE = new PromptService(api);
