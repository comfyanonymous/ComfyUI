import type {
  IContextMenuOptions,
  ContextMenu,
  LGraphNode as TLGraphNode,
  IWidget,
  LGraphCanvas,
  IContextMenuValue,
  LGraphNodeConstructor,
  ISerialisedNode,
  IButtonWidget,
} from "@comfyorg/frontend";
import type {ComfyNodeDef, ComfyApiPrompt} from "typings/comfy.js";

import {app} from "scripts/app.js";
import {ComfyWidgets} from "scripts/widgets.js";
import {RgthreeBaseServerNode} from "./base_node.js";
import {rgthree} from "./rgthree.js";
import {addConnectionLayoutSupport} from "./utils.js";
import {NodeTypesString} from "./constants.js";

const LAST_SEED_BUTTON_LABEL = "♻️ (Use Last Queued Seed)";

const SPECIAL_SEED_RANDOM = -1;
const SPECIAL_SEED_INCREMENT = -2;
const SPECIAL_SEED_DECREMENT = -3;
const SPECIAL_SEEDS = [SPECIAL_SEED_RANDOM, SPECIAL_SEED_INCREMENT, SPECIAL_SEED_DECREMENT];

interface SeedSerializedCtx {
  inputSeed?: number;
  seedUsed?: number;
}

class RgthreeSeed extends RgthreeBaseServerNode {
  static override title = NodeTypesString.SEED;
  static override type = NodeTypesString.SEED;
  static comfyClass = NodeTypesString.SEED;

  override serialize_widgets = true;

  private logger = rgthree.newLogSession(`[Seed]`);

  static override exposedActions = ["Randomize Each Time", "Use Last Queued Seed"];

  lastSeed?: number = undefined;
  serializedCtx: SeedSerializedCtx = {};
  seedWidget!: IWidget;
  lastSeedButton!: IWidget;
  lastSeedValue: IWidget | null = null;

  randMax = 1125899906842624;
  // We can have a full range of seeds, including negative. But, for the randomRange we'll
  // only generate positives, since that's what folks assume.
  // const min = Math.max(-1125899906842624, this.seedWidget.options.min);
  randMin = 0;
  randomRange = 1125899906842624;

  private handleApiHijackingBound = this.handleApiHijacking.bind(this);

  constructor(title = RgthreeSeed.title) {
    super(title);

    rgthree.addEventListener(
      "comfy-api-queue-prompt-before",
      this.handleApiHijackingBound as EventListener,
    );
  }

  override onRemoved() {
    rgthree.addEventListener(
      "comfy-api-queue-prompt-before",
      this.handleApiHijackingBound as EventListener,
    );
  }

  override configure(info: ISerialisedNode): void {
    super.configure(info);
    if (this.properties?.["showLastSeed"]) {
      this.addLastSeedValue();
    }
  }

  override async handleAction(action: string) {
    if (action === "Randomize Each Time") {
      this.seedWidget.value = SPECIAL_SEED_RANDOM;
    } else if (action === "Use Last Queued Seed") {
      this.seedWidget.value = this.lastSeed != null ? this.lastSeed : this.seedWidget.value;
      this.lastSeedButton.name = LAST_SEED_BUTTON_LABEL;
      this.lastSeedButton.disabled = true;
    }
  }

  override onNodeCreated() {
    super.onNodeCreated?.();
    // Grab the already available widgets, and remove the built-in control_after_generate
    for (const [i, w] of this.widgets.entries()) {
      if (w.name === "seed") {
        this.seedWidget = w; // as ComfyWidget;
        this.seedWidget.value = SPECIAL_SEED_RANDOM;
      } else if (w.name === "control_after_generate") {
        this.widgets.splice(i, 1);
      }
    }

    // Update random values in case seed comes down with different options.
    let step = this.seedWidget.options.step || 1;
    this.randMax = Math.min(1125899906842624, this.seedWidget.options.max ?? 0);
    // We can have a full range of seeds, including negative. But, for the randomRange we'll
    // only generate positives, since that's what folks assume.
    this.randMin = Math.max(0, this.seedWidget.options.min ?? 0);
    this.randomRange = (this.randMax - Math.max(0, this.randMin)) / (step / 10);

    this.addWidget(
      "button",
      "🎲 Randomize Each Time",
      "",
      () => {
        this.seedWidget.value = SPECIAL_SEED_RANDOM;
      },
      {serialize: false},
    );

    this.addWidget(
      "button",
      "🎲 New Fixed Random",
      "",
      () => {
        this.seedWidget.value =
          Math.floor(Math.random() * this.randomRange) * (step / 10) + this.randMin;
      },
      {serialize: false},
    );

    this.lastSeedButton = this.addWidget(
      "button",
      LAST_SEED_BUTTON_LABEL,
      "",
      () => {
        this.seedWidget.value = this.lastSeed != null ? this.lastSeed : this.seedWidget.value;
        this.lastSeedButton.name = LAST_SEED_BUTTON_LABEL;
        this.lastSeedButton.disabled = true;
      },
      {width: 50, serialize: false} as any,
    ) as IButtonWidget;
    this.lastSeedButton.disabled = true;
  }

  override getExtraMenuOptions(canvas: LGraphCanvas, options: IContextMenuValue[]) {
    super.getExtraMenuOptions?.apply(this, [...arguments] as any);
    options.splice(options.length - 1, 0, {
      content: "Show/Hide Last Seed Value",
      callback: (
        _value: IContextMenuValue,
        _options: IContextMenuOptions,
        _event: MouseEvent,
        _parentMenu: ContextMenu | undefined,
        _node: TLGraphNode,
      ) => {
        this.properties["showLastSeed"] = !this.properties["showLastSeed"];
        if (this.properties["showLastSeed"]) {
          this.addLastSeedValue();
        } else {
          this.removeLastSeedValue();
        }
      },
    });
    return [];
  }

  addLastSeedValue() {
    if (this.lastSeedValue) return;
    this.lastSeedValue = ComfyWidgets["STRING"](
      this,
      "last_seed",
      ["STRING", {multiline: true}],
      app,
    ).widget as unknown as IWidget;
    this.lastSeedValue!.inputEl!.readOnly = true;
    this.lastSeedValue!.inputEl!.style.fontSize = "0.75rem";
    this.lastSeedValue!.inputEl!.style.textAlign = "center";
    this.computeSize();
  }

  removeLastSeedValue() {
    if (!this.lastSeedValue) return;
    this.lastSeedValue!.inputEl!.remove();
    this.widgets.splice(this.widgets.indexOf(this.lastSeedValue), 1);
    this.lastSeedValue = null;
    this.computeSize();
  }

  /**
   * Intercepts the prompt right before ComfyUI sends it to the server (as fired from rgthree) so we
   * can inspect the prompt and workflow data and change swap in the seeds.
   *
   * Note, the original implementation tried to change the widget value itself when the graph was
   * queued (and the relied on ComfyUI serializing the data changed data) and then changing it back.
   * This worked well until other extensions kept calling graphToPrompt during asynchronous
   * operations within, causing the widget to get confused without a reliable state to reflect upon.
   */
  handleApiHijacking(e: CustomEvent<ComfyApiPrompt>) {
    // Don't do any work if we're muted/bypassed.
    if (this.mode === LiteGraph.NEVER || this.mode === 4) {
      return;
    }

    const workflow = e.detail.workflow;
    const output = e.detail.output;

    let workflowNode = workflow?.nodes?.find((n: ISerialisedNode) => n.id === this.id) ?? null;
    let outputInputs = output?.[this.id]?.inputs;

    if (
      !workflowNode ||
      !outputInputs ||
      outputInputs[this.seedWidget.name || "seed"] === undefined
    ) {
      const [n, v] = this.logger.warnParts(
        `Node ${this.id} not found in prompt data sent to server. This may be fine if only ` +
          `queuing part of the workflow. If not, then this could be a bug.`,
      );
      console[n]?.(...v);
      return;
    }

    const seedToUse = this.getSeedToUse();
    const seedWidgetndex = this.widgets.indexOf(this.seedWidget);

    workflowNode.widgets_values![seedWidgetndex] = seedToUse;
    outputInputs[this.seedWidget.name || "seed"] = seedToUse;

    this.lastSeed = seedToUse;
    if (seedToUse != this.seedWidget.value) {
      this.lastSeedButton.name = `♻️ ${this.lastSeed}`;
      this.lastSeedButton.disabled = false;
    } else {
      this.lastSeedButton.name = LAST_SEED_BUTTON_LABEL;
      this.lastSeedButton.disabled = true;
    }
    if (this.lastSeedValue) {
      this.lastSeedValue.value = `Last Seed: ${this.lastSeed}`;
    }
  }

  /**
   * Determines a seed to use depending on the seed widget's current value and the last used seed.
   * There are no sideffects to calling this method.
   */
  private getSeedToUse() {
    const inputSeed = Number(this.seedWidget.value);
    let seedToUse: number | null = null;

    // If our input seed was a special seed, then handle it.
    if (SPECIAL_SEEDS.includes(inputSeed)) {
      // If the last seed was not a special seed and we have increment/decrement, then do that on
      // the last seed.
      if (typeof this.lastSeed === "number" && !SPECIAL_SEEDS.includes(this.lastSeed)) {
        if (inputSeed === SPECIAL_SEED_INCREMENT) {
          seedToUse = this.lastSeed + 1;
        } else if (inputSeed === SPECIAL_SEED_DECREMENT) {
          seedToUse = this.lastSeed - 1;
        }
      }
      // If we don't have a seed to use, or it's special seed (like we incremented into one), then
      // we randomize.
      if (seedToUse == null || SPECIAL_SEEDS.includes(seedToUse)) {
        seedToUse =
          Math.floor(Math.random() * this.randomRange) *
            ((this.seedWidget.options.step || 1) / 10) +
          this.randMin;
      }
    }

    return seedToUse ?? inputSeed;
  }

  static override setUp(comfyClass: typeof LGraphNode, nodeData: ComfyNodeDef) {
    RgthreeBaseServerNode.registerForOverride(comfyClass, nodeData, RgthreeSeed);
  }

  static override onRegisteredForOverride(comfyClass: any, ctxClass: any) {
    addConnectionLayoutSupport(RgthreeSeed, app, [
      ["Left", "Right"],
      ["Right", "Left"],
    ]);
    setTimeout(() => {
      RgthreeSeed.category = comfyClass.category;
    });
  }
}

app.registerExtension({
  name: "rgthree.Seed",
  async beforeRegisterNodeDef(nodeType: typeof LGraphNode, nodeData: ComfyNodeDef) {
    if (nodeData.name === RgthreeSeed.type) {
      RgthreeSeed.setUp(nodeType, nodeData);
    }
  },
});
