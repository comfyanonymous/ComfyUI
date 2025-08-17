import type {
  LLink,
  LGraphNode,
  INodeOutputSlot,
  INodeInputSlot,
  ISerialisedNode,
  IComboWidget,
  IBaseWidget,
} from "@comfyorg/frontend";
import type {ComfyNodeDef} from "typings/comfy.js";

import {api} from "scripts/api.js";
import {wait} from "rgthree/common/shared_utils.js";
import {rgthree} from "./rgthree.js";

/** Wraps a node instance keeping closure without mucking the finicky types. */
export class PowerPrompt {
  readonly isSimple: boolean;
  readonly node: LGraphNode;
  readonly promptEl: HTMLTextAreaElement;
  nodeData: ComfyNodeDef;
  readonly combos: {[key: string]: IComboWidget} = {};
  readonly combosValues: {[key: string]: string[]} = {};
  boundOnFreshNodeDefs!: (event: CustomEvent) => void;

  private configuring = false;

  constructor(node: LGraphNode, nodeData: ComfyNodeDef) {
    this.node = node;
    this.node.properties = this.node.properties || {};

    this.node.properties["combos_filter"] = "";

    this.nodeData = nodeData;
    this.isSimple = this.nodeData.name.includes("Simple");

    this.promptEl = (node.widgets![0]! as any).inputEl;
    this.addAndHandleKeyboardLoraEditWeight();

    this.patchNodeRefresh();

    const oldConfigure = this.node.configure;
    this.node.configure = (info: ISerialisedNode) => {
      this.configuring = true;
      oldConfigure?.apply(this.node, [info]);
      this.configuring = false;
    };

    const oldOnConnectionsChange = this.node.onConnectionsChange;
    this.node.onConnectionsChange = (
      type: number,
      slotIndex: number,
      isConnected: boolean,
      link_info: LLink,
      _ioSlot: INodeOutputSlot | INodeInputSlot,
    ) => {
      oldOnConnectionsChange?.apply(this.node, [type, slotIndex, isConnected, link_info, _ioSlot]);
      this.onNodeConnectionsChange(type, slotIndex, isConnected, link_info, _ioSlot);
    };

    const oldOnConnectInput = this.node.onConnectInput;
    this.node.onConnectInput = (
      inputIndex: number,
      outputType: INodeOutputSlot["type"],
      outputSlot: INodeOutputSlot,
      outputNode: LGraphNode,
      outputIndex: number,
    ) => {
      let canConnect = true;
      if (oldOnConnectInput) {
        canConnect = oldOnConnectInput.apply(this.node, [
          inputIndex,
          outputType,
          outputSlot,
          outputNode,
          outputIndex,
        ]);
      }
      return (
        this.configuring ||
        rgthree.loadingApiJson ||
        (canConnect && !this.node.inputs[inputIndex]!.disabled)
      );
    };

    const oldOnConnectOutput = this.node.onConnectOutput;
    this.node.onConnectOutput = (
      outputIndex: number,
      inputType: INodeInputSlot["type"],
      inputSlot: INodeInputSlot,
      inputNode: LGraphNode,
      inputIndex: number,
    ) => {
      let canConnect = true;
      if (oldOnConnectOutput) {
        canConnect = oldOnConnectOutput?.apply(this.node, [
          outputIndex,
          inputType,
          inputSlot,
          inputNode,
          inputIndex,
        ]);
      }
      return (
        this.configuring ||
        rgthree.loadingApiJson ||
        (canConnect && !this.node.outputs[outputIndex]!.disabled)
      );
    };

    const onPropertyChanged = this.node.onPropertyChanged;
    this.node.onPropertyChanged = (property: string, value: any, prevValue: any) => {
      const v = onPropertyChanged && onPropertyChanged.call(this.node, property, value, prevValue);
      if (property === "combos_filter") {
        this.refreshCombos(this.nodeData);
      }
      return v ?? true;
    };

    // Strip all widgets but prompt (we'll re-add them in refreshCombos)
    // this.node.widgets.splice(1);
    for (let i = this.node.widgets!.length - 1; i >= 0; i--) {
      if (this.shouldRemoveServerWidget(this.node.widgets![i]!)) {
        this.node.widgets!.splice(i, 1);
      }
    }

    this.refreshCombos(nodeData);
    setTimeout(() => {
      this.stabilizeInputsOutputs();
    }, 32);
  }

  /**
   * Cleans up optional out puts when we don't have the optional input. Purely a vanity function.
   */
  onNodeConnectionsChange(
    _type: number,
    _slotIndex: number,
    _isConnected: boolean,
    _linkInfo: LLink,
    _ioSlot: INodeOutputSlot | INodeInputSlot,
  ) {
    this.stabilizeInputsOutputs();
  }

  private stabilizeInputsOutputs() {
    // If we are currently "configuring" then skip this stabilization. The connected nodes may
    // not yet be configured.
    if (this.configuring || rgthree.loadingApiJson) {
      return;
    }
    // If our first input is connected, then we can show the proper output.
    const clipLinked = this.node.inputs.some((i) => i.name.includes("clip") && !!i.link);
    const modelLinked = this.node.inputs.some((i) => i.name.includes("model") && !!i.link);
    for (const output of this.node.outputs) {
      const type = (output.type as string).toLowerCase();
      if (type.includes("model")) {
        output.disabled = !modelLinked;
      } else if (type.includes("conditioning")) {
        output.disabled = !clipLinked;
      } else if (type.includes("clip")) {
        output.disabled = !clipLinked;
      } else if (type.includes("string")) {
        // Our text prompt is always enabled, but let's color it so it stands out
        // if the others are disabled. #7F7 is Litegraph's default.
        output.color_off = "#7F7";
        output.color_on = "#7F7";
      }
      if (output.disabled) {
        // this.node.disconnectOutput(index);
      }
    }
  }

  onFreshNodeDefs(event: CustomEvent) {
    this.refreshCombos(event.detail[this.nodeData.name]);
  }

  shouldRemoveServerWidget(widget: IBaseWidget) {
    return (
      widget.name?.startsWith("insert_") ||
      widget.name?.startsWith("target_") ||
      widget.name?.startsWith("crop_") ||
      widget.name?.startsWith("values_")
    );
  }

  refreshCombos(nodeData: ComfyNodeDef) {
    this.nodeData = nodeData;
    let filter: RegExp | null = null;
    if ((this.node.properties["combos_filter"] as string)?.trim()) {
      try {
        filter = new RegExp((this.node.properties["combos_filter"] as string).trim(), "i");
      } catch (e) {
        console.error(`Could not parse "${filter}" for Regular Expression`, e);
        filter = null;
      }
    }

    // Add the combo for hidden inputs of nodeData
    let data = Object.assign(
      {},
      this.nodeData.input?.optional || {},
      this.nodeData.input?.hidden || {},
    );

    for (const [key, value] of Object.entries(data)) {
      //Object.entries(this.nodeData.input?.hidden || {})) {
      if (Array.isArray(value[0])) {
        let values = value[0] as string[];
        if (key.startsWith("insert")) {
          values = filter
            ? values.filter(
                (v, i) => i < 1 || (i == 1 && v.match(/^disable\s[a-z]/i)) || filter?.test(v),
              )
            : values;
          const shouldShow =
            values.length > 2 || (values.length > 1 && !values[1]!.match(/^disable\s[a-z]/i));
          if (shouldShow) {
            if (!this.combos[key]) {
              this.combos[key] = this.node.addWidget(
                "combo",
                key,
                values[0]!,
                (selected) => {
                  if (selected !== values[0] && !selected.match(/^disable\s[a-z]/i)) {
                    // We wait a frame because if we use a keydown event to call, it'll wipe out
                    // the selection.
                    wait().then(() => {
                      if (key.includes("embedding")) {
                        this.insertSelectionText(`embedding:${selected}`);
                      } else if (key.includes("saved")) {
                        this.insertSelectionText(
                          this.combosValues[`values_${key}`]![values.indexOf(selected)]!,
                        );
                      } else if (key.includes("lora")) {
                        this.insertSelectionText(`<lora:${selected}:1.0>`);
                      }
                      this.combos[key]!.value = values[0]!;
                    });
                  }
                },
                {
                  values,
                  serialize: true, // Don't include this in prompt.
                },
              ) as IComboWidget;
              (this.combos[key]! as any).oldComputeSize = this.combos[key]!.computeSize;
              let node = this.node;
              this.combos[key]!.computeSize = function (width: number) {
                const size = (this as any).oldComputeSize?.(width) || [
                  width,
                  LiteGraph.NODE_WIDGET_HEIGHT,
                ];
                if (this === node.widgets![node.widgets!.length - 1]) {
                  size[1] += 10;
                }
                return size;
              };
            }
            this.combos[key]!.options!.values = values;
            this.combos[key]!.value = values[0]!;
          } else if (!shouldShow && this.combos[key]) {
            this.node.widgets!.splice(this.node.widgets!.indexOf(this.combos[key]!), 1);
            delete this.combos[key];
          }
        } else if (key.startsWith("values")) {
          this.combosValues[key] = values;
        }
      }
    }
  }

  insertSelectionText(text: string) {
    if (!this.promptEl) {
      console.error("Asked to insert text, but no textbox found.");
      return;
    }
    let prompt = this.promptEl.value;
    // Use selectionEnd as the split; if we have highlighted text, then we likely don't want to
    // overwrite it (we could have just deleted it more easily).
    let first = prompt.substring(0, this.promptEl.selectionEnd).replace(/ +$/, "");
    first = first + (["\n"].includes(first[first.length - 1]!) ? "" : first.length ? " " : "");
    let second = prompt.substring(this.promptEl.selectionEnd).replace(/^ +/, "");
    second = (["\n"].includes(second[0]!) ? "" : second.length ? " " : "") + second;
    this.promptEl.value = first + text + second;
    this.promptEl.focus();
    this.promptEl.selectionStart = first.length;
    this.promptEl.selectionEnd = first.length + text.length;
  }

  /**
   * Adds a keydown event listener to our prompt so we can see if we're using the
   * ctrl/cmd + up/down arrows shortcut. This kind of competes with the core extension
   * "Comfy.EditAttention" but since that only handles parenthesis and listens on window, we should
   * be able to intercept and cancel the bubble if we're doing the same action within the lora tag.
   */
  addAndHandleKeyboardLoraEditWeight() {
    this.promptEl.addEventListener("keydown", (event: KeyboardEvent) => {
      // If we're not doing a ctrl/cmd + arrow key, then bail.
      if (!(event.key === "ArrowUp" || event.key === "ArrowDown")) return;
      if (!event.ctrlKey && !event.metaKey) return;
      // Unfortunately, we can't see Comfy.EditAttention delta in settings, so we hardcode to 0.01.
      // We can acutally do better too, let's make it .1 by default, and .01 if also holding shift.
      const delta = event.shiftKey ? 0.01 : 0.1;

      let start = this.promptEl.selectionStart;
      let end = this.promptEl.selectionEnd;
      let fullText = this.promptEl.value;
      let selectedText = fullText.substring(start, end);

      // We don't care about fully rewriting Comfy.EditAttention, we just want to see if our
      // selected text is a lora, which will always start with "<lora:". So work backwards until we
      // find something that we know can't be a lora, or a "<".
      if (!selectedText) {
        const stopOn = "<>()\r\n\t"; // Allow spaces, since they can be in the filename
        if (fullText[start] == ">") {
          start -= 2;
          end -= 2;
        }
        if (fullText[end - 1] == "<") {
          start += 2;
          end += 2;
        }
        while (!stopOn.includes(fullText[start]!) && start > 0) {
          start--;
        }
        while (!stopOn.includes(fullText[end - 1]!) && end < fullText.length) {
          end++;
        }
        selectedText = fullText.substring(start, end);
      }

      // Bail if this isn't a lora.
      if (!selectedText.startsWith("<lora:") || !selectedText.endsWith(">")) {
        return;
      }

      let weight = Number(selectedText.match(/:(-?\d*(\.\d*)?)>$/)?.[1]) ?? 1;
      weight += event.key === "ArrowUp" ? delta : -delta;
      const updatedText = selectedText.replace(/(:-?\d*(\.\d*)?)?>$/, `:${weight.toFixed(2)}>`);

      // Handle the new value and cancel the bubble so Comfy.EditAttention doesn't also try.
      this.promptEl.setRangeText(updatedText, start, end, "select");
      event.preventDefault();
      event.stopPropagation();
    });
  }

  /**
   * Patches over api.getNodeDefs in comfy's api.js to fire a custom event that we can listen to
   * here and manually refresh our combos when a request comes in to fetch the node data; which
   * only happens once at startup (but before custom nodes js runs), and then after clicking
   * the "Refresh" button in the floating menu, which is what we care about.
   */
  patchNodeRefresh() {
    this.boundOnFreshNodeDefs = this.onFreshNodeDefs.bind(this);
    api.addEventListener("fresh-node-defs", this.boundOnFreshNodeDefs as EventListener);
    const oldNodeRemoved = this.node.onRemoved;
    this.node.onRemoved = () => {
      oldNodeRemoved?.call(this.node);
      api.removeEventListener("fresh-node-defs", this.boundOnFreshNodeDefs as EventListener);
    };
  }
}
