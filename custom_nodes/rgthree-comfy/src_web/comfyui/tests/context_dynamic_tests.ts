import type {LGraphNode} from "@comfyorg/frontend";

import {NodeTypesString} from "../constants.js";
import {wait} from "rgthree/common/shared_utils.js";
import {describe, should, beforeEach, expect, describeRun} from "../testing/runner.js";
import {ComfyUITestEnvironment} from "../testing/comfyui_env.js";

const env = new ComfyUITestEnvironment();

function verifyInputAndOutputName(
  node: LGraphNode,
  index: number,
  inputName: string | null,
  isLinked?: boolean,
) {
  if (inputName != null) {
    expect(node.inputs[index]!.name).toBe(`input ${index} name`, inputName);
  }
  if (isLinked) {
    expect(node.inputs[index]!.link).toBeANumber(`input ${index} connection`);
  } else if (isLinked === false) {
    expect(node.inputs[index]!.link).toBeNullOrUndefined(`input ${index} connection`);
  }
  if (inputName != null) {
    if (inputName === "+") {
      expect(node.outputs[index]).toBeUndefined(`output ${index}`);
    } else {
      let outputName =
        inputName === "base_ctx" ? "CONTEXT" : inputName.replace(/^\+\s/, "").toUpperCase();
      expect(node.outputs[index]!.name).toBe(`output ${index} name`, outputName);
    }
  }
}

function vertifyInputsStructure(node: LGraphNode, expectedLength: number) {
  expect(node.inputs.length).toBe("inputs length", expectedLength);
  expect(node.outputs.length).toBe("outputs length", expectedLength - 1);
  verifyInputAndOutputName(node, expectedLength - 1, "+", false);
}

describe("TestContextDynamic", async () => {
  let nodeConfig!: LGraphNode;
  let nodeCtx!: LGraphNode;

  let lastNode: LGraphNode | null = null;

  await beforeEach(async () => {
    await env.clear();
    lastNode = nodeConfig = await env.addNode(NodeTypesString.KSAMPLER_CONFIG);
    lastNode = nodeCtx = await env.addNode(NodeTypesString.DYNAMIC_CONTEXT);
    nodeConfig.connect(0, nodeCtx, 1); // steps
    nodeConfig.connect(2, nodeCtx, 2); // cfg
    nodeConfig.connect(4, nodeCtx, 3); // scheduler
    nodeConfig.connect(0, nodeCtx, 4); // This is the step.1
    nodeConfig.connect(0, nodeCtx, 5); // This is the step.2
    nodeCtx.disconnectInput(2);
    nodeCtx.disconnectInput(5);
    nodeConfig.connect(0, nodeCtx, 6); // This is the step.3
    nodeCtx.disconnectInput(6);
    await wait();
  });

  await should("add correct inputs", async () => {
    vertifyInputsStructure(nodeCtx, 8);
    let i = 0;
    verifyInputAndOutputName(nodeCtx, i++, "base_ctx", false);
    verifyInputAndOutputName(nodeCtx, i++, "+ steps", true);
    verifyInputAndOutputName(nodeCtx, i++, "+ cfg", false);
    verifyInputAndOutputName(nodeCtx, i++, "+ scheduler", true);
    verifyInputAndOutputName(nodeCtx, i++, "+ steps.1", true);
    verifyInputAndOutputName(nodeCtx, i++, "+ steps.2", false);
    verifyInputAndOutputName(nodeCtx, i++, "+ steps.3", false);
  });

  await should("add evaluate correct outputs", async () => {
    const displayAny1 = await env.addNode(NodeTypesString.DISPLAY_ANY, {placement: "right"});
    const displayAny2 = await env.addNode(NodeTypesString.DISPLAY_ANY, {placement: "under"});
    const displayAny3 = await env.addNode(NodeTypesString.DISPLAY_ANY, {placement: "under"});
    const displayAny4 = await env.addNode(NodeTypesString.DISPLAY_ANY, {placement: "under"});

    nodeCtx.connect(1, displayAny1, 0); // steps
    nodeCtx.connect(3, displayAny2, 0); // scheduler
    nodeCtx.connect(4, displayAny3, 0); // steps.1
    nodeCtx.connect(6, displayAny4, 0); // steps.3 (unlinked)

    await env.queuePrompt();

    expect(displayAny1.widgets![0]!.value).toBe("output 1", 30);
    expect(displayAny2.widgets![0]!.value).toBe("output 3", '"normal"');
    expect(displayAny3.widgets![0]!.value).toBe("output 4", 30);
    expect(displayAny4.widgets![0]!.value).toBe("output 6", "None");
  });

  await describeRun("Nested", async () => {
    let nodeConfig2!: LGraphNode;
    let nodeCtx2!: LGraphNode;

    await beforeEach(async () => {
      nodeConfig2 = await env.addNode(NodeTypesString.KSAMPLER_CONFIG, {placement: "start"});
      nodeConfig2.widgets = nodeConfig2.widgets || [];
      nodeConfig2.widgets[0]!.value = 111;
      nodeConfig2.widgets[2]!.value = 11.1;
      nodeCtx2 = await env.addNode(NodeTypesString.DYNAMIC_CONTEXT, {placement: "right"});
      nodeConfig2.connect(0, nodeCtx2, 1); // steps
      nodeConfig2.connect(2, nodeCtx2, 2); // cfg
      nodeConfig2.connect(3, nodeCtx2, 3); // sampler
      nodeConfig2.connect(2, nodeCtx2, 4); // This is the cfg.1
      nodeConfig2.connect(0, nodeCtx2, 5); // This is the steps.1
      nodeCtx2.disconnectInput(2);
      nodeCtx2.disconnectInput(5);
      nodeConfig2.connect(2, nodeCtx2, 6); // This is the cfg.2
      nodeCtx2.disconnectInput(6);

      await wait();
    });

    await should("disallow context node to be connected to non-first spot.", async () => {
      // Connect to first node.
      let expectedInputs = 8;

      nodeCtx2.connect(0, nodeCtx, expectedInputs - 1);
      console.log(nodeCtx.inputs);

      vertifyInputsStructure(nodeCtx, expectedInputs);
      verifyInputAndOutputName(nodeCtx, 0, "base_ctx", false);
      verifyInputAndOutputName(nodeCtx, nodeCtx.inputs.length - 1, null, false);

      nodeCtx2.connect(0, nodeCtx, 0);
      expectedInputs = 14;
      vertifyInputsStructure(nodeCtx, expectedInputs);
      verifyInputAndOutputName(nodeCtx, 0, "base_ctx", true);
      verifyInputAndOutputName(nodeCtx, expectedInputs - 1, null, false);
    });

    await should("add inputs from connected above owned.", async () => {
      // Connect to first node.
      nodeCtx2.connect(0, nodeCtx, 0);

      let expectedInputs = 14;
      vertifyInputsStructure(nodeCtx, expectedInputs);
      let i = 0;
      verifyInputAndOutputName(nodeCtx, i++, "base_ctx", true);
      verifyInputAndOutputName(nodeCtx, i++, "steps", false);
      verifyInputAndOutputName(nodeCtx, i++, "cfg", false);
      verifyInputAndOutputName(nodeCtx, i++, "sampler", false);
      verifyInputAndOutputName(nodeCtx, i++, "cfg.1", false);
      verifyInputAndOutputName(nodeCtx, i++, "steps.1", false);
      verifyInputAndOutputName(nodeCtx, i++, "cfg.2", false);
      verifyInputAndOutputName(nodeCtx, i++, "+ steps.2", true);
      verifyInputAndOutputName(nodeCtx, i++, "+ cfg.3", false);
      verifyInputAndOutputName(nodeCtx, i++, "+ scheduler", true);
      verifyInputAndOutputName(nodeCtx, i++, "+ steps.3", true);
      verifyInputAndOutputName(nodeCtx, i++, "+ steps.4", false);
      verifyInputAndOutputName(nodeCtx, i++, "+ steps.5", false);
      verifyInputAndOutputName(nodeCtx, i++, "+", false);
    });

    await should("add then remove inputs when disconnected.", async () => {
      // Connect to first node.
      nodeCtx2.connect(0, nodeCtx, 0);

      let expectedInputs = 14;
      expect(nodeCtx.inputs.length).toBe("inputs length", expectedInputs);
      expect(nodeCtx.outputs.length).toBe("outputs length", expectedInputs - 1);

      nodeCtx.disconnectInput(0);

      expectedInputs = 8;
      expect(nodeCtx.inputs.length).toBe("inputs length", expectedInputs);
      expect(nodeCtx.outputs.length).toBe("outputs length", expectedInputs - 1);
      let i = 0;
      verifyInputAndOutputName(nodeCtx, i++, "base_ctx", false);
      verifyInputAndOutputName(nodeCtx, i++, "+ steps", true);
      verifyInputAndOutputName(nodeCtx, i++, "+ cfg", false);
      verifyInputAndOutputName(nodeCtx, i++, "+ scheduler", true);
      verifyInputAndOutputName(nodeCtx, i++, "+ steps.1", true);
      verifyInputAndOutputName(nodeCtx, i++, "+ steps.2", false);
      verifyInputAndOutputName(nodeCtx, i++, "+ steps.3", false);
      verifyInputAndOutputName(nodeCtx, i++, "+", false);
    });
  });
});
