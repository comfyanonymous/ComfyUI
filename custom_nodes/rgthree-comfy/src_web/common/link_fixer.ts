import type {
  LGraph,
  LGraphNode,
  LLink,
  ISlotType,
  INodeOutputSlot,
  INodeInputSlot,
  SerialisedLLinkArray,
  LinkId,
  ISerialisedNode,
  ISerialisedGraph,
  NodeId,
} from "@comfyorg/frontend";

/**
 * The bad links data returned from either a fixer `check()`, or the results of a `fix()` call.
 */
export interface BadLinksData<T = ISerialisedGraph | LGraph> {
  hasBadLinks: boolean;
  graph: T;
  patches: number;
  deletes: number;
}

enum IoDirection {
  INPUT,
  OUTPUT,
}

/**
 * Data interface that mimics a nodes `inputs` and `outputs` holding the _to be_ mutated node data
 * during a check.
 */
interface PatchedNodeSlots {
  [nodeId: string]: {
    inputs?: {[slot: number]: number | null};
    outputs?: {
      [slots: number]: {
        links: number[];
        changes: {[linkId: number]: "ADD" | "REMOVE"};
      };
    };
  };
}

/**
 * Link data derived from either a ISerialisedGraph or LGraph `links` property.
 */
interface LinkData {
  id: LinkId;
  origin_id: NodeId;
  origin_slot: number;
  target_id: NodeId;
  target_slot: number;
  type: ISlotType;
}

/**
 * Returns a list of links data for the given links type; either from an LGraph or SerializedGraph.
 */
function getLinksData(
  links: ISerialisedGraph["links"] | LGraph["links"] | {[key: string]: LLink},
): LinkData[] {
  if (links instanceof Map) {
    const data: LinkData[] = [];
    for (const [key, llink] of links.entries()) {
      if (!llink) continue;
      data.push(llink);
    }
    return data;
  }
  // This is apparently marked deprecated in ComfyUI but who knows if we would get stale data in
  // here that's not a map (handled above). Go ahead and handle it anyway.
  if (!Array.isArray(links)) {
    const data: LinkData[] = [];
    for (const key in links) {
      const llink = (links.hasOwnProperty(key) && links[key]) || null;
      if (!llink) continue;
      data.push(llink);
    }
    return data;
  }
  return links.map((link: SerialisedLLinkArray) => ({
    id: link[0],
    origin_id: link[1],
    origin_slot: link[2],
    target_id: link[3],
    target_slot: link[4],
    type: link[5],
  }));
}

/** The instruction data for fixing a node's inputs or outputs. */
interface WorkflowLinkFixerNodeInstruction {
  node: ISerialisedNode | LGraphNode;
  op: "REMOVE" | "ADD";
  dir: IoDirection;
  slot: number;
  linkId: number;
  linkIdToUse: number | null;
}

/** The instruction data for fixing a link from a workflow links. */
interface WorkflowLinkFixerLinksInstruction {
  op: "DELETE";
  linkId: number;
  reason: string;
}

type WorkflowLinkFixerInstruction =
  | WorkflowLinkFixerNodeInstruction
  | WorkflowLinkFixerLinksInstruction;

/**
 * The WorkflowLinkFixer for either ISerialisedGraph or a live LGraph.
 *
 * Use `WorkflowLinkFixer.create(graph: ISerialisedGraph | LGraph)` to create a new instance.
 */
export abstract class WorkflowLinkFixer<
  G extends ISerialisedGraph | LGraph,
  N extends ISerialisedNode | LGraphNode,
> {
  silent: boolean = false;
  checkedData: BadLinksData<G> | null = null;

  protected logger: {log: (...args: any[]) => void} = console;
  protected graph: G;
  protected patchedNodeSlots: PatchedNodeSlots = {};
  protected instructions: WorkflowLinkFixerInstruction[] = [];

  /**
   * Creates the WorkflowLinkFixer for the given graph type.
   */
  static create(graph: ISerialisedGraph): WorkflowLinkFixerSerialized;
  static create(graph: LGraph): WorkflowLinkFixerGraph;
  static create(
    graph: ISerialisedGraph | LGraph,
  ): WorkflowLinkFixerSerialized | WorkflowLinkFixerGraph {
    if (typeof (graph as LGraph).getNodeById === "function") {
      return new WorkflowLinkFixerGraph(graph as LGraph);
    }
    return new WorkflowLinkFixerSerialized(graph as ISerialisedGraph);
  }

  protected constructor(graph: G) {
    this.graph = graph;
  }

  abstract getNodeById(id: NodeId): N | null;
  abstract deleteGraphLink(id: LinkId): true | string;

  /**
   * Checks the current graph data for any bad links.
   */
  check(force: boolean = false): BadLinksData<G> {
    if (this.checkedData && !force) {
      return {...this.checkedData};
    }
    this.instructions = [];
    this.patchedNodeSlots = {};

    const instructions: (WorkflowLinkFixerInstruction | null)[] = [];

    const links: LinkData[] = getLinksData(this.graph.links);
    links.reverse();
    for (const link of links) {
      if (!link) continue;

      const originNode = this.getNodeById(link.origin_id);
      const originHasLink = () =>
        this.nodeHasLinkId(originNode!, IoDirection.OUTPUT, link.origin_slot, link.id);
      const patchOrigin = (op: "ADD" | "REMOVE", id = link.id) =>
        this.getNodePatchInstruction(originNode!, IoDirection.OUTPUT, link.origin_slot, id, op);

      const targetNode = this.getNodeById(link.target_id);
      const targetHasLink = () =>
        this.nodeHasLinkId(targetNode!, IoDirection.INPUT, link.target_slot, link.id);
      const targetHasAnyLink = () =>
        this.nodeHasAnyLink(targetNode!, IoDirection.INPUT, link.target_slot);
      const patchTarget = (op: "ADD" | "REMOVE", id = link.id) =>
        this.getNodePatchInstruction(targetNode!, IoDirection.INPUT, link.target_slot, id, op);

      const originLog = `origin(${link.origin_id}).outputs[${link.origin_slot}].links`;
      const targetLog = `target(${link.target_id}).inputs[${link.target_slot}].link`;

      if (!originNode || !targetNode) {
        if (!originNode && !targetNode) {
          // This can fall through and continue; we remove it after this loop.
        } else if (!originNode && targetNode) {
          this.log(
            `Link ${link.id} is funky... ` +
              `origin ${link.origin_id} does not exist, but target ${link.target_id} does.`,
          );
          if (targetHasLink()) {
            this.log(` > [PATCH] ${targetLog} does have link, will remove the inputs' link first.`);
            instructions.push(patchTarget("REMOVE", -1));
          }
        } else if (!targetNode && originNode) {
          this.log(
            `Link ${link.id} is funky... ` +
              `target ${link.target_id} does not exist, but origin ${link.origin_id} does.`,
          );
          if (originHasLink()) {
            this.log(` > [PATCH] Origin's links' has ${link.id}; will remove the link first.`);
            instructions.push(patchOrigin("REMOVE"));
          }
        }
        continue;
      }

      if (targetHasLink() || originHasLink()) {
        if (!originHasLink()) {
          this.log(
            `${link.id} is funky... ${originLog} does NOT contain it, but ${targetLog} does.`,
          );
          this.log(` > [PATCH] Attempt a fix by adding this ${link.id} to ${originLog}.`);
          instructions.push(patchOrigin("ADD"));
        } else if (!targetHasLink()) {
          this.log(
            `${link.id} is funky... ${targetLog} is NOT correct (is ${
              targetNode.inputs?.[link.target_slot]?.link
            }), but ${originLog} contains it`,
          );
          if (!targetHasAnyLink()) {
            this.log(` > [PATCH] ${targetLog} is not defined, will set to ${link.id}.`);
            let instruction = patchTarget("ADD");
            if (!instruction) {
              this.log(
                ` > [PATCH] Nvm, ${targetLog} already patched. Removing ${link.id} from ${originLog}.`,
              );
              instruction = patchOrigin("REMOVE");
            }
            instructions.push(instruction);
          } else {
            this.log(` > [PATCH] ${targetLog} is defined, removing ${link.id} from ${originLog}.`);
            instructions.push(patchOrigin("REMOVE"));
          }
        }
      }
    }

    // Now that we've cleaned up the inputs, outputs, run through it looking for dangling links.,
    for (let link of links) {
      if (!link) continue;
      const originNode = this.getNodeById(link.origin_id);
      const targetNode = this.getNodeById(link.target_id);
      if (!originNode && !targetNode) {
        instructions.push({
          op: "DELETE",
          linkId: link.id,
          reason: `Both nodes #${link.origin_id} & #${link.target_id} are removed`,
        });
      }
      // Now that we've manipulated the linking, check again if they both exist.
      if (
        (!originNode ||
          !this.nodeHasLinkId(originNode, IoDirection.OUTPUT, link.origin_slot, link.id)) &&
        (!targetNode ||
          !this.nodeHasLinkId(targetNode, IoDirection.INPUT, link.target_slot, link.id))
      ) {
        instructions.push({
          op: "DELETE",
          linkId: link.id,
          reason:
            `both origin node #${link.origin_id} ` +
            `${!originNode ? "is removed" : `is missing link id output slot ${link.origin_slot}`}` +
            `and target node #${link.target_id} ` +
            `${!targetNode ? "is removed" : `is missing link id input slot ${link.target_slot}`}.`,
        });
        continue;
      }
    }

    this.instructions = instructions.filter((i) => !!i);
    this.checkedData = {
      hasBadLinks: !!this.instructions.length,
      graph: this.graph,
      patches: this.instructions.filter((i) => !!(i as WorkflowLinkFixerNodeInstruction).node)
        .length,
      deletes: this.instructions.filter((i) => i.op === "DELETE").length,
    };
    return {...this.checkedData};
  }

  /**
   * Fixes a checked graph by running through the instructions generated during the check run. Also
   * double-checks for inconsistencies after the fix, recursively calling itself up to five times
   * before giving up.
   */
  fix(force: boolean = false, times?: number): BadLinksData<G> {
    if (!this.checkedData || force) {
      this.check(force);
    }
    let patches = 0;
    let deletes = 0;
    for (const instruction of this.instructions) {
      if ((instruction as WorkflowLinkFixerNodeInstruction).node) {
        let {node, slot, linkIdToUse, dir, op} = instruction as WorkflowLinkFixerNodeInstruction;
        if (dir == IoDirection.INPUT) {
          node.inputs = node.inputs || [];
          const old = node.inputs[slot]?.link;
          node.inputs[slot] = node.inputs[slot] || ({} as INodeInputSlot);
          node.inputs[slot].link = linkIdToUse;
          this.log(`Node #${node.id}: Set link ${linkIdToUse} to input slot ${slot} (was ${old})`);
        } else if (op === "ADD" && linkIdToUse != null) {
          node.outputs = node.outputs || [];
          node.outputs[slot] = node.outputs[slot] || ({} as INodeOutputSlot);
          node.outputs[slot].links = node.outputs[slot].links || [];
          node.outputs[slot].links.push(linkIdToUse);
          this.log(`Node #${node.id}: Add link ${linkIdToUse} to output slot #${slot}`);
        } else if (op === "REMOVE" && linkIdToUse != null) {
          // We should never not have this data since the check call would have found it to be
          // removed, but we can be safe and appease TS compiler at the same time.
          if (node.outputs?.[slot]?.links?.length === undefined) {
            this.log(
              `Node #${node.id}: Couldn't remove link ${linkIdToUse} from output slot #${slot}` +
                ` because it didn't exist.`,
            );
          } else {
            let linkIdIndex = node.outputs![slot].links.indexOf(linkIdToUse);
            node.outputs[slot].links.splice(linkIdIndex, 1);
            this.log(`Node #${node.id}: Remove link ${linkIdToUse} from output slot #${slot}`);
          }
        } else {
          throw new Error("Unhandled Node Instruction");
        }
        patches++;
      } else if (instruction.op === "DELETE") {
        const wasDeleted = this.deleteGraphLink(instruction.linkId);
        if (wasDeleted === true) {
          this.log(`Link #${instruction.linkId}: Removed workflow link b/c ${instruction.reason}`);
        } else {
          this.log(`Error Link #${instruction.linkId} was not removed!`);
        }
        deletes += wasDeleted ? 1 : 0;
      } else {
        throw new Error("Unhandled Instruction");
      }
    }

    const newCheck = this.check(force);
    times = times == null ? 5 : times;
    let newFix = null;
    // If we still have bad links, then recurse (up to five times).
    if (newCheck.hasBadLinks && times > 0) {
      newFix = this.fix(true, times - 1);
    }

    return {
      hasBadLinks: newFix?.hasBadLinks ?? newCheck.hasBadLinks,
      graph: this.graph,
      patches: patches + (newFix?.patches ?? 0),
      deletes: deletes + (newFix?.deletes ?? 0),
    };
  }

  /** Logs if not silent. */
  protected log(...args: any[]) {
    if (this.silent) return;
    this.logger.log(...args);
  }

  /**
   * Patches a node for a check run, returning the instruction that would be made.
   */
  private getNodePatchInstruction(
    node: N,
    ioDir: IoDirection,
    slot: number,
    linkId: number,
    op: "ADD" | "REMOVE",
  ): WorkflowLinkFixerNodeInstruction | null {
    const nodeId = node.id;
    this.patchedNodeSlots[nodeId] = this.patchedNodeSlots[nodeId] || {};
    const patchedNode = this.patchedNodeSlots[nodeId];
    if (ioDir == IoDirection.INPUT) {
      patchedNode["inputs"] = patchedNode["inputs"] || {};
      // We can set to null (delete), so undefined means we haven't set it at all.
      if (patchedNode["inputs"][slot] !== undefined) {
        this.log(
          ` > Already set ${nodeId}.inputs[${slot}] to ${patchedNode["inputs"][slot]} Skipping.`,
        );
        return null;
      }
      let linkIdToUse = op === "REMOVE" ? null : linkId;
      patchedNode["inputs"][slot] = linkIdToUse;
      return {node, dir: ioDir, op, slot, linkId, linkIdToUse};
    }

    patchedNode["outputs"] = patchedNode["outputs"] || {};
    patchedNode["outputs"][slot] = patchedNode["outputs"][slot] || {
      links: [...(node.outputs?.[slot]?.links || [])],
      changes: {},
    };
    if (patchedNode["outputs"][slot]["changes"][linkId] !== undefined) {
      this.log(
        ` > Already set ${nodeId}.outputs[${slot}] to ${patchedNode["outputs"][slot]}! Skipping.`,
      );
      return null;
    }
    patchedNode["outputs"][slot]["changes"][linkId] = op;
    if (op === "ADD") {
      let linkIdIndex = patchedNode["outputs"][slot]["links"].indexOf(linkId);
      if (linkIdIndex !== -1) {
        this.log(` > Hmmm.. asked to add ${linkId} but it is already in list...`);
        return null;
      }
      patchedNode["outputs"][slot]["links"].push(linkId);
      return {node, dir: ioDir, op, slot, linkId, linkIdToUse: linkId};
    }

    let linkIdIndex = patchedNode["outputs"][slot]["links"].indexOf(linkId);
    if (linkIdIndex === -1) {
      this.log(` > Hmmm.. asked to remove ${linkId} but it doesn't exist...`);
      return null;
    }
    patchedNode["outputs"][slot]["links"].splice(linkIdIndex, 1);
    return {node, dir: ioDir, op, slot, linkId, linkIdToUse: linkId};
  }

  /** Checks if a node (or patched data) has a linkId. */
  private nodeHasLinkId(node: N, ioDir: IoDirection, slot: number, linkId: number) {
    const nodeId = node.id;
    let has = false;
    if (ioDir === IoDirection.INPUT) {
      let nodeHasIt = node.inputs?.[slot]?.link === linkId;
      if (this.patchedNodeSlots[nodeId]?.["inputs"]) {
        let patchedHasIt = this.patchedNodeSlots[nodeId]["inputs"][slot] === linkId;
        has = patchedHasIt;
      } else {
        has = nodeHasIt;
      }
    } else {
      let nodeHasIt = node.outputs?.[slot]?.links?.includes(linkId);
      if (this.patchedNodeSlots[nodeId]?.["outputs"]?.[slot]?.["changes"][linkId]) {
        let patchedHasIt = this.patchedNodeSlots[nodeId]["outputs"][slot].links.includes(linkId);
        has = !!patchedHasIt;
      } else {
        has = !!nodeHasIt;
      }
    }
    return has;
  }

  /** Checks if a node (or patched data) has a linkId. */
  private nodeHasAnyLink(node: N, ioDir: IoDirection, slot: number) {
    // Patched data should be canonical. We can double check if fixing too.
    const nodeId = node.id;
    let hasAny = false;
    if (ioDir === IoDirection.INPUT) {
      let nodeHasAny = node.inputs?.[slot]?.link != null;
      if (this.patchedNodeSlots[nodeId]?.["inputs"]) {
        let patchedHasAny = this.patchedNodeSlots[nodeId]["inputs"][slot] != null;
        hasAny = patchedHasAny;
      } else {
        hasAny = !!nodeHasAny;
      }
    } else {
      let nodeHasAny = node.outputs?.[slot]?.links?.length;
      if (this.patchedNodeSlots[nodeId]?.["outputs"]?.[slot]?.["changes"]) {
        let patchedHasAny = this.patchedNodeSlots[nodeId]["outputs"][slot].links?.length;
        hasAny = !!patchedHasAny;
      } else {
        hasAny = !!nodeHasAny;
      }
    }
    return hasAny;
  }
}

/**
 * A WorkflowLinkFixer for serialized data.
 */
class WorkflowLinkFixerSerialized extends WorkflowLinkFixer<ISerialisedGraph, ISerialisedNode> {
  constructor(graph: ISerialisedGraph) {
    super(graph);
  }

  getNodeById(id: NodeId) {
    return this.graph.nodes.find((node) => Number(node.id) === id) ?? null;
  }

  override fix(force: boolean = false, times?: number) {
    const ret = super.fix(force, times);
    // If we're a serialized graph, we can filter out the links because it's just an array.
    this.graph.links = this.graph.links.filter((l) => !!l);
    return ret;
  }

  deleteGraphLink(id: LinkId) {
    // Sometimes we got objects instead of serializzed array for links if passed after ComfyUI's
    // loadGraphData modifies the data. Let's find the id handling the bastardized objects just in
    // case.
    const idx = this.graph.links.findIndex((l) => l && (l[0] === id || (l as any).id === id));
    if (idx === -1) {
      return `Link #${id} not found in workflow links.`;
    }
    this.graph.links.splice(idx, 1);
    return true;
  }
}

/**
 * A WorkflowLinkFixer for live LGraph data.
 */
class WorkflowLinkFixerGraph extends WorkflowLinkFixer<LGraph, LGraphNode> {
  constructor(graph: LGraph) {
    super(graph);
  }

  getNodeById(id: NodeId) {
    return this.graph.getNodeById(id) ?? null;
  }

  deleteGraphLink(id: LinkId) {
    if (this.graph.links instanceof Map) {
      if (!this.graph.links.has(id)) {
        return `Link #${id} not found in workflow links.`;
      }
      this.graph.links.delete(id);
      return true;
    }
    if (this.graph.links[id] == null) {
      return `Link #${id} not found in workflow links.`;
    }
    delete this.graph.links[id];
    return true;
  }
}
