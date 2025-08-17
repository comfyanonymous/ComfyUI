import type {
  Vector2,
  LGraphCanvas as TLGraphCanvas,
  LLink,
  LGraph,
  IContextMenuOptions,
  ContextMenu,
  LGraphNode as TLGraphNode,
  INodeSlot,
  INodeInputSlot,
  INodeOutputSlot,
  IContextMenuValue,
  ISlotType,
  LGraphNodeConstructor,
  LGraphEventMode,
  NodeProperty,
  LinkDirection,
  Point,
  ComfyApp,
  LGraphGroup,
  NodeId,
} from "@comfyorg/frontend";
import type {ComfyNodeDef} from "typings/comfy.js";
import type {Constructor} from "typings/rgthree";

import {app} from "scripts/app.js";
import {api} from "scripts/api.js";
import {Resolver, getResolver, wait} from "rgthree/common/shared_utils.js";
import {RgthreeHelpDialog} from "rgthree/common/dialog.js";

/**
 * Override the api.getNodeDefs call to add a hook for refreshing node defs.
 * This is necessary for power prompt's custom combos. Since API implements
 * add/removeEventListener already, this is rather trivial.
 */
const oldApiGetNodeDefs = api.getNodeDefs;
api.getNodeDefs = async function () {
  const defs = await oldApiGetNodeDefs.call(api);
  this.dispatchEvent(new CustomEvent("fresh-node-defs", {detail: defs}));
  return defs;
};

export enum IoDirection {
  INPUT,
  OUTPUT,
}

const PADDING = 0;

type LiteGraphDir =
  | typeof LiteGraph.LEFT
  | typeof LiteGraph.RIGHT
  | typeof LiteGraph.UP
  | typeof LiteGraph.DOWN;
export const LAYOUT_LABEL_TO_DATA: {[label: string]: [LiteGraphDir, Vector2, Vector2]} = {
  Left: [LiteGraph.LEFT, [0, 0.5], [PADDING, 0]],
  Right: [LiteGraph.RIGHT, [1, 0.5], [-PADDING, 0]],
  Top: [LiteGraph.UP, [0.5, 0], [0, PADDING]],
  Bottom: [LiteGraph.DOWN, [0.5, 1], [0, -PADDING]],
};
export const LAYOUT_LABEL_OPPOSITES: {[label: string]: string} = {
  Left: "Right",
  Right: "Left",
  Top: "Bottom",
  Bottom: "Top",
};
export const LAYOUT_CLOCKWISE = ["Top", "Right", "Bottom", "Left"];

interface MenuConfig {
  name: string | ((node: TLGraphNode) => string);
  property?: string;
  prepareValue?: (value: NodeProperty | undefined, node: TLGraphNode) => any;
  callback?: (node: TLGraphNode, value?: string) => void;
  subMenuOptions?: (string | null)[] | ((node: TLGraphNode) => (string | null)[]);
}

export function addMenuItem(
  node: LGraphNodeConstructor,
  _app: ComfyApp,
  config: MenuConfig,
  after = "Shape",
) {
  const oldGetExtraMenuOptions = node.prototype.getExtraMenuOptions;
  node.prototype.getExtraMenuOptions = function (
    canvas: TLGraphCanvas,
    menuOptions: IContextMenuValue[],
  ) {
    oldGetExtraMenuOptions && oldGetExtraMenuOptions.apply(this, [canvas, menuOptions]);
    addMenuItemOnExtraMenuOptions(this, config, menuOptions, after);
  };
}

/**
 * Waits for the canvas to be available on app using a single promise.
 */
let canvasResolver: Resolver<TLGraphCanvas> | null = null;
export function waitForCanvas() {
  if (canvasResolver === null) {
    canvasResolver = getResolver<TLGraphCanvas>();
    function _waitForCanvas() {
      if (!canvasResolver!.completed) {
        if (app?.canvas) {
          canvasResolver!.resolve(app.canvas);
        } else {
          requestAnimationFrame(_waitForCanvas);
        }
      }
    }
    _waitForCanvas();
  }
  return canvasResolver.promise;
}

/**
 * Waits for the graph to be available on app using a single promise.
 */
let graphResolver: Resolver<LGraph> | null = null;
export function waitForGraph() {
  if (graphResolver === null) {
    graphResolver = getResolver<LGraph>();
    function _wait() {
      if (!graphResolver!.completed) {
        if (app?.graph) {
          graphResolver!.resolve(app.graph);
        } else {
          requestAnimationFrame(_wait);
        }
      }
    }
    _wait();
  }
  return graphResolver.promise;
}

export function addMenuItemOnExtraMenuOptions(
  node: TLGraphNode,
  config: MenuConfig,
  menuOptions: (IContextMenuValue | null)[],
  after = "Shape",
) {
  let idx = menuOptions
    .slice()
    .reverse()
    .findIndex((option) => (option as any)?.isRgthree);
  if (idx == -1) {
    idx = menuOptions.findIndex((option) => option?.content?.includes(after)) + 1;
    if (!idx) {
      idx = menuOptions.length - 1;
    }
    // Add a separator, and move to the next one.
    menuOptions.splice(idx, 0, null);
    idx++;
  } else {
    idx = menuOptions.length - idx;
  }

  const subMenuOptions =
    typeof config.subMenuOptions === "function"
      ? config.subMenuOptions(node)
      : config.subMenuOptions;

  menuOptions.splice(idx, 0, {
    content: typeof config.name == "function" ? config.name(node) : config.name,
    has_submenu: !!subMenuOptions?.length,
    isRgthree: true, // Mark it, so we can find it.
    callback: (
      value: IContextMenuValue,
      _options: IContextMenuOptions,
      event: MouseEvent,
      parentMenu: ContextMenu | undefined,
      _node: TLGraphNode,
    ) => {
      if (!!subMenuOptions?.length) {
        new LiteGraph.ContextMenu(
          subMenuOptions.map((option) => (option ? {content: option} : null)),
          {
            event,
            parentMenu,
            callback: (
              subValue: IContextMenuValue,
              _options: IContextMenuOptions,
              _event: MouseEvent,
              _parentMenu: ContextMenu | undefined,
              _node: TLGraphNode,
            ) => {
              if (config.property) {
                node.properties = node.properties || {};
                node.properties[config.property] = config.prepareValue
                  ? config.prepareValue(subValue!.content || "", node)
                  : subValue!.content || "";
              }
              config.callback && config.callback(node, subValue?.content);
            },
          },
        );
        return;
      }
      if (config.property) {
        node.properties = node.properties || {};
        node.properties[config.property] = config.prepareValue
          ? config.prepareValue(node.properties[config.property], node)
          : !node.properties[config.property];
      }
      config.callback && config.callback(node, value?.content);
    },
  } as IContextMenuValue);
}

export function addConnectionLayoutSupport(
  node: LGraphNodeConstructor,
  app: ComfyApp,
  options = [
    ["Left", "Right"],
    ["Right", "Left"],
  ],
  callback?: (node: TLGraphNode) => void,
) {
  addMenuItem(node, app, {
    name: "Connections Layout",
    property: "connections_layout",
    subMenuOptions: options.map((option) => option[0] + (option[1] ? " -> " + option[1] : "")),
    prepareValue: (value, node) => {
      const values = String(value).split(" -> ");
      if (!values[1] && !node.outputs?.length) {
        values[1] = LAYOUT_LABEL_OPPOSITES[values[0]!]!;
      }
      if (!LAYOUT_LABEL_TO_DATA[values[0]!] || !LAYOUT_LABEL_TO_DATA[values[1]!]) {
        throw new Error(`New Layout invalid: [${values[0]}, ${values[1]}]`);
      }
      return values;
    },
    callback: (node) => {
      callback && callback(node);
      node.graph?.setDirtyCanvas(true, true);
    },
  });

  // [🤮] This is deprecated in a https://github.com/Comfy-Org/litegraph.js/pull/716 @v0.9.9 and
  // replaced by getInputPos and getOutputPos conveniently added below.
  node.prototype.getConnectionPos = function (isInput: boolean, slotNumber: number, out: Vector2) {
    // Purposefully do not need to call the old one.
    return getConnectionPosForLayout(this, isInput, slotNumber, out);
  };
  node.prototype.getInputPos = function (slotNumber: number): Vector2 {
    // Purposefully do not need to call the existing one because we're overriding.
    return getConnectionPosForLayout(this, true, slotNumber, [0, 0]);
  };
  node.prototype.getOutputPos = function (slotNumber: number): Vector2 {
    // Purposefully do not need to call the existing one because we're overriding.
    return getConnectionPosForLayout(this, false, slotNumber, [0, 0]);
  };
}

export function setConnectionsLayout(node: TLGraphNode, newLayout: [string, string]) {
  newLayout = newLayout || (node as any).defaultConnectionsLayout || ["Left", "Right"];
  // If we didn't supply an output layout, and there's no outputs, then just choose the opposite of the
  // input as a safety.
  if (!newLayout[1] && !node.outputs?.length) {
    newLayout[1] = LAYOUT_LABEL_OPPOSITES[newLayout[0]!]!;
  }
  if (!LAYOUT_LABEL_TO_DATA[newLayout[0]] || !LAYOUT_LABEL_TO_DATA[newLayout[1]]) {
    throw new Error(`New Layout invalid: [${newLayout[0]}, ${newLayout[1]}]`);
  }
  node.properties = node.properties || {};
  node.properties["connections_layout"] = newLayout;
}

/** Allows collapsing of connections into one. Pretty unusable, unless you're the muter. */
export function setConnectionsCollapse(
  node: TLGraphNode,
  collapseConnections: boolean | null = null,
) {
  node.properties = node.properties || {};
  collapseConnections =
    collapseConnections !== null ? collapseConnections : !node.properties["collapse_connections"];
  node.properties["collapse_connections"] = collapseConnections;
}

export function getConnectionPosForLayout(
  node: TLGraphNode,
  isInput: boolean,
  slotNumber: number,
  out: Vector2,
) {
  out = out || new Float32Array(2);
  node.properties = node.properties || {};
  const layout = node.properties["connections_layout"] ||
    (node as any).defaultConnectionsLayout || ["Left", "Right"];
  const collapseConnections = (node.properties["collapse_connections"] as boolean) || false;
  const offset = (node.constructor as any).layout_slot_offset ?? LiteGraph.NODE_SLOT_HEIGHT * 0.5;
  let side = isInput ? layout[0] : layout[1];
  const otherSide = isInput ? layout[1] : layout[0];
  let data = LAYOUT_LABEL_TO_DATA[side]!; // || LAYOUT_LABEL_TO_DATA[isInput ? 'Left' : 'Right'];
  const slotList = node[isInput ? "inputs" : "outputs"];
  const cxn = slotList[slotNumber];
  if (!cxn) {
    console.log("No connection found.. weird", isInput, slotNumber);
    return out;
  }
  // Experimental; doesn't work without node.clip_area set (so it won't draw outside),
  // but litegraph.core inexplicably clips the title off which we want... so, no go.
  // if (cxn.hidden) {
  //   out[0] = node.pos[0] - 100000
  //   out[1] = node.pos[1] - 100000
  //   return out
  // }
  if (cxn.disabled) {
    // Let's store the original colors if have them and haven't yet overridden
    if (cxn.color_on !== "#666665") {
      (cxn as any)._color_on_org = (cxn as any)._color_on_org || cxn.color_on;
      (cxn as any)._color_off_org = (cxn as any)._color_off_org || cxn.color_off;
    }
    cxn.color_on = "#666665";
    cxn.color_off = "#666665";
  } else if (cxn.color_on === "#666665") {
    cxn.color_on = (cxn as any)._color_on_org || undefined;
    cxn.color_off = (cxn as any)._color_off_org || undefined;
  }
  const displaySlot = collapseConnections
    ? 0
    : slotNumber -
      slotList.reduce<number>((count, ioput, index) => {
        count += index < slotNumber && ioput.hidden ? 1 : 0;
        return count;
      }, 0);
  // Set the direction first. This is how the connection line will be drawn.
  cxn.dir = data[0];

  // If we are only 10px tall or wide, then look at connections_dir for the direction.
  const connections_dir = node.properties["connections_dir"] as [LinkDirection, LinkDirection];
  if ((node.size[0] == 10 || node.size[1] == 10) && connections_dir) {
    cxn.dir = connections_dir[isInput ? 0 : 1]!;
  }

  if (side === "Left") {
    if (node.flags.collapsed) {
      var w = (node as any)._collapsed_width || LiteGraph.NODE_COLLAPSED_WIDTH;
      out[0] = node.pos[0];
      out[1] = node.pos[1] - LiteGraph.NODE_TITLE_HEIGHT * 0.5;
    } else {
      // If we're an output, then the litegraph.core hates us; we need to blank out the name
      // because it's not flexible enough to put the text on the inside.
      toggleConnectionLabel(cxn, !isInput || collapseConnections || !!(node as any).hideSlotLabels);
      out[0] = node.pos[0] + offset;
      if ((node.constructor as any)?.type.includes("Reroute")) {
        out[1] = node.pos[1] + node.size[1] * 0.5;
      } else {
        out[1] =
          node.pos[1] +
          (displaySlot + 0.7) * LiteGraph.NODE_SLOT_HEIGHT +
          ((node.constructor as any).slot_start_y || 0);
      }
    }
  } else if (side === "Right") {
    if (node.flags.collapsed) {
      var w = (node as any)._collapsed_width || LiteGraph.NODE_COLLAPSED_WIDTH;
      out[0] = node.pos[0] + w;
      out[1] = node.pos[1] - LiteGraph.NODE_TITLE_HEIGHT * 0.5;
    } else {
      // If we're an input, then the litegraph.core hates us; we need to blank out the name
      // because it's not flexible enough to put the text on the inside.
      toggleConnectionLabel(cxn, isInput || collapseConnections || !!(node as any).hideSlotLabels);
      out[0] = node.pos[0] + node.size[0] + 1 - offset;
      if ((node.constructor as any)?.type.includes("Reroute")) {
        out[1] = node.pos[1] + node.size[1] * 0.5;
      } else {
        out[1] =
          node.pos[1] +
          (displaySlot + 0.7) * LiteGraph.NODE_SLOT_HEIGHT +
          ((node.constructor as any).slot_start_y || 0);
      }
    }

    // Right now, only reroute uses top/bottom, so this may not work for other nodes
    // (like, applying to nodes with titles, collapsed, multiple inputs/outputs, etc).
  } else if (side === "Top") {
    if (!(cxn as any).has_old_label) {
      (cxn as any).has_old_label = true;
      (cxn as any).old_label = cxn.label;
      cxn.label = " ";
    }
    out[0] = node.pos[0] + node.size[0] * 0.5;
    out[1] = node.pos[1] + offset;
  } else if (side === "Bottom") {
    if (!(cxn as any).has_old_label) {
      (cxn as any).has_old_label = true;
      (cxn as any).old_label = cxn.label;
      cxn.label = " ";
    }
    out[0] = node.pos[0] + node.size[0] * 0.5;
    out[1] = node.pos[1] + node.size[1] - offset;
  }
  return out;
}

function toggleConnectionLabel(cxn: any, hide = true) {
  if (hide) {
    if (!(cxn as any).has_old_label) {
      (cxn as any).has_old_label = true;
      (cxn as any).old_label = cxn.label;
    }
    cxn.label = " ";
  } else if (!hide && (cxn as any).has_old_label) {
    (cxn as any).has_old_label = false;
    cxn.label = (cxn as any).old_label;
    (cxn as any).old_label = undefined;
  }
  return cxn;
}

export function addHelpMenuItem(
  node: TLGraphNode,
  content: string,
  menuOptions: (IContextMenuValue | null)[],
) {
  addMenuItemOnExtraMenuOptions(
    node,
    {
      name: "🛟 Node Help",
      callback: (node) => {
        if ((node as any).showHelp) {
          (node as any).showHelp();
        } else {
          new RgthreeHelpDialog(node, content).show();
        }
      },
    },
    menuOptions,
    "Properties Panel",
  );
}

export enum PassThroughFollowing {
  ALL,
  NONE,
  REROUTE_ONLY,
}

/**
 * Determines if, when doing a chain lookup for connected nodes, we want to pass through this node,
 * like reroutes, etc.
 */
export function shouldPassThrough(
  node?: TLGraphNode | null,
  passThroughFollowing = PassThroughFollowing.ALL,
) {
  const type = (node?.constructor as typeof TLGraphNode)?.type;
  if (!type || passThroughFollowing === PassThroughFollowing.NONE) {
    return false;
  }
  if (passThroughFollowing === PassThroughFollowing.REROUTE_ONLY) {
    return type.includes("Reroute");
  }
  return (
    type.includes("Reroute") || type.includes("Node Combiner") || type.includes("Node Collector")
  );
}

function filterOutPassthroughNodes(
  infos: ConnectedNodeInfo[],
  passThroughFollowing = PassThroughFollowing.ALL,
) {
  return infos.filter((i) => !shouldPassThrough(i.node, passThroughFollowing));
}

/**
 * Looks through the immediate chain of a node to collect all connected nodes, passing through nodes
 * like reroute, etc. Will also disconnect duplicate nodes from a provided node
 */
export function getConnectedInputNodes(
  startNode: TLGraphNode,
  currentNode?: TLGraphNode,
  slot?: number,
  passThroughFollowing = PassThroughFollowing.ALL,
): TLGraphNode[] {
  return getConnectedNodesInfo(
    startNode,
    IoDirection.INPUT,
    currentNode,
    slot,
    passThroughFollowing,
  ).map((n) => n.node);
}
export function getConnectedInputInfosAndFilterPassThroughs(
  startNode: TLGraphNode,
  currentNode?: TLGraphNode,
  slot?: number,
  passThroughFollowing = PassThroughFollowing.ALL,
) {
  return filterOutPassthroughNodes(
    getConnectedNodesInfo(startNode, IoDirection.INPUT, currentNode, slot, passThroughFollowing),
    passThroughFollowing,
  );
}
export function getConnectedInputNodesAndFilterPassThroughs(
  startNode: TLGraphNode,
  currentNode?: TLGraphNode,
  slot?: number,
  passThroughFollowing = PassThroughFollowing.ALL,
): TLGraphNode[] {
  return getConnectedInputInfosAndFilterPassThroughs(
    startNode,
    currentNode,
    slot,
    passThroughFollowing,
  ).map((n) => n.node);
}

export function getConnectedOutputNodes(
  startNode: TLGraphNode,
  currentNode?: TLGraphNode,
  slot?: number,
  passThroughFollowing = PassThroughFollowing.ALL,
): TLGraphNode[] {
  return getConnectedNodesInfo(
    startNode,
    IoDirection.OUTPUT,
    currentNode,
    slot,
    passThroughFollowing,
  ).map((n) => n.node);
}

export function getConnectedOutputNodesAndFilterPassThroughs(
  startNode: TLGraphNode,
  currentNode?: TLGraphNode,
  slot?: number,
  passThroughFollowing = PassThroughFollowing.ALL,
): TLGraphNode[] {
  return filterOutPassthroughNodes(
    getConnectedNodesInfo(startNode, IoDirection.OUTPUT, currentNode, slot, passThroughFollowing),
    passThroughFollowing,
  ).map((n) => n.node);
}

export type ConnectedNodeInfo = {
  node: TLGraphNode;
  travelFromSlot: number;
  travelToSlot: number;
  originTravelFromSlot: number;
};

export function getConnectedNodesInfo(
  startNode: TLGraphNode,
  dir = IoDirection.INPUT,
  currentNode?: TLGraphNode,
  slot?: number,
  passThroughFollowing = PassThroughFollowing.ALL,
  originTravelFromSlot?: number,
): ConnectedNodeInfo[] {
  currentNode = currentNode || startNode;
  let rootNodes: ConnectedNodeInfo[] = [];
  if (startNode === currentNode || shouldPassThrough(currentNode, passThroughFollowing)) {
    let linkIds: Array<number | undefined | null>;

    slot = slot != null && slot > -1 ? slot : undefined;
    if (dir == IoDirection.OUTPUT) {
      if (slot != null) {
        linkIds = [...(currentNode.outputs?.[slot]?.links || [])];
      } else {
        linkIds = currentNode.outputs?.flatMap((i) => i.links) || [];
      }
    } else {
      if (slot != null) {
        linkIds = [currentNode.inputs?.[slot]?.link];
      } else {
        linkIds = currentNode.inputs?.map((i) => i.link) || [];
      }
    }
    const graph = currentNode.graph ?? app.graph;
    for (const linkId of linkIds) {
      let link: LLink | null = null;
      if (typeof linkId == "number") {
        link = graph.links[linkId] ?? null;
      }
      if (!link) {
        continue;
      }
      const travelFromSlot = dir == IoDirection.OUTPUT ? link.origin_slot : link.target_slot;
      const connectedId = dir == IoDirection.OUTPUT ? link.target_id : link.origin_id;
      const travelToSlot = dir == IoDirection.OUTPUT ? link.target_slot : link.origin_slot;
      originTravelFromSlot = originTravelFromSlot != null ? originTravelFromSlot : travelFromSlot;
      const originNode: TLGraphNode = graph.getNodeById(connectedId)!;
      if (!link) {
        console.error("No connected node found... weird");
        continue;
      }
      if (rootNodes.some((n) => n.node == originNode)) {
        console.log(
          `${startNode.title} (${startNode.id}) seems to have two links to ${originNode.title} (${
            originNode.id
          }). One may be stale: ${linkIds.join(", ")}`,
        );
      } else {
        // Add the node and, if it's a pass through, let's collect all its nodes as well.
        rootNodes.push({node: originNode, travelFromSlot, travelToSlot, originTravelFromSlot});
        if (shouldPassThrough(originNode, passThroughFollowing)) {
          for (const foundNode of getConnectedNodesInfo(
            startNode,
            dir,
            originNode,
            undefined,
            undefined,
            originTravelFromSlot,
          )) {
            if (!rootNodes.map((n) => n.node).includes(foundNode.node)) {
              rootNodes.push(foundNode);
            }
          }
        }
      }
    }
  }
  return rootNodes;
}

export type ConnectionType = {
  type: string | string[];
  name: string | undefined;
  label: string | undefined;
};

/**
 * Follows a connection until we find a type associated with a slot.
 * `skipSelf` skips the current slot, useful when we may have a dynamic slot that we want to start
 * from, but find a type _after_ it (in case it needs to change).
 */
export function followConnectionUntilType(
  node: TLGraphNode,
  dir: IoDirection,
  slotNum?: number,
  skipSelf = false,
): ConnectionType | null {
  const slots = dir === IoDirection.OUTPUT ? node.outputs : node.inputs;
  if (!slots || !slots.length) {
    return null;
  }
  let type: ConnectionType | null = null;
  if (slotNum) {
    if (!slots[slotNum]) {
      return null;
    }
    type = getTypeFromSlot(slots[slotNum], dir, skipSelf);
  } else {
    for (const slot of slots) {
      type = getTypeFromSlot(slot, dir, skipSelf);
      if (type) {
        break;
      }
    }
  }
  return type;
}

/**
 * Gets the type from a slot. If the type is '*' then it will follow the node to find the next slot.
 */
function getTypeFromSlot(
  slot: INodeInputSlot | INodeOutputSlot | undefined,
  dir: IoDirection,
  skipSelf = false,
): ConnectionType | null {
  let graph = app.canvas.getCurrentGraph()!;
  let type = slot?.type;
  if (!skipSelf && type != null && type != "*") {
    return {type: type as string, label: slot?.label, name: slot?.name};
  }
  const links = getSlotLinks(slot);
  for (const link of links) {
    const connectedId = dir == IoDirection.OUTPUT ? link.link.target_id : link.link.origin_id;
    const connectedSlotNum =
      dir == IoDirection.OUTPUT ? link.link.target_slot : link.link.origin_slot;
    const connectedNode: TLGraphNode = graph.getNodeById(connectedId)!;
    // Reversed since if we're traveling down the output we want the connected node's input, etc.
    const connectedSlots =
      dir === IoDirection.OUTPUT ? connectedNode.inputs : connectedNode.outputs;
    let connectedSlot = connectedSlots[connectedSlotNum];
    if (connectedSlot?.type != null && connectedSlot?.type != "*") {
      return {
        type: connectedSlot.type as string,
        label: connectedSlot?.label,
        name: connectedSlot?.name,
      };
    } else if (connectedSlot?.type == "*") {
      return followConnectionUntilType(connectedNode, dir);
    }
  }
  return null;
}

export async function replaceNode(
  existingNode: TLGraphNode,
  typeOrNewNode: string | TLGraphNode,
  inputNameMap?: Map<string, string>,
) {
  const existingCtor = existingNode.constructor as typeof TLGraphNode;

  const newNode =
    typeof typeOrNewNode === "string" ? LiteGraph.createNode(typeOrNewNode)! : typeOrNewNode;
  // Port title (maybe) the position, size, and properties from the old node.
  if (existingNode.title != existingCtor.title) {
    newNode.title = existingNode.title;
  }
  newNode.pos = [...existingNode.pos] as Point;
  newNode.properties = {...existingNode.properties};
  const oldComputeSize = [...existingNode.computeSize()];
  // oldSize to use. If we match the smallest size (computeSize) then don't record and we'll use
  // the smalles side after conversion.
  const oldSize = [
    existingNode.size[0] === oldComputeSize[0] ? null : existingNode.size[0],
    existingNode.size[1] === oldComputeSize[1] ? null : existingNode.size[1],
  ];

  let setSizeIters = 0;
  const setSizeFn = () => {
    // Size gets messed up when ComfyUI adds the text widget, so reset after a delay.
    // Since we could be adding many more slots, let's take the larger of the two.
    const newComputesize = newNode.computeSize();
    newNode.size[0] = Math.max(oldSize[0] || 0, newComputesize[0]);
    newNode.size[1] = Math.max(oldSize[1] || 0, newComputesize[1]);
    setSizeIters++;
    if (setSizeIters > 10) {
      requestAnimationFrame(setSizeFn);
    }
  };
  setSizeFn();

  // We now collect the links data, inputs and outputs, of the old node since these will be
  // lost when we remove it.
  const links: {
    node: TLGraphNode;
    slot: number | string;
    targetNode: TLGraphNode;
    targetSlot: number | string;
  }[] = [];
  const graph = existingNode.graph || app.graph;
  for (const [index, output] of existingNode.outputs.entries()) {
    for (const linkId of output.links || []) {
      const link: LLink = graph.links[linkId]!;
      if (!link) continue;
      const targetNode = graph.getNodeById(link.target_id)!;
      links.push({node: newNode, slot: output.name, targetNode, targetSlot: link.target_slot});
    }
  }
  for (const [index, input] of existingNode.inputs.entries()) {
    const linkId = input.link;
    if (linkId) {
      const link: LLink = graph.links[linkId]!;
      const originNode = graph.getNodeById(link.origin_id)!;
      links.push({
        node: originNode,
        slot: link.origin_slot,
        targetNode: newNode,
        targetSlot: inputNameMap?.has(input.name)
          ? inputNameMap.get(input.name)!
          : input.name || index,
      });
    }
  }
  // Add the new node, remove the old node.
  graph.add(newNode);
  await wait();
  // Now go through and connect the other nodes up as they were.
  for (const link of links) {
    link.node.connect(link.slot, link.targetNode, link.targetSlot);
  }
  await wait();
  graph.remove(existingNode);
  newNode.size = newNode.computeSize();
  newNode.setDirtyCanvas(true, true);
  return newNode;
}

export function getOriginNodeByLink(linkId?: number | null) {
  let node: TLGraphNode | null = null;
  if (linkId != null) {
    const link = getLinkById(linkId);
    node = (link != null && getNodeById(link.origin_id)) || null;
  }
  return node;
}

/**
 * Gets a link by id across all graphs and subgraphs.
 */
export function getLinkById(linkId?: number | null) {
  if (linkId == null) return null;
  let link: LLink | null = app.graph.links[linkId] ?? null;
  link = link ?? app.canvas.getCurrentGraph()?.links[linkId] ?? null;
  const subgraphs = app.graph.rootGraph.subgraphs.values();
  let subgraph;
  while (!link && (subgraph = subgraphs.next().value)) {
    link = subgraph?.links[linkId] ?? null;
  }
  return link;
}

/**
 * Gets a node by id across all graphs and subgraphs.
 */
export function getNodeById(id: NodeId) {
  if (id == null) return null;
  let node = app.graph.getNodeById(id);
  node = node ?? app.canvas.getCurrentGraph()?.getNodeById(id) ?? null;
  const subgraphs = app.graph.rootGraph.subgraphs.values();
  let subgraph;
  while (!node && (subgraph = subgraphs.next().value)) {
    node = subgraph?.getNodeById(id) ?? null;
  }
  return node;
}

export function applyMixins(original: Constructor<TLGraphNode>, constructors: any[]) {
  constructors.forEach((baseCtor) => {
    Object.getOwnPropertyNames(baseCtor.prototype).forEach((name) => {
      Object.defineProperty(
        original.prototype,
        name,
        Object.getOwnPropertyDescriptor(baseCtor.prototype, name) || Object.create(null),
      );
    });
  });
}

/**
 * Retruns a list of `{id: number, link: LLlink}` for a given input or output.
 *
 * Obviously, for an input, this will be a max of one.
 */
export function getSlotLinks(inputOrOutput?: INodeInputSlot | INodeOutputSlot | null) {
  const links: {id: number; link: LLink}[] = [];
  if (!inputOrOutput) {
    return links;
  }
  if ((inputOrOutput as INodeOutputSlot).links?.length) {
    const output = inputOrOutput as INodeOutputSlot;
    for (const linkId of output.links || []) {
      const link: LLink = (app.graph as LGraph).links[linkId]!;
      if (link) {
        links.push({id: linkId, link: link});
      }
    }
  }
  if ((inputOrOutput as INodeInputSlot).link) {
    const input = inputOrOutput as INodeInputSlot;
    const link: LLink = (app.graph as LGraph).links[input.link!]!;
    if (link) {
      links.push({id: input.link!, link: link});
    }
  }
  return links;
}

/**
 * Given a node, whether we're dealing with INPUTS or OUTPUTS, and the server data, re-arrange then
 * slots to match the order.
 */
export async function matchLocalSlotsToServer(
  node: TLGraphNode,
  direction: IoDirection,
  serverNodeData: ComfyNodeDef,
) {
  const serverSlotNames =
    direction == IoDirection.INPUT
      ? Object.keys(serverNodeData.input?.optional || {})
      : serverNodeData.output_name;
  const serverSlotTypes =
    direction == IoDirection.INPUT
      ? (Object.values(serverNodeData.input?.optional || {}).map((i) => i[0]) as string[])
      : serverNodeData.output;
  const slots = direction == IoDirection.INPUT ? node.inputs : node.outputs;

  // Let's go through the node data names and make sure our current ones match, and update if not.
  let firstIndex = slots.findIndex((o, i) => i !== serverSlotNames.indexOf(o.name));
  if (firstIndex > -1) {
    // Have mismatches. First, let's go through and save all our links by name.
    const links: {[key: string]: {id: number; link: LLink}[]} = {};
    slots.map((slot) => {
      // There's a chance we have duplicate names on an upgrade, so we'll collect all links to one
      // name so we don't ovewrite our list per name.
      links[slot.name] = links[slot.name] || [];
      links[slot.name]?.push(...getSlotLinks(slot));
    });

    // Now, go through and rearrange outputs by splicing
    for (const [index, serverSlotName] of serverSlotNames.entries()) {
      const currentNodeSlot = slots.map((s) => s.name).indexOf(serverSlotName);
      if (currentNodeSlot > -1) {
        if (currentNodeSlot != index) {
          const splicedItem = slots.splice(currentNodeSlot, 1)[0]!;
          slots.splice(index, 0, splicedItem as any);
        }
      } else if (currentNodeSlot === -1) {
        const splicedItem = {
          name: serverSlotName,
          type: serverSlotTypes![index],
          links: [],
        };
        slots.splice(index, 0, splicedItem as any);
      }
    }

    if (slots.length > serverSlotNames.length) {
      for (let i = slots.length - 1; i > serverSlotNames.length - 1; i--) {
        if (direction == IoDirection.INPUT) {
          node.disconnectInput(i);
          node.removeInput(i);
        } else {
          node.disconnectOutput(i);
          node.removeOutput(i);
        }
      }
    }

    // Now, go through the link data again and make sure the origin_slot is the correct slot.
    for (const [name, slotLinks] of Object.entries(links)) {
      let currentNodeSlot = slots.map((s) => s.name).indexOf(name);
      if (currentNodeSlot > -1) {
        for (const linkData of slotLinks) {
          if (direction == IoDirection.INPUT) {
            linkData.link.target_slot = currentNodeSlot;
          } else {
            linkData.link.origin_slot = currentNodeSlot;
            // If our next node is a Reroute, then let's get it to update the type.
            const nextNode = app.graph.getNodeById(linkData.link.target_id);
            // (Check nextNode, as sometimes graphs seem to have very stale data and that node id
            //  doesn't exist).
            if (nextNode && (nextNode.constructor as any)?.type!.includes("Reroute")) {
              (nextNode as any).stabilize && (nextNode as any).stabilize();
            }
          }
        }
      }
    }
  }
}

export function isValidConnection(ioA?: INodeSlot | null, ioB?: INodeSlot | null) {
  if (!ioA || !ioB) {
    return false;
  }
  const typeA = String(ioA.type);
  const typeB = String(ioB.type);
  // What does litegraph think, which includes looking at array values.
  let isValid = LiteGraph.isValidConnection(typeA, typeB);

  // This is here to fix the churn happening in list types in comfyui itself..
  // https://github.com/comfyanonymous/ComfyUI/issues/1674
  if (!isValid) {
    let areCombos =
      (typeA.includes(",") && typeB === "COMBO") || (typeA === "COMBO" && typeB.includes(","));
    // We don't want to let any old combo connect to any old combo, so we'll look at the names too.
    if (areCombos) {
      // Some nodes use "_name" and some use "model" and "ckpt", so normalize
      const nameA = ioA.name.toUpperCase().replace("_NAME", "").replace("CKPT", "MODEL");
      const nameB = ioB.name.toUpperCase().replace("_NAME", "").replace("CKPT", "MODEL");
      isValid = nameA.includes(nameB) || nameB.includes(nameA);
    }
  }
  return isValid;
}

/**
 * Patches the LiteGraph.isValidConnection so old nodes can connect to this new COMBO type for all
 * lists (without users needing to go through and re-create all their nodes one by one).
 */
const oldIsValidConnection = LiteGraph.isValidConnection;
LiteGraph.isValidConnection = function (typeA: ISlotType, typeB: ISlotType): boolean {
  let isValid = oldIsValidConnection.call(LiteGraph, typeA, typeB);
  if (!isValid) {
    typeA = String(typeA);
    typeB = String(typeB);
    // This is waaaay too liberal and now any combos can connect to any combos. But we only have the
    // types (not names like my util above), and connecting too liberally is better than old nodes
    // with lists not being able to connect to this new COMBO type. And, anyway, it matches the
    // current behavior today with new nodes anyway, where all lists are COMBO types.
    // Refs: https://github.com/comfyanonymous/ComfyUI/issues/1674
    //       https://github.com/comfyanonymous/ComfyUI/pull/1675
    let areCombos =
      (typeA.includes(",") && typeB === "COMBO") || (typeA === "COMBO" && typeB.includes(","));
    isValid = areCombos;
  }
  return isValid;
};

/**
 * Returns a list of output nodes given a list of nodes.
 */
export function getOutputNodes(nodes: TLGraphNode[]) {
  return (
    nodes?.filter((n) => {
      return (
        n.mode != LiteGraph.NEVER && ((n.constructor as any).nodeData as ComfyNodeDef)?.output_node
      );
    }) || []
  );
}

/**
 * Changes the mode of a node. We must go through this to change a node's mode after the
 * introduction of subgraphs, as ComfyUI doesn't update the mode of a node in a subgraph on its own.
 */
export function changeModeOfNodes(nodeOrNodes: TLGraphNode | TLGraphNode[], mode: LGraphEventMode) {
  reduceNodesDepthFirst(nodeOrNodes, (n) => {
    n.mode = mode;
  });
}

/**
 * Performs depth-first traversal of nodes and their subgraphs.
 * Adapted from ComfyUI Frontend's method.
 */
export function reduceNodesDepthFirst(
  nodeOrNodes: TLGraphNode | TLGraphNode[],
  reduceFn: (node: TLGraphNode) => void,
): void;
export function reduceNodesDepthFirst<T>(
  nodeOrNodes: TLGraphNode | TLGraphNode[],
  reduceFn: (node: TLGraphNode, reduceTo: T) => T | void,
  reduceTo: T,
): T;
export function reduceNodesDepthFirst<T>(
  nodeOrNodes: TLGraphNode | TLGraphNode[],
  reduceFn: (node: TLGraphNode, reduceTo: T) => T | void,
  reduceTo?: T,
): T {
  const nodes = Array.isArray(nodeOrNodes) ? nodeOrNodes : [nodeOrNodes];
  const stack: Array<{node: TLGraphNode}> = nodes.map((node) => ({node}));

  // Process stack iteratively (DFS)
  while (stack.length > 0) {
    const {node} = stack.pop()!;
    const result = reduceFn(node, reduceTo as T);
    if (result !== undefined && result !== reduceTo) {
      reduceTo = result;
    }

    // If it's a subgraph and we should expand, add children to stack
    if (node.isSubgraphNode?.() && node.subgraph) {
      // Process children in reverse order to maintain left-to-right DFS processing
      // when popping from stack (LIFO). Iterate backwards to avoid array reversal.
      const children = node.subgraph.nodes;
      for (let i = children.length - 1; i >= 0; i--) {
        stack.push({node: children[i]!});
      }
    }
  }
  return reduceTo as T;
}

/**
 * Found an issue where group._nodes had nodes that weren't in the actual group. group._nodes is
 * marked deprecated, so we'll go ahead and use _children and filter.
 */
export function getGroupNodes(group: LGraphGroup): TLGraphNode[] {
  return Array.from(group._children).filter((c) => c instanceof LGraphNode);
}

/**
 * Gets a full color string, including parsing from the LGraphCanvas data.
 */
export function getFullColor(
  color?: string,
  liteGraphKey: "groupcolor" | "color" | "bgcolor" = "color",
) {
  if (!color) {
    return "";
  }
  if (LGraphCanvas.node_colors[color]) {
    color = LGraphCanvas.node_colors[color]![liteGraphKey];
  }
  color = color.replace("#", "").toLocaleLowerCase();
  if (color.length === 3) {
    color = color.replace(/(.)(.)(.)/, "$1$1$2$2$3$3");
  }
  return `#${color}`;
}
