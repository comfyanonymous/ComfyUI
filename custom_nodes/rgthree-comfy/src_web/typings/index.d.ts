/**
 * Declare any global properties and import our other typings here.
 */
import "@comfyorg/frontend";
import "./litegraph";
import "./rgthree";
import "./comfy";

import type {
  LGraphGroup as TLGraphGroup,
  LGraphNode as TLGraphNode,
  LGraph as TLGraph,
  LGraphCanvas as TLGraphCanvas,
  LiteGraph as TLiteGraph,
} from "@comfyorg/frontend";

declare global {
  const LiteGraph: typeof TLiteGraph;
  const LGraph: typeof TLGraph;
  const LGraphNode: typeof TLGraphNode;
  const LGraphCanvas: typeof TLGraphCanvas;
  const LGraphGroup: typeof TLGraphGroup;
  interface Window {
    // Used in the common/comfyui_shim to determine if we're in the app or not.
    comfyAPI: {
      // So much more stuffed in here, add as needed.
      [key: string]: any;
    };
  }
}
