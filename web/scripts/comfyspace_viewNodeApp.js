
import { ComfyApp } from "./app.js";

export class ComfyViewNodeApp extends ComfyApp {
  async setup() {
    LGraphCanvas.prototype.processMouseWheel =()=>{}
    await fetch("/api/listComfyExtensions").then(resp => resp.json()).then(data => {
      if(data.paths) {
        console.log("data.paths", data.paths);
        this.extensionFilesPath = data.paths;
      }
    }).catch(error => {
      console.error("error fetching comfy extensions", error);
    }); 
    await super.setup();
  }
  async registerNodes() {
      const app = this;
      // Load node definitions from the backend
      // const defs = await api.getNodeDefs();
      const queryParams = new URLSearchParams(window.location.search);
    const nodeDefs = queryParams.get('nodeDefs');
      const defs = JSON.parse(nodeDefs);

      await this.registerNodesFromDefs(defs);
      await this.#invokeExtensionsAsync("registerCustomNodes");
      
      
      let currentPosition = [20, 50]; // Start at the top-left corner of the canvas.
      const canvasWidth = app.canvasEl.offsetWidth; // Dynamically get canvas width.
      const rowGap = 60; // Vertical gap between rows.
      const gap = 20;
      let maxHeightInRow = 0; // Track the tallest node in the current row.
      Object.keys(defs).forEach((nodeType, index) => {
        const node = LiteGraph.createNode(nodeType);
        const nodeWidth = node.size[0];
        const nodeHeight = node.size[1];
      
        // Check if the node can fit in the canvas width at all
        if (nodeWidth > canvasWidth) {
          console.warn(`Node of type ${nodeType} exceeds the canvas width and cannot be placed.`);
        }
      
        // Preemptively move to the next row if the current node would exceed the canvas width
        if (currentPosition[0] + nodeWidth + gap > canvasWidth) {
          currentPosition[0] = 0; // Reset X position to start of the next row
          currentPosition[1] += maxHeightInRow + rowGap; // Move Y position down
          maxHeightInRow = nodeHeight; // Start tracking the new row's maxHeight with the current node
        } else {
          // The node fits in the current row, update maxHeightInRow
          maxHeightInRow = Math.max(maxHeightInRow, nodeHeight);
        }
      
        // Place the node at the current position
        node.pos = [...currentPosition];
        app.graph.add(node);
      
        // Move currentPosition right for the next node
        currentPosition[0] += nodeWidth + gap;
  
      });
      // After adding and positioning all nodes
      const totalHeightRequired = currentPosition[1] + maxHeightInRow + rowGap; // Add one more rowGap for bottom padding
      var message = { type: 'updateCanvasHeight', height: totalHeightRequired };
      // Send the message to the parent window
      window.parent.postMessage(message, window.location.origin); 
    }
  /**
   * Invoke an async extension callback
   * Each callback will be invoked concurrently
   * @param {string} method The extension callback to execute
   * @param  {...any} args Any arguments to pass to the callback
   * @returns
   */
  async #invokeExtensionsAsync(method, ...args) {
      return await Promise.all(
      this.extensions.map(async (ext) => {
          if (method in ext) {
          try {
              return await ext[method](...args, this);
          } catch (error) {
              console.error(
              `Error calling extension '${ext.name}' method '${method}'`,
              { error },
              { extension: ext },
              { args }
              );
          }
          }
      })
      );
  }

}