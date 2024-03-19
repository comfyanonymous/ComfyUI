
import { ComfyApp } from "./app.js";
const COMFYUI_CORE_EXTENSIONS = [
  // "/extensions/core/clipspace.js",
  "/extensions/core/colorPalette.js",
  // "/extensions/core/contextMenuFilter.js",
  // "/extensions/core/dynamicPrompts.js",
  // "/extensions/core/editAttention.js",
  // "/extensions/core/groupNode.js",
  // "/extensions/core/groupNodeManage.js",
  // "/extensions/core/groupOptions.js",
  // "/extensions/core/invertMenuScrolling.js",
  // "/extensions/core/keybinds.js",
  // "/extensions/core/linkRenderMode.js",
  "/extensions/core/maskeditor.js",
  // "/extensions/core/nodeTemplates.js",
  // "/extensions/core/noteNode.js",
  // "/extensions/core/rerouteNode.js",
  "/extensions/core/saveImageExtraOutput.js",
  "/extensions/core/slotDefaults.js",
  "/extensions/core/snapToGrid.js",
  "/extensions/core/undoRedo.js",
  "/extensions/core/uploadImage.js",
  "/extensions/core/widgetInputs.js",
  "/extensions/dp.js",
]
export class ComfyViewNodePackageApp extends ComfyApp {
  /** @type {{nodeDefs:string,jsFilePaths:string}} */
  nodePackage = null;
  pacakgeID = null;
  nodeType= null;
  extensionFilesPath =  COMFYUI_CORE_EXTENSIONS;
  canvasHeightRequired = 500;
  constructor() {
    super();
    const params = new URLSearchParams(window.location.search);
    this.pacakgeID = params.get("packageID").replaceAll("/", "_");
    this.nodeType = params.get('node');
  }
  async setupMouseWheel() {
    LGraphCanvas.prototype.processMouseWheel = function(e) {
      if (!this.graph || !this.allow_dragcanvas) {
          return;
      }
  
      // Get the wheel delta
      var delta = e.wheelDeltaY != null ? e.wheelDeltaY : e.detail * -60;
  
      // Scaling factor to adjust the sensitivity, might need tuning based on testing
      var sensitivity = 0.5;
  
      // Apply the delta directly after scaling for sensitivity adjustment
      // this.ds.offset[1] -= delta * sensitivity;
      this.ds.offset[1] += delta * sensitivity;
  
      // Mark the canvas as needing a redraw
      this.dirty_canvas = true;
      this.dirty_bgcanvas = true;
  
      e.preventDefault();
      return false; // Prevent the default scrolling behavior of the browser
  };
  
  
  }
  async setup() {
    // to disable mousewheel zooming
    // LGraphCanvas.prototype.processMouseWheel =()=>{}
    this.setupMouseWheel();
    LGraphCanvas.prototype.processMouseMove = ()=>{}
    // LGraphCanvas.prototype.processMouseDown = ()=>{}
    if(this.pacakgeID) {
      try {
        const resp = await fetch("https://www.nodecafe.org/api/getNodePackage?id="+this.pacakgeID);
        this.nodePackage = (await resp.json())?.data;
        this.nodeDefs = JSON.parse(this.nodePackage.nodeDefs??"{}"); 
      } catch (error) {
        console.error("Error fetching node package", error);
      }
    }
    await this.loadPackageExtensions();
    await super.setup();
    await this.addNodesToGraph();
    this.canvasEl.addEventListener("click", (e)=> {
			var node = app.graph.getNodeOnPos( e.clientX, e.clientY, app.graph._nodes, 5 );
			window.parent.postMessage({ type: "onClickNodeEvent", nodeType: node.type }, window.location.origin);
		});
		this.canvasEl.addEventListener("mousemove", (e)=> {
			var node = app.graph.getNodeOnPos( e.clientX, e.clientY, app.graph._nodes, 5 );		
			if(node) {
				app.canvasEl.style.cursor = "pointer";
			} else {
				app.canvasEl.style.cursor = "default";
			}
		});
  }
  async loadPackageExtensions() {
    const jsFilePaths = JSON.parse(this.nodePackage?.jsFilePaths || "[]");
    this.extensionFilesPath = this.extensionFilesPath.concat(jsFilePaths.map(ext=>"/extensions/"+this.pacakgeID+"/"+ext));
}
  async addNodesToGraph() {
    window.addEventListener('message', function(event) {
      // Check the message received
      if (event.data === 'showAllNodes') {
        // Handle the "showAllNodes" message
        console.log('Handling showAllNodes message');
    
        // Your code to show all nodes goes here
        // For example, this could involve altering the display style of certain elements
      }
    });
	  const LEFT_PADDING = 20;
    let currentPosition = [LEFT_PADDING, 50]; // Start at the top-left corner of the canvas.
    const canvasWidth = this.canvasEl.offsetWidth; // Dynamically get canvas width.
    const rowGap = 60; // Vertical gap between rows.
    const gap = 20;
    let maxHeightInRow = 0; // Track the tallest node in the current row.
    Object.keys(this.nodeDefs).forEach((nodeType, index) => {
      if(this.nodeType && this.nodeType !== nodeType) {
        return;
      }
      try{
        const node = LiteGraph.createNode(nodeType);
        const nodeWidth = node.size[0];
        const nodeHeight = node.size[1];
      
        // Check if the node can fit in the canvas width at all
        if (nodeWidth > canvasWidth) {
          console.warn(`Node of type ${nodeType} exceeds the canvas width and cannot be placed.`);
        }
      
        // Preemptively move to the next row if the current node would exceed the canvas width
        if (currentPosition[0] + nodeWidth + gap > canvasWidth) {
          currentPosition[0] = LEFT_PADDING; // Reset X position to start of the next row
          currentPosition[1] += maxHeightInRow + rowGap; // Move Y position down
          maxHeightInRow = nodeHeight; // Start tracking the new row's maxHeight with the current node
        } else {
          // The node fits in the current row, update maxHeightInRow
          maxHeightInRow = Math.max(maxHeightInRow, nodeHeight);
        }
      
        // Place the node at the current position
        node.pos = [...currentPosition];
        this.graph.add(node);
      
        // Move currentPosition right for the next node
        currentPosition[0] += nodeWidth + gap;

        // After adding and positioning all nodes
        const totalHeightRequired = currentPosition[1] + maxHeightInRow + rowGap; // Add one more rowGap for bottom padding
        var message = { type: 'updateCanvasHeight', height: totalHeightRequired };
        this.graph.canvasHeightRequired = totalHeightRequired;
        // Send the message to the parent window
        window.parent.postMessage(message, window.location.origin); 
        // Adjust canvas height to fit all nodes
      } catch(e) {
        console.error("Error adding node", nodeType, e);
      }
    });
  }
}