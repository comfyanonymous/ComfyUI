
import { ComfyApp } from "./app.js";

const COMFYUI_CORE_EXTENSIONS = [
  // "/extensions/core/clipspace.js",
  "/extensions/core/colorPalette.js",
  // "/extensions/core/contextMenuFilter.js",
  // "/extensions/core/dynamicPrompts.js",
  // "/extensions/core/editAttention.js",
  "/extensions/core/groupNode.js",
  "/extensions/core/groupNodeManage.js",
  "/extensions/core/groupOptions.js",
  // "/extensions/core/invertMenuScrolling.js",
  // "/extensions/core/keybinds.js",
  // "/extensions/core/linkRenderMode.js",
  "/extensions/core/maskeditor.js",
  // "/extensions/core/nodeTemplates.js",
  "/extensions/core/noteNode.js",
  "/extensions/core/rerouteNode.js",
  "/extensions/core/saveImageExtraOutput.js",
  "/extensions/core/slotDefaults.js",
  "/extensions/core/snapToGrid.js",
  // "/extensions/core/undoRedo.js",
  "/extensions/core/uploadImage.js",
  "/extensions/core/widgetInputs.js",
  "/extensions/dp.js",
]

export class ComfyViewWorkflowApp extends ComfyApp {
  #workflow = null
  
  async setup() {
    this.extensionFilesPath = COMFYUI_CORE_EXTENSIONS;

    const queryParams = new URLSearchParams(window.location.search);
    const workflowVersionID = queryParams.get('workflowVersionID');
    await fetch("/api/getCloudflowVersion/?id=" + workflowVersionID, {
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        console.log("getCloudflowVersion data", data);
        const workflowVer = data.data;
        this.nodeDefs = JSON.parse(workflowVer.nodeDefs);
        this.#workflow = JSON.parse(workflowVer.json);
      })
      .catch((error) => {
        console.error(error);
      });

    await super.setup();
    await this.loadGraphData(this.#workflow);
  }
}