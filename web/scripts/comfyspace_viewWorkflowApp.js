
import { ComfyApp } from "./app.js";

export class ComfyViewWorkflowApp extends ComfyApp {
  #defs = null;
  #workflow = null

  async setup() {
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
        this.#defs = JSON.parse(workflowVer.nodeDefs);
        this.#workflow = JSON.parse(workflowVer.json);
      })
      .catch((error) => {
        console.error(error);
      });

    await super.setup();
    await this.loadGraphData(this.#workflow);
  }
  async registerNodes() {
    console.log("registerNodes",  this.#defs,'workflow', this.#workflow);
      
      await this.registerNodesFromDefs(this.#defs);
      await this.#invokeExtensionsAsync("registerCustomNodes");
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