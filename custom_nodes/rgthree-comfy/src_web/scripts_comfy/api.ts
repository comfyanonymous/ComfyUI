interface ComfyApi extends EventTarget {
  api_base: string;
  getNodeDefs(): any;
  apiURL(url: string): string;
  queuePrompt(num: number, data: { output: {}; workflow: {} }): Promise<{}>;
  fetchApi(route: string, options?: RequestInit) : Promise<Response>;
  interrupt(): void;
}

export declare const api: ComfyApi;
