import type {RgthreeModelInfo} from "typings/rgthree.js";

export type ModelInfoType = "loras" | "checkpoints";

type ModelsOptions = {
  type: ModelInfoType;
  files?: string[];
};

type GetModelsOptions = ModelsOptions & {
  type: ModelInfoType;
  files?: string[];
  format?: null | "details";
};

type GetModelsInfoOptions = GetModelsOptions & {
  light?: boolean;
};

type GetModelsResponseDetails = {
  file: string;
  modified: number;
  has_info: boolean;
  image?: string;
};

class RgthreeApi {
  private baseUrl!: string;
  private comfyBaseUrl!: string;
  getCheckpointsPromise: Promise<string[]> | null = null;
  getSamplersPromise: Promise<string[]> | null = null;
  getSchedulersPromise: Promise<string[]> | null = null;
  getLorasPromise: Promise<GetModelsResponseDetails[]> | null = null;
  getWorkflowsPromise: Promise<string[]> | null = null;

  constructor(baseUrl?: string) {
    this.setBaseUrl(baseUrl);
  }

  setBaseUrl(baseUrlArg?: string) {
    let baseUrl = null;
    if (baseUrlArg) {
      baseUrl = baseUrlArg;
    } else if (window.location.pathname.includes("/rgthree/")) {
      // Try to find how many relatives paths we need to go back to hit ./rgthree/api
      const parts = window.location.pathname.split("/rgthree/")[1]?.split("/");
      if (parts && parts.length) {
        baseUrl = parts.map(() => "../").join("") + "rgthree/api";
      }
    }
    this.baseUrl = baseUrl || "./rgthree/api";

    // Calculate the comfyUI api base path by checkin gif we're on an rgthree independant page (as
    // we'll always use '/rgthree/' prefix) and, if so, assume the path before `/rgthree/` is the
    // base path. If we're not, then just use the same pathname logic as the ComfyUI api.js uses.
    const comfyBasePathname = location.pathname.includes("/rgthree/")
      ? location.pathname.split("rgthree/")[0]!
      : location.pathname;
    this.comfyBaseUrl = comfyBasePathname.split("/").slice(0, -1).join("/");
  }

  apiURL(route: string) {
    return `${this.baseUrl}${route}`;
  }

  fetchApi(route: string, options?: RequestInit) {
    return fetch(this.apiURL(route), options);
  }

  async fetchJson(route: string, options?: RequestInit) {
    const r = await this.fetchApi(route, options);
    return await r.json();
  }

  async postJson(route: string, json: any) {
    const body = new FormData();
    body.append("json", JSON.stringify(json));
    return await rgthreeApi.fetchJson(route, {method: "POST", body});
  }

  getLoras(force = false) {
    if (!this.getLorasPromise || force) {
      this.getLorasPromise = this.fetchJson("/loras?format=details", {cache: "no-store"});
    }
    return this.getLorasPromise;
  }

  async fetchApiJsonOrNull<T>(route: string, options?: RequestInit) {
    const response = await this.fetchJson(route, options);
    if (response.status === 200 && response.data) {
      return (response.data as T) || null;
    }
    return null;
  }

  /**
   * Fetches the lora information.
   *
   * @param light Whether or not to generate a json file if there isn't one. This isn't necessary if
   * we're just checking for values, but is more necessary when opening an info dialog.
   */

  async getModelsInfo(options: GetModelsInfoOptions): Promise<RgthreeModelInfo[]> {
    const params = new URLSearchParams();
    if (options.files?.length) {
      params.set("files", options.files.join(","));
    }
    if (options.light) {
      params.set("light", "1");
    }
    if (options.format) {
      params.set("format", options.format);
    }
    const path = `/${options.type}/info?` + params.toString();
    return (await this.fetchApiJsonOrNull<RgthreeModelInfo[]>(path)) || [];
  }
  async getLorasInfo(options: Omit<GetModelsInfoOptions, "type"> = {}) {
    return this.getModelsInfo({type: "loras", ...options});
  }
  async getCheckpointsInfo(options: Omit<GetModelsInfoOptions, "type"> = {}) {
    return this.getModelsInfo({type: "checkpoints", ...options});
  }

  async refreshModelsInfo(options: ModelsOptions) {
    const params = new URLSearchParams();
    if (options.files?.length) {
      params.set("files", options.files.join(","));
    }
    const path = `/${options.type}/info/refresh?` + params.toString();
    const infos = await this.fetchApiJsonOrNull<RgthreeModelInfo[]>(path);
    return infos;
  }
  async refreshLorasInfo(options: Omit<ModelsOptions, "type"> = {}) {
    return this.refreshModelsInfo({type: "loras", ...options});
  }
  async refreshCheckpointsInfo(options: Omit<ModelsOptions, "type"> = {}) {
    return this.refreshModelsInfo({type: "checkpoints", ...options});
  }

  async clearModelsInfo(options: ModelsOptions) {
    const params = new URLSearchParams();
    if (options.files?.length) {
      // encodeURIComponent ?
      params.set("files", options.files.join(","));
    }
    const path = `/${options.type}/info/clear?` + params.toString();
    await this.fetchApiJsonOrNull<RgthreeModelInfo[]>(path);
    return;
  }
  async clearLorasInfo(options: Omit<ModelsOptions, "type"> = {}) {
    return this.clearModelsInfo({type: "loras", ...options});
  }
  async clearCheckpointsInfo(options: Omit<ModelsOptions, "type"> = {}) {
    return this.clearModelsInfo({type: "checkpoints", ...options});
  }

  /**
   * Saves partial data sending it to the backend..
   */
  async saveModelInfo(
    type: ModelInfoType,
    file: string,
    data: Partial<RgthreeModelInfo>,
  ): Promise<RgthreeModelInfo | null> {
    const body = new FormData();
    body.append("json", JSON.stringify(data));
    return await this.fetchApiJsonOrNull<RgthreeModelInfo>(
      `/${type}/info?file=${encodeURIComponent(file)}`,
      {cache: "no-store", method: "POST", body},
    );
  }

  async saveLoraInfo(
    file: string,
    data: Partial<RgthreeModelInfo>,
  ): Promise<RgthreeModelInfo | null> {
    return this.saveModelInfo("loras", file, data);
  }

  async saveCheckpointsInfo(
    file: string,
    data: Partial<RgthreeModelInfo>,
  ): Promise<RgthreeModelInfo | null> {
    return this.saveModelInfo("checkpoints", file, data);
  }

  /**
   * [🤮] Fetches from the ComfyUI given a similar functionality to the real ComfyUI API
   * implementation, but can be available on independant pages outside of the ComfyUI UI. This is
   * because ComfyUI frontend stopped serving its modules independantly and opted for a giant bundle
   * instead which no longer allows us to load its `api.js` file separately.
   */
  fetchComfyApi(route: string, options?: any): Promise<any> {
    const url = this.comfyBaseUrl + "/api" + route;
    options = options || {};
    options.headers = options.headers || {};
    options.cache = options.cache || "no-cache";
    return fetch(url, options);
  }
}

export const rgthreeApi = new RgthreeApi();
