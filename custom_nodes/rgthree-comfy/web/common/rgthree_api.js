class RgthreeApi {
    constructor(baseUrl) {
        this.getCheckpointsPromise = null;
        this.getSamplersPromise = null;
        this.getSchedulersPromise = null;
        this.getLorasPromise = null;
        this.getWorkflowsPromise = null;
        this.setBaseUrl(baseUrl);
    }
    setBaseUrl(baseUrlArg) {
        var _a;
        let baseUrl = null;
        if (baseUrlArg) {
            baseUrl = baseUrlArg;
        }
        else if (window.location.pathname.includes("/rgthree/")) {
            const parts = (_a = window.location.pathname.split("/rgthree/")[1]) === null || _a === void 0 ? void 0 : _a.split("/");
            if (parts && parts.length) {
                baseUrl = parts.map(() => "../").join("") + "rgthree/api";
            }
        }
        this.baseUrl = baseUrl || "./rgthree/api";
        const comfyBasePathname = location.pathname.includes("/rgthree/")
            ? location.pathname.split("rgthree/")[0]
            : location.pathname;
        this.comfyBaseUrl = comfyBasePathname.split("/").slice(0, -1).join("/");
    }
    apiURL(route) {
        return `${this.baseUrl}${route}`;
    }
    fetchApi(route, options) {
        return fetch(this.apiURL(route), options);
    }
    async fetchJson(route, options) {
        const r = await this.fetchApi(route, options);
        return await r.json();
    }
    async postJson(route, json) {
        const body = new FormData();
        body.append("json", JSON.stringify(json));
        return await rgthreeApi.fetchJson(route, { method: "POST", body });
    }
    getLoras(force = false) {
        if (!this.getLorasPromise || force) {
            this.getLorasPromise = this.fetchJson("/loras?format=details", { cache: "no-store" });
        }
        return this.getLorasPromise;
    }
    async fetchApiJsonOrNull(route, options) {
        const response = await this.fetchJson(route, options);
        if (response.status === 200 && response.data) {
            return response.data || null;
        }
        return null;
    }
    async getModelsInfo(options) {
        var _a;
        const params = new URLSearchParams();
        if ((_a = options.files) === null || _a === void 0 ? void 0 : _a.length) {
            params.set("files", options.files.join(","));
        }
        if (options.light) {
            params.set("light", "1");
        }
        if (options.format) {
            params.set("format", options.format);
        }
        const path = `/${options.type}/info?` + params.toString();
        return (await this.fetchApiJsonOrNull(path)) || [];
    }
    async getLorasInfo(options = {}) {
        return this.getModelsInfo({ type: "loras", ...options });
    }
    async getCheckpointsInfo(options = {}) {
        return this.getModelsInfo({ type: "checkpoints", ...options });
    }
    async refreshModelsInfo(options) {
        var _a;
        const params = new URLSearchParams();
        if ((_a = options.files) === null || _a === void 0 ? void 0 : _a.length) {
            params.set("files", options.files.join(","));
        }
        const path = `/${options.type}/info/refresh?` + params.toString();
        const infos = await this.fetchApiJsonOrNull(path);
        return infos;
    }
    async refreshLorasInfo(options = {}) {
        return this.refreshModelsInfo({ type: "loras", ...options });
    }
    async refreshCheckpointsInfo(options = {}) {
        return this.refreshModelsInfo({ type: "checkpoints", ...options });
    }
    async clearModelsInfo(options) {
        var _a;
        const params = new URLSearchParams();
        if ((_a = options.files) === null || _a === void 0 ? void 0 : _a.length) {
            params.set("files", options.files.join(","));
        }
        const path = `/${options.type}/info/clear?` + params.toString();
        await this.fetchApiJsonOrNull(path);
        return;
    }
    async clearLorasInfo(options = {}) {
        return this.clearModelsInfo({ type: "loras", ...options });
    }
    async clearCheckpointsInfo(options = {}) {
        return this.clearModelsInfo({ type: "checkpoints", ...options });
    }
    async saveModelInfo(type, file, data) {
        const body = new FormData();
        body.append("json", JSON.stringify(data));
        return await this.fetchApiJsonOrNull(`/${type}/info?file=${encodeURIComponent(file)}`, { cache: "no-store", method: "POST", body });
    }
    async saveLoraInfo(file, data) {
        return this.saveModelInfo("loras", file, data);
    }
    async saveCheckpointsInfo(file, data) {
        return this.saveModelInfo("checkpoints", file, data);
    }
    fetchComfyApi(route, options) {
        const url = this.comfyBaseUrl + "/api" + route;
        options = options || {};
        options.headers = options.headers || {};
        options.cache = options.cache || "no-cache";
        return fetch(url, options);
    }
}
export const rgthreeApi = new RgthreeApi();
