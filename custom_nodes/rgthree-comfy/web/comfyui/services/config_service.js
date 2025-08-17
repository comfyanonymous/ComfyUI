import { rgthreeConfig } from "../../../rgthree/config.js";
import { getObjectValue, setObjectValue } from "../../../rgthree/common/shared_utils.js";
import { rgthreeApi } from "../../../rgthree/common/rgthree_api.js";
class ConfigService extends EventTarget {
    getConfigValue(key, def) {
        return getObjectValue(rgthreeConfig, key, def);
    }
    getFeatureValue(key, def) {
        key = "features." + key.replace(/^features\./, "");
        return getObjectValue(rgthreeConfig, key, def);
    }
    async setConfigValues(changed) {
        const body = new FormData();
        body.append("json", JSON.stringify(changed));
        const response = await rgthreeApi.fetchJson("/config", { method: "POST", body });
        if (response.status === "ok") {
            for (const [key, value] of Object.entries(changed)) {
                setObjectValue(rgthreeConfig, key, value);
                this.dispatchEvent(new CustomEvent("config-change", { detail: { key, value } }));
            }
        }
        else {
            return false;
        }
        return true;
    }
}
export const SERVICE = new ConfigService();
