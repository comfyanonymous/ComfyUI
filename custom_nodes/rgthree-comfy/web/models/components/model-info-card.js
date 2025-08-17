import { RgthreeCustomElement } from "../../common/components/base_custom_element.js";
export class RgthreeModelInfoCard extends RgthreeCustomElement {
    constructor() {
        super(...arguments);
        this.data = {};
    }
    getModified(value, data, currentElement, contextElement) {
        const date = new Date(value);
        return String(`${date.toLocaleDateString()} ${date.toLocaleTimeString()}`);
    }
    getCivitaiLink(links) {
        return (links === null || links === void 0 ? void 0 : links.find((i) => i.includes("civitai.com/models"))) || null;
    }
    setModelData(data) {
        this.data = data;
    }
    hasBaseModel(baseModel) {
        return this.data.baseModel === baseModel;
    }
    hasData(field) {
        var _a;
        if (field === "civitai") {
            return !!((_a = this.getCivitaiLink(this.data.links)) === null || _a === void 0 ? void 0 : _a.length);
        }
        return !!this.data[field];
    }
    matchesQueryText(query) {
        var _a;
        return (_a = (this.data.name || this.data.file)) === null || _a === void 0 ? void 0 : _a.includes(query);
    }
    hide() {
        this.classList.add("-is-hidden");
    }
    show() {
        this.classList.remove("-is-hidden");
    }
}
RgthreeModelInfoCard.NAME = "rgthree-model-info-card";
RgthreeModelInfoCard.TEMPLATES = "components/model-info-card.html";
RgthreeModelInfoCard.CSS = "components/model-info-card.css";
RgthreeModelInfoCard.USE_SHADOW = false;
