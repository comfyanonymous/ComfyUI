import { createElement, getActionEls, query, queryAll } from "../common/utils_dom.js";
import { rgthreeApi } from "../common/rgthree_api.js";
import { RgthreeModelInfoCard } from "./components/model-info-card.js";
function parseQuery(query) {
    const matches = query.match(/"[^\"]+"/g) || [];
    for (const match of matches) {
        let cleaned = match.substring(1, match.length - 1);
        cleaned = cleaned.replace(/\s+/g, " ").trim().replace(/\s/g, "__SPACE__");
        query = query.replace(match, ` ${cleaned} `);
    }
    const queryParts = query
        .replace(/\s+/g, " ")
        .trim()
        .split(" ")
        .map((p) => p.replace(/__SPACE__/g, " "));
    return queryParts;
}
export class ModelsInfoPage {
    constructor() {
        this.selectBaseModel = createElement('select[name="baseModel"][on="change:filter"]');
        this.searchbox = query("#searchbox");
        this.modelsList = query("#models-list");
        this.queryLast = "";
        this.doSearchDebounce = 0;
        console.log("hello model page");
        this.init();
    }
    async init() {
        this.searchbox.addEventListener("input", (e) => {
            if (this.doSearchDebounce) {
                return;
            }
            this.doSearchDebounce = setTimeout(() => {
                this.doSearch();
                this.doSearchDebounce = 0;
            }, 250);
        });
        const loras = await rgthreeApi.getLorasInfo({ light: true });
        console.log(loras);
        const baseModels = new Set();
        for (const lora of loras) {
            const el = RgthreeModelInfoCard.create();
            el.setModelData(lora);
            el.bindWhenConnected(lora);
            console.log(el);
            lora.baseModel && baseModels.add(lora.baseModel);
            this.modelsList.appendChild(createElement("li.model-item", { child: el }));
        }
        if (baseModels.size > 1) {
            createElement(`option[value="ALL"][text="Choose base model."]`, {
                parent: this.selectBaseModel,
            });
            for (const baseModel of baseModels.values()) {
                createElement(`option[value="${baseModel}"][text="${baseModel}"]`, {
                    parent: this.selectBaseModel,
                });
            }
            this.searchbox.insertAdjacentElement("afterend", this.selectBaseModel);
        }
        const data = getActionEls(document.body);
        for (const dataItem of Object.values(data)) {
            for (const [event, action] of Object.entries(dataItem.actions)) {
                dataItem.el.addEventListener(event, (e) => {
                    if (typeof this[action] != "function") {
                        throw new Error(`"${action}" does not exist on instance.`);
                    }
                    this[action](e);
                });
            }
        }
    }
    filter() {
        const parts = parseQuery(this.queryLast);
        const baseModel = this.selectBaseModel.value;
        const els = queryAll(RgthreeModelInfoCard.NAME);
        const shouldHide = (el) => {
            let hide = baseModel !== "ALL" && !el.hasBaseModel(baseModel);
            if (!hide) {
                for (let part of parts) {
                    let negate = false;
                    if (part.startsWith("-")) {
                        negate = true;
                        part = part.substring(1);
                    }
                    if (!part)
                        continue;
                    if (part.startsWith("has:")) {
                        if (part === "has:civitai") {
                            hide = !el.hasData(part.replace("has:", ""));
                        }
                    }
                    else {
                        hide = !el.matchesQueryText(part);
                    }
                    hide = negate ? !hide : hide;
                    if (hide) {
                        break;
                    }
                }
            }
            return hide;
        };
        for (const el of els) {
            const hide = shouldHide(el);
            if (hide) {
                el.hide();
            }
            else {
                el.show();
            }
        }
        console.log("filter");
    }
    doSearch() {
        const query = this.searchbox.value.trim();
        if (this.queryLast != query) {
            this.queryLast = query;
            this.filter();
        }
    }
}
