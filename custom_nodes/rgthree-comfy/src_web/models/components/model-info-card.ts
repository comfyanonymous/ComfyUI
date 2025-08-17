import {RgthreeCustomElement} from "rgthree/common/components/base_custom_element";
import {RgthreeModelInfo} from "typings/rgthree";

export class RgthreeModelInfoCard extends RgthreeCustomElement {
  static override readonly NAME = "rgthree-model-info-card";
  static override readonly TEMPLATES = "components/model-info-card.html";
  static override readonly CSS = "components/model-info-card.css";
  static override readonly USE_SHADOW = false;

  private data: RgthreeModelInfo = {};

  getModified(
    value: number,
    data: any,
    currentElement: HTMLElement,
    contextElement: RgthreeModelInfoCard,
  ) {
    const date = new Date(value);
    return String(`${date.toLocaleDateString()} ${date.toLocaleTimeString()}`);
  }

  getCivitaiLink(links: string[] | undefined) {
    return links?.find((i) => i.includes("civitai.com/models")) || null;
  }

  setModelData(data: RgthreeModelInfo) {
    this.data = data;
  }

  hasBaseModel(baseModel: string) {
    return this.data.baseModel === baseModel;
  }

  hasData(field: string) {
    // return !!this.data.hasInfoFile;
    if (field === "civitai") {
      return !!this.getCivitaiLink(this.data.links)?.length;
    }
    return !!(this.data as any)[field];
  }

  matchesQueryText(query: string) {
    return (this.data.name || this.data.file)?.includes(query);
  }

  hide() {
    this.classList.add("-is-hidden");
  }

  show() {
    this.classList.remove("-is-hidden");
  }
}
