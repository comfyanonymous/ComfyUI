import type {RgthreeModelInfo} from "typings/rgthree.js";
import {ModelInfoType, rgthreeApi} from "./rgthree_api.js";
import {api} from "scripts/api.js";

/**
 * Abstract class defining information syncing for different types.
 */
abstract class BaseModelInfoService extends EventTarget {
  private readonly fileToInfo = new Map<string, RgthreeModelInfo | null>();
  protected abstract readonly modelInfoType: ModelInfoType;

  protected abstract readonly apiRefreshEventString: string;

  constructor() {
    super();
    this.init();
  }

  private init() {
    api.addEventListener(
      this.apiRefreshEventString,
      this.handleAsyncUpdate.bind(this) as EventListener,
    );
  }

  async getInfo(file: string, refresh: boolean, light: boolean) {
    if (this.fileToInfo.has(file) && !refresh) {
      return this.fileToInfo.get(file)!;
    }
    return this.fetchInfo(file, refresh, light);
  }

  async refreshInfo(file: string) {
    return this.fetchInfo(file, true);
  }

  async clearFetchedInfo(file: string) {
    await rgthreeApi.clearModelsInfo({type: this.modelInfoType, files: [file]});
    this.fileToInfo.delete(file);
    return null;
  }

  async savePartialInfo(file: string, data: Partial<RgthreeModelInfo>) {
    let info = await rgthreeApi.saveModelInfo(this.modelInfoType, file, data);
    this.fileToInfo.set(file, info);
    return info;
  }

  handleAsyncUpdate(event: CustomEvent<{data: RgthreeModelInfo}>) {
    const info = event.detail?.data as RgthreeModelInfo;
    if (info?.file) {
      this.setFreshInfo(info.file, info);
    }
  }

  private async fetchInfo(file: string, refresh = false, light = false) {
    let info = null;
    if (!refresh) {
      info = await rgthreeApi.getModelsInfo({type: this.modelInfoType, files: [file], light});
    } else {
      info = await rgthreeApi.refreshModelsInfo({type: this.modelInfoType, files: [file]});
    }
    info = info?.[0] ?? null;
    if (!light) {
      this.fileToInfo.set(file, info);
    }
    return info;
  }

  /**
   * Single point to set data into the info cache, and fire an event. Note, this doesn't determine
   * if the data is actually different.
   */
  private setFreshInfo(file: string, info: RgthreeModelInfo) {
    this.fileToInfo.set(file, info);
    // this.dispatchEvent(
    //   new CustomEvent("rgthree-model-service-lora-details", { detail: { lora: info } }),
    // );
  }
}

/**
 * Lora type implementation of ModelInfoTypeService.
 */
class LoraInfoService extends BaseModelInfoService {
  protected override readonly apiRefreshEventString = "rgthree-refreshed-loras-info";
  protected override readonly modelInfoType = 'loras';
}

/**
 * Checkpoint type implementation of ModelInfoTypeService.
 */
class CheckpointInfoService extends BaseModelInfoService {
  protected override readonly apiRefreshEventString = "rgthree-refreshed-checkpoints-info";
  protected override readonly modelInfoType = 'checkpoints';
}

export const LORA_INFO_SERVICE = new LoraInfoService();
export const CHECKPOINT_INFO_SERVICE = new CheckpointInfoService();
