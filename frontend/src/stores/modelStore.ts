import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import type { ModelFile } from '@/platform/assets/schemas/assetSchema'
import { assetService } from '@/platform/assets/services/assetService'
import { useSettingStore } from '@/platform/settings/settingStore'
import { api } from '@/scripts/api'

/** (Internal helper) finds a value in a metadata object from any of a list of keys. */
function _findInMetadata(
  metadata: Record<string, string | null>,
  ...keys: string[]
): string | null {
  for (const key of keys) {
    if (key in metadata) {
      const value = metadata[key]
      return value || null
    }
    for (const k in metadata) {
      if (k.endsWith(key)) {
        const value = metadata[k]
        return value || null
      }
    }
  }
  return null
}

/** Defines and holds metadata for a model */
export class ComfyModelDef {
  /** Path to the model */
  readonly path_index: number
  /** Proper filename of the model */
  readonly file_name: string
  /** Normalized filename of the model, with all backslashes replaced with forward slashes */
  readonly normalized_file_name: string
  /** Directory containing the model, eg 'checkpoints' */
  readonly directory: string
  /** Simplified copy of name, used as a default title. Excludes the directory and the '.safetensors' file extension */
  readonly simplified_file_name: string
  /** Key for the model, used to uniquely identify the model. */
  readonly key: string
  /** Title / display name of the model, sometimes same as the name but not always */
  title: string
  /** Metadata: architecture ID for the model, such as 'stable-diffusion-xl-v1-base' */
  architecture_id: string = ''
  /** Metadata: author of the model */
  author: string = ''
  /** Metadata: resolution of the model, eg '1024x1024' */
  resolution: string = ''
  /** Metadata: description of the model */
  description: string = ''
  /** Metadata: usage hint for the model */
  usage_hint: string = ''
  /** Metadata: trigger phrase for the model */
  trigger_phrase: string = ''
  /** Metadata: tags list for the model */
  tags: string[] = []
  /** Metadata: image for the model */
  image: string = ''
  /** Whether the model metadata has been loaded from the server, used for `load()` */
  has_loaded_metadata: boolean = false
  /** If true, a metadata load request has been triggered, but may or may not yet have finished loading */
  is_load_requested: boolean = false
  /** A string full of auto-computed lowercase-only searchable text for this model */
  searchable: string = ''

  constructor(name: string, directory: string, pathIndex: number) {
    this.path_index = pathIndex
    this.file_name = name
    this.normalized_file_name = name.replaceAll('\\', '/')
    this.simplified_file_name = this.normalized_file_name.split('/').pop() ?? ''
    if (this.simplified_file_name.endsWith('.safetensors')) {
      this.simplified_file_name = this.simplified_file_name.slice(
        0,
        -'.safetensors'.length
      )
    }
    this.title = this.simplified_file_name
    this.directory = directory
    this.key = `${directory}/${this.normalized_file_name}`
    this.updateSearchable()
  }

  updateSearchable() {
    this.searchable = [
      this.file_name,
      this.title,
      this.author,
      this.description,
      this.trigger_phrase,
      this.tags.join(', ')
    ]
      .join('\n')
      .toLowerCase()
  }

  /** Loads the model metadata from the server, filling in this object if data is available */
  async load(): Promise<void> {
    if (this.has_loaded_metadata || this.is_load_requested) {
      return
    }
    this.is_load_requested = true
    try {
      const metadata = await api.viewMetadata(this.directory, this.file_name)
      if (!metadata) {
        return
      }
      this.title =
        _findInMetadata(
          metadata,
          'modelspec.title',
          'title',
          'display_name',
          'name'
        ) || this.title
      this.architecture_id =
        _findInMetadata(metadata, 'modelspec.architecture', 'architecture') ||
        ''
      this.author =
        _findInMetadata(metadata, 'modelspec.author', 'author') || ''
      this.description =
        _findInMetadata(metadata, 'modelspec.description', 'description') || ''
      this.resolution =
        _findInMetadata(metadata, 'modelspec.resolution', 'resolution') || ''
      this.usage_hint =
        _findInMetadata(metadata, 'modelspec.usage_hint', 'usage_hint') || ''
      this.trigger_phrase =
        _findInMetadata(
          metadata,
          'modelspec.trigger_phrase',
          'trigger_phrase'
        ) || ''
      this.image =
        _findInMetadata(
          metadata,
          'modelspec.thumbnail',
          'thumbnail',
          'image',
          'icon'
        ) || ''
      const tagsCommaSeparated =
        _findInMetadata(metadata, 'modelspec.tags', 'tags') || ''
      this.tags = tagsCommaSeparated.split(',').map((tag) => tag.trim())
      this.has_loaded_metadata = true
      this.updateSearchable()
    } catch (error) {
      console.error('Error loading model metadata', this.file_name, this, error)
    }
  }
}

export enum ResourceState {
  Uninitialized,
  Loading,
  Loaded
}

export class ModelFolder {
  /** Models in this folder */
  models: Record<string, ComfyModelDef> = {}
  state: ResourceState = ResourceState.Uninitialized

  constructor(
    public directory: string,
    private getModelsFunc: (folder: string) => Promise<ModelFile[]>
  ) {}

  get key(): string {
    return this.directory + '/'
  }

  /**
   * Loads the models in this folder from the server
   */
  async load() {
    if (this.state !== ResourceState.Uninitialized) {
      return this
    }
    this.state = ResourceState.Loading
    const models = await this.getModelsFunc(this.directory)
    for (const model of models) {
      this.models[`${model.pathIndex}/${model.name}`] = new ComfyModelDef(
        model.name,
        this.directory,
        model.pathIndex
      )
    }
    this.state = ResourceState.Loaded
    return this
  }
}

/** Model store handler, wraps individual per-folder model stores */
export const useModelStore = defineStore('models', () => {
  const settingStore = useSettingStore()
  const modelFolderNames = ref<string[]>([])
  const modelFolderByName = ref<Record<string, ModelFolder>>({})
  const modelFolders = computed<ModelFolder[]>(() =>
    modelFolderNames.value.map(
      (folderName) => modelFolderByName.value[folderName]
    )
  )
  const models = computed<ComfyModelDef[]>(() =>
    modelFolders.value.flatMap((folder) => Object.values(folder.models))
  )

  /**
   * Loads the model folders from the server
   */
  async function loadModelFolders() {
    const useAssetAPI: boolean = settingStore.get('Comfy.Assets.UseAssetAPI')

    const resData = useAssetAPI
      ? await assetService.getAssetModelFolders()
      : await api.getModelFolders()
    modelFolderNames.value = resData.map((folder) => folder.name)
    modelFolderByName.value = {}
    for (const folderName of modelFolderNames.value) {
      const getModelsFunc = useAssetAPI
        ? (folder: string) => assetService.getAssetModels(folder)
        : (folder: string) => api.getModels(folder)

      modelFolderByName.value[folderName] = new ModelFolder(
        folderName,
        getModelsFunc
      )
    }
  }

  async function getLoadedModelFolder(
    folderName: string
  ): Promise<ModelFolder | null> {
    const folder = modelFolderByName.value[folderName]
    return folder ? await folder.load() : null
  }

  /**
   * Loads all model folders' contents from the server
   */
  async function loadModels() {
    return Promise.all(modelFolders.value.map((folder) => folder.load()))
  }

  return {
    models,
    modelFolders,
    loadModelFolders,
    loadModels,
    getLoadedModelFolder
  }
})
