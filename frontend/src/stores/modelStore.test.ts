import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { assetService } from '@/platform/assets/services/assetService'
import { useSettingStore } from '@/platform/settings/settingStore'
import { api } from '@/scripts/api'
import { useModelStore } from '@/stores/modelStore'

// Mock the api
vi.mock('@/scripts/api', () => ({
  api: {
    getModels: vi.fn(),
    getModelFolders: vi.fn(),
    viewMetadata: vi.fn(),
    apiURL: vi.fn((path: string) => `http://localhost:8188${path}`),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn()
  }
}))

// Mock the assetService
vi.mock('@/platform/assets/services/assetService', () => ({
  assetService: {
    getAssetModelFolders: vi.fn(),
    getAssetModels: vi.fn()
  }
}))

// Mock the settingStore
vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: vi.fn()
}))

function enableMocks(useAssetAPI = false) {
  // Mock settingStore to return the useAssetAPI setting
  const mockSettingStore = {
    get: vi.fn().mockImplementation((key: string) => {
      if (key === 'Comfy.Assets.UseAssetAPI') {
        return useAssetAPI
      }
      return false
    })
  }
  vi.mocked(useSettingStore, { partial: true }).mockReturnValue(
    mockSettingStore
  )

  // Mock experimental API - returns objects with name and folders properties
  vi.mocked(api.getModels).mockResolvedValue([
    { name: 'sdxl.safetensors', pathIndex: 0 },
    { name: 'sdv15.safetensors', pathIndex: 0 },
    { name: 'noinfo.safetensors', pathIndex: 0 }
  ])
  vi.mocked(api.getModelFolders).mockResolvedValue([
    { name: 'checkpoints', folders: ['/path/to/checkpoints'] },
    { name: 'vae', folders: ['/path/to/vae'] }
  ])

  // Mock asset API - also returns objects with name and folders properties
  vi.mocked(assetService.getAssetModelFolders).mockResolvedValue([
    { name: 'checkpoints', folders: ['/path/to/checkpoints'] },
    { name: 'vae', folders: ['/path/to/vae'] }
  ])
  vi.mocked(assetService.getAssetModels).mockResolvedValue([
    { name: 'sdxl.safetensors', pathIndex: 0 },
    { name: 'sdv15.safetensors', pathIndex: 0 },
    { name: 'noinfo.safetensors', pathIndex: 0 }
  ])

  vi.mocked(api.viewMetadata).mockImplementation((_, model) => {
    if (model === 'noinfo.safetensors') {
      return Promise.resolve({})
    }
    return Promise.resolve({
      'modelspec.title': `Title of ${model}`,
      display_name: 'Should not show',
      'modelspec.architecture': 'stable-diffusion-xl-base-v1',
      'modelspec.author': `Author of ${model}`,
      'modelspec.description': `Description of ${model}`,
      'modelspec.resolution': '1024x1024',
      trigger_phrase: `Trigger phrase of ${model}`,
      usage_hint: `Usage hint of ${model}`,
      tags: `tags,for,${model}`
    })
  })
}

describe('useModelStore', () => {
  let store: ReturnType<typeof useModelStore>

  beforeEach(async () => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    vi.resetAllMocks()
  })

  it('should load models', async () => {
    enableMocks()
    store = useModelStore()
    await store.loadModelFolders()
    const folderStore = await store.getLoadedModelFolder('checkpoints')
    expect(folderStore).toBeDefined()
    expect(Object.keys(folderStore!.models)).toHaveLength(3)
  })

  it('should load model metadata', async () => {
    enableMocks()
    store = useModelStore()
    await store.loadModelFolders()
    const folderStore = await store.getLoadedModelFolder('checkpoints')
    expect(folderStore).toBeDefined()
    const model = folderStore!.models['0/sdxl.safetensors']
    await model.load()
    expect(model.title).toBe('Title of sdxl.safetensors')
    expect(model.architecture_id).toBe('stable-diffusion-xl-base-v1')
    expect(model.author).toBe('Author of sdxl.safetensors')
    expect(model.description).toBe('Description of sdxl.safetensors')
    expect(model.resolution).toBe('1024x1024')
    expect(model.trigger_phrase).toBe('Trigger phrase of sdxl.safetensors')
    expect(model.usage_hint).toBe('Usage hint of sdxl.safetensors')
    expect(model.tags).toHaveLength(3)
  })

  it('should handle no metadata', async () => {
    enableMocks()
    store = useModelStore()
    await store.loadModelFolders()
    const folderStore = await store.getLoadedModelFolder('checkpoints')
    expect(folderStore).toBeDefined()
    const model = folderStore!.models['0/noinfo.safetensors']
    await model.load()
    expect(model.file_name).toBe('noinfo.safetensors')
    expect(model.title).toBe('noinfo')
    expect(model.architecture_id).toBe('')
    expect(model.author).toBe('')
    expect(model.description).toBe('')
    expect(model.resolution).toBe('')
  })

  it('should cache model information', async () => {
    enableMocks()
    store = useModelStore()
    await store.loadModelFolders()
    expect(api.getModels).toHaveBeenCalledTimes(0)
    await store.getLoadedModelFolder('checkpoints')
    expect(api.getModels).toHaveBeenCalledTimes(1)
    await store.getLoadedModelFolder('checkpoints')
    expect(api.getModels).toHaveBeenCalledTimes(1)
  })

  describe('API switching functionality', () => {
    it('should use experimental API for complete workflow when UseAssetAPI setting is false', async () => {
      enableMocks(false) // useAssetAPI = false
      store = useModelStore()
      await store.loadModelFolders()
      const folderStore = await store.getLoadedModelFolder('checkpoints')

      // Both APIs return objects with .name property, modelStore extracts folder.name in both cases
      expect(api.getModelFolders).toHaveBeenCalledTimes(1)
      expect(api.getModels).toHaveBeenCalledWith('checkpoints')
      expect(assetService.getAssetModelFolders).toHaveBeenCalledTimes(0)
      expect(assetService.getAssetModels).toHaveBeenCalledTimes(0)
      expect(folderStore).toBeDefined()
      expect(Object.keys(folderStore!.models)).toHaveLength(3)
    })

    it('should use asset API for complete workflow when UseAssetAPI setting is true', async () => {
      enableMocks(true) // useAssetAPI = true
      store = useModelStore()
      await store.loadModelFolders()
      const folderStore = await store.getLoadedModelFolder('checkpoints')

      // Both APIs return objects with .name property, modelStore extracts folder.name in both cases
      expect(assetService.getAssetModelFolders).toHaveBeenCalledTimes(1)
      expect(assetService.getAssetModels).toHaveBeenCalledWith('checkpoints')
      expect(api.getModelFolders).toHaveBeenCalledTimes(0)
      expect(api.getModels).toHaveBeenCalledTimes(0)
      expect(folderStore).toBeDefined()
      expect(Object.keys(folderStore!.models)).toHaveLength(3)
    })
  })
})
