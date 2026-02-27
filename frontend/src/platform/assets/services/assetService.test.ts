import { beforeEach, describe, expect, it, vi } from 'vitest'

import { assetService } from '@/platform/assets/services/assetService'

const mockDistributionState = vi.hoisted(() => ({ isCloud: false }))
const mockSettingStoreGet = vi.hoisted(() => vi.fn(() => false))

vi.mock('@/platform/distribution/types', () => ({
  get isCloud() {
    return mockDistributionState.isCloud
  }
}))

vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: vi.fn(() => ({
    get: mockSettingStoreGet
  }))
}))

vi.mock('@/stores/modelToNodeStore', () => {
  const registeredNodeTypes: Record<string, string> = {
    CheckpointLoaderSimple: 'ckpt_name',
    LoraLoader: 'lora_name'
  }
  return {
    useModelToNodeStore: vi.fn(() => ({
      getRegisteredNodeTypes: () => registeredNodeTypes,
      getCategoryForNodeType: vi.fn()
    }))
  }
})

vi.mock('@/scripts/api', () => ({
  api: {
    fetchApi: vi.fn()
  }
}))

vi.mock('@/i18n', () => ({
  st: vi.fn((_key: string, fallback: string) => fallback)
}))

describe(assetService.shouldUseAssetBrowser, () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockDistributionState.isCloud = false
    mockSettingStoreGet.mockReturnValue(false)
  })

  it('returns false when not on cloud', () => {
    mockDistributionState.isCloud = false
    mockSettingStoreGet.mockReturnValue(true)

    expect(
      assetService.shouldUseAssetBrowser('CheckpointLoaderSimple', 'ckpt_name')
    ).toBe(false)
  })

  it('returns false when asset API setting is disabled', () => {
    mockDistributionState.isCloud = true
    mockSettingStoreGet.mockReturnValue(false)

    expect(
      assetService.shouldUseAssetBrowser('CheckpointLoaderSimple', 'ckpt_name')
    ).toBe(false)
  })

  it('returns false when node type is not eligible', () => {
    mockDistributionState.isCloud = true
    mockSettingStoreGet.mockReturnValue(true)

    expect(
      assetService.shouldUseAssetBrowser('UnknownNode', 'some_input')
    ).toBe(false)
  })

  it('returns true when cloud, setting enabled, and node is eligible', () => {
    mockDistributionState.isCloud = true
    mockSettingStoreGet.mockReturnValue(true)

    expect(
      assetService.shouldUseAssetBrowser('CheckpointLoaderSimple', 'ckpt_name')
    ).toBe(true)
  })

  it('returns false when nodeType is undefined', () => {
    mockDistributionState.isCloud = true
    mockSettingStoreGet.mockReturnValue(true)

    expect(assetService.shouldUseAssetBrowser(undefined, 'ckpt_name')).toBe(
      false
    )
  })

  it('returns false when widget name does not match registered input', () => {
    mockDistributionState.isCloud = true
    mockSettingStoreGet.mockReturnValue(true)

    expect(
      assetService.shouldUseAssetBrowser(
        'CheckpointLoaderSimple',
        'wrong_input'
      )
    ).toBe(false)
  })
})
