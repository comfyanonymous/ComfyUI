import { createTestingPinia } from '@pinia/testing'
import { mount } from '@vue/test-utils'
import ProgressSpinner from 'primevue/progressspinner'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import PackCard from '@/workbench/extensions/manager/components/manager/packCard/PackCard.vue'
import type {
  MergedNodePack,
  RegistryPack
} from '@/workbench/extensions/manager/types/comfyManagerTypes'

const translateMock = vi.hoisted(() =>
  vi.fn((key: string, choice?: number) =>
    typeof choice === 'number' ? `${key}-${choice}` : key
  )
)
const dateMock = vi.hoisted(() => vi.fn(() => '2024. 1. 1.'))
const storageMap = vi.hoisted(() => new Map<string, unknown>())

// Mock dependencies
vi.mock('vue-i18n', () => ({
  useI18n: vi.fn(() => ({
    d: dateMock,
    t: translateMock
  })),
  createI18n: vi.fn(() => ({
    global: {
      t: translateMock,
      te: vi.fn(() => true)
    }
  }))
}))

vi.mock('@/workbench/extensions/manager/stores/comfyManagerStore', () => ({
  useComfyManagerStore: vi.fn(() => ({
    isPackInstalled: vi.fn(() => false),
    isPackEnabled: vi.fn(() => true),
    isPackInstalling: vi.fn(() => false),
    installedPacksIds: []
  }))
}))

vi.mock('@/stores/workspace/colorPaletteStore', () => ({
  useColorPaletteStore: vi.fn(() => ({
    completedActivePalette: { light_theme: true }
  }))
}))

vi.mock('@vueuse/core', () => ({
  whenever: vi.fn(),
  useStorage: vi.fn((key: string, defaultValue: unknown) => {
    if (!storageMap.has(key)) storageMap.set(key, defaultValue)
    return storageMap.get(key)
  }),
  createSharedComposable: vi.fn((fn) => {
    let cached: ReturnType<typeof fn>
    return (...args: Parameters<typeof fn>) => (cached ??= fn(...args))
  })
}))

vi.mock('@/config', () => ({
  default: {
    app_version: '1.24.0-1'
  }
}))

vi.mock('@/stores/systemStatsStore', () => ({
  useSystemStatsStore: vi.fn(() => ({
    systemStats: {
      system: { os: 'Darwin' },
      devices: [{ type: 'mps', name: 'Metal' }]
    }
  }))
}))

describe('PackCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    storageMap.clear()
  })

  const createWrapper = (props: {
    nodePack: MergedNodePack | RegistryPack
    isSelected?: boolean
  }) => {
    const wrapper = mount(PackCard, {
      props,
      global: {
        plugins: [createTestingPinia({ stubActions: false })],
        components: {
          ProgressSpinner
        },
        stubs: {
          PackBanner: true,
          PackVersionBadge: true,
          PackCardFooter: true
        },
        mocks: {
          $t: vi.fn((key: string) => key)
        }
      }
    })

    return wrapper
  }

  const mockNodePack: RegistryPack = {
    id: 'test-package',
    name: 'Test Package',
    description: 'Test package description',
    author: 'Test Author',
    latest_version: {
      createdAt: '2024-01-01T00:00:00Z'
    }
  } as RegistryPack

  describe('basic rendering', () => {
    it('should render package card with basic information', () => {
      const wrapper = createWrapper({ nodePack: mockNodePack })

      expect(wrapper.exists()).toBe(true)
      expect(wrapper.text()).toContain('Test Package')
      expect(wrapper.text()).toContain('Test package description')
      expect(wrapper.text()).toContain('Test Author')
    })

    it('should render date correctly', () => {
      const wrapper = createWrapper({ nodePack: mockNodePack })

      expect(wrapper.text()).toContain('2024. 1. 1.')
    })

    it('should apply selected ring when isSelected is true', () => {
      const wrapper = createWrapper({
        nodePack: mockNodePack,
        isSelected: true
      })

      expect(wrapper.find('.ring-3').exists()).toBe(true)
    })

    it('should not apply selected ring when isSelected is false', () => {
      const wrapper = createWrapper({
        nodePack: mockNodePack,
        isSelected: false
      })

      expect(wrapper.find('.ring-3').exists()).toBe(false)
    })
  })

  describe('component behavior', () => {
    it('should render without errors', () => {
      const wrapper = createWrapper({ nodePack: mockNodePack })

      expect(wrapper.exists()).toBe(true)
      expect(wrapper.find('.rounded-lg').exists()).toBe(true)
    })
  })

  describe('package information display', () => {
    it('should display package name', () => {
      const wrapper = createWrapper({ nodePack: mockNodePack })

      expect(wrapper.text()).toContain('Test Package')
    })

    it('should display package description', () => {
      const wrapper = createWrapper({ nodePack: mockNodePack })

      expect(wrapper.text()).toContain('Test package description')
    })

    it('should display author name', () => {
      const wrapper = createWrapper({ nodePack: mockNodePack })

      expect(wrapper.text()).toContain('Test Author')
    })

    it('should handle missing description', () => {
      const packWithoutDescription = { ...mockNodePack, description: undefined }
      const wrapper = createWrapper({ nodePack: packWithoutDescription })

      expect(wrapper.find('p').exists()).toBe(false)
    })

    it('should handle missing author', () => {
      const packWithoutAuthor = { ...mockNodePack, author: undefined }
      const wrapper = createWrapper({ nodePack: packWithoutAuthor })

      // Should still render without errors
      expect(wrapper.exists()).toBe(true)
    })

    it('should use localized singular/plural nodes label', () => {
      const packWithNodes = {
        ...mockNodePack,
        comfy_nodes: ['node-a']
      } as MergedNodePack

      const wrapper = createWrapper({ nodePack: packWithNodes })

      expect(wrapper.text()).toContain('g.nodesCount-1')
      expect(translateMock).toHaveBeenCalledWith('g.nodesCount', 1)
    })
  })

  describe('component structure', () => {
    it('should render PackBanner component', () => {
      const wrapper = createWrapper({ nodePack: mockNodePack })

      expect(wrapper.find('pack-banner-stub').exists()).toBe(true)
    })

    it('should render PackVersionBadge component', () => {
      const wrapper = createWrapper({ nodePack: mockNodePack })

      expect(wrapper.find('pack-version-badge-stub').exists()).toBe(true)
    })

    it('should render PackCardFooter component', () => {
      const wrapper = createWrapper({ nodePack: mockNodePack })

      expect(wrapper.find('pack-card-footer-stub').exists()).toBe(true)
    })
  })
})
