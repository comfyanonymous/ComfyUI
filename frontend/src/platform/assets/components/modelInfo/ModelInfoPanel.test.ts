import { mount } from '@vue/test-utils'
import { createTestingPinia } from '@pinia/testing'
import { describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'

import type { AssetDisplayItem } from '@/platform/assets/composables/useAssetBrowser'

import ModelInfoPanel from './ModelInfoPanel.vue'

vi.mock('@/composables/useCopyToClipboard', () => ({
  useCopyToClipboard: () => ({
    copyToClipboard: vi.fn()
  })
}))

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: { en: {} },
  missingWarn: false,
  fallbackWarn: false
})

describe('ModelInfoPanel', () => {
  const createMockAsset = (
    overrides: Partial<AssetDisplayItem> = {}
  ): AssetDisplayItem => ({
    id: 'test-id',
    name: 'test-model.safetensors',
    asset_hash: 'hash123',
    size: 1024,
    mime_type: 'application/octet-stream',
    tags: ['models', 'checkpoints'],
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    last_access_time: '2024-01-01T00:00:00Z',
    secondaryText: 'A test model description',
    badges: [],
    stats: {},
    ...overrides
  })

  const mountPanel = (asset: AssetDisplayItem) => {
    return mount(ModelInfoPanel, {
      props: { asset },
      global: {
        plugins: [createTestingPinia({ stubActions: false }), i18n]
      }
    })
  }

  describe('Basic Info Section', () => {
    it('renders basic info section', () => {
      const wrapper = mountPanel(createMockAsset())
      expect(wrapper.text()).toContain('assetBrowser.modelInfo.basicInfo')
    })

    it('displays asset filename', () => {
      const asset = createMockAsset({ name: 'my-model.safetensors' })
      const wrapper = mountPanel(asset)
      expect(wrapper.text()).toContain('my-model.safetensors')
    })

    it('displays name from user_metadata when present', () => {
      const asset = createMockAsset({
        user_metadata: { name: 'My Custom Model' }
      })
      const wrapper = mountPanel(asset)
      expect(wrapper.text()).toContain('My Custom Model')
    })

    it('falls back to asset name when user_metadata.name not present', () => {
      const asset = createMockAsset({ name: 'fallback-model.safetensors' })
      const wrapper = mountPanel(asset)
      expect(wrapper.text()).toContain('fallback-model.safetensors')
    })

    it('renders source link when source_arn is present', () => {
      const asset = createMockAsset({
        user_metadata: { source_arn: 'civitai:model:123:version:456' }
      })
      const wrapper = mountPanel(asset)
      const link = wrapper.find(
        'a[href="https://civitai.com/models/123?modelVersionId=456"]'
      )
      expect(link.exists()).toBe(true)
      expect(link.attributes('target')).toBe('_blank')
    })

    it('displays Civitai icon for Civitai source', () => {
      const asset = createMockAsset({
        user_metadata: { source_arn: 'civitai:model:123:version:456' }
      })
      const wrapper = mountPanel(asset)
      expect(
        wrapper.find('img[src="/assets/images/civitai.svg"]').exists()
      ).toBe(true)
    })

    it('does not render source field when source_arn is absent', () => {
      const asset = createMockAsset()
      const wrapper = mountPanel(asset)
      const links = wrapper.findAll('a')
      expect(links).toHaveLength(0)
    })
  })

  describe('Model Tagging Section', () => {
    it('renders model tagging section', () => {
      const wrapper = mountPanel(createMockAsset())
      expect(wrapper.text()).toContain('assetBrowser.modelInfo.modelTagging')
    })

    it('renders model type field', () => {
      const wrapper = mountPanel(createMockAsset())
      expect(wrapper.text()).toContain('assetBrowser.modelInfo.modelType')
    })

    it('renders base models field', () => {
      const asset = createMockAsset({
        user_metadata: { base_model: ['SDXL'] }
      })
      const wrapper = mountPanel(asset)
      expect(wrapper.text()).toContain(
        'assetBrowser.modelInfo.compatibleBaseModels'
      )
    })

    it('renders additional tags field', () => {
      const wrapper = mountPanel(createMockAsset())
      expect(wrapper.text()).toContain('assetBrowser.modelInfo.additionalTags')
    })
  })

  describe('Model Description Section', () => {
    it('renders trigger phrases when present', () => {
      const asset = createMockAsset({
        user_metadata: { trained_words: ['trigger1', 'trigger2'] }
      })
      const wrapper = mountPanel(asset)
      expect(wrapper.text()).toContain('trigger1')
      expect(wrapper.text()).toContain('trigger2')
    })

    it('renders description section', () => {
      const wrapper = mountPanel(createMockAsset())
      expect(wrapper.text()).toContain(
        'assetBrowser.modelInfo.modelDescription'
      )
    })

    it('does not render trigger phrases field when empty', () => {
      const asset = createMockAsset()
      const wrapper = mountPanel(asset)
      expect(wrapper.text()).not.toContain(
        'assetBrowser.modelInfo.triggerPhrases'
      )
    })
  })

  describe('Accordion Structure', () => {
    it('renders all three section labels', () => {
      const wrapper = mountPanel(createMockAsset())
      expect(wrapper.text()).toContain('assetBrowser.modelInfo.basicInfo')
      expect(wrapper.text()).toContain('assetBrowser.modelInfo.modelTagging')
      expect(wrapper.text()).toContain(
        'assetBrowser.modelInfo.modelDescription'
      )
    })
  })
})
