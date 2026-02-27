import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

import type { SecretMetadata } from '../types'
import SecretListItem from './SecretListItem.vue'

vi.mock('../providers', () => ({
  getProviderLabel: (provider: string | undefined) => {
    if (provider === 'huggingface') return 'HuggingFace'
    if (provider === 'civitai') return 'Civitai'
    return ''
  },
  getProviderLogo: () => undefined
}))

function createMockSecret(
  overrides: Partial<SecretMetadata> = {}
): SecretMetadata {
  return {
    id: 'secret-1',
    name: 'Test Secret',
    provider: 'huggingface',
    created_at: '2024-01-15T10:00:00Z',
    updated_at: '2024-01-15T10:00:00Z',
    ...overrides
  }
}

function mountComponent(props: {
  secret: SecretMetadata
  loading?: boolean
  disabled?: boolean
}) {
  return mount(SecretListItem, {
    props,
    global: {
      stubs: {
        Button: {
          template:
            '<button :disabled="disabled" @click="$emit(\'click\')"><slot /></button>',
          props: ['disabled', 'variant', 'size', 'aria-label']
        }
      },
      directives: {
        tooltip: () => {}
      },
      mocks: {
        $t: (key: string, params?: object) =>
          `${key}${params ? JSON.stringify(params) : ''}`
      }
    }
  })
}

describe('SecretListItem', () => {
  describe('rendering', () => {
    it('displays secret name', () => {
      const secret = createMockSecret({ name: 'My API Key' })
      const wrapper = mountComponent({ secret })

      expect(wrapper.text()).toContain('My API Key')
    })

    it('displays provider label when provider exists', () => {
      const secret = createMockSecret({ provider: 'huggingface' })
      const wrapper = mountComponent({ secret })

      expect(wrapper.text()).toContain('HuggingFace')
    })

    it('displays Civitai provider label', () => {
      const secret = createMockSecret({ provider: 'civitai' })
      const wrapper = mountComponent({ secret })

      expect(wrapper.text()).toContain('Civitai')
    })

    it('hides provider badge when no provider', () => {
      const secret = createMockSecret({ provider: undefined })
      const wrapper = mountComponent({ secret })

      expect(wrapper.text()).not.toContain('HuggingFace')
      expect(wrapper.text()).not.toContain('Civitai')
    })

    it('displays created date', () => {
      const secret = createMockSecret({ created_at: '2024-01-15T10:00:00Z' })
      const wrapper = mountComponent({ secret })

      expect(wrapper.text()).toContain('secrets.createdAt')
    })

    it('displays last used date when available', () => {
      const secret = createMockSecret({ last_used_at: '2024-01-20T10:00:00Z' })
      const wrapper = mountComponent({ secret })

      expect(wrapper.text()).toContain('secrets.lastUsed')
    })

    it('hides last used when not available', () => {
      const secret = createMockSecret({ last_used_at: undefined })
      const wrapper = mountComponent({ secret })

      expect(wrapper.text()).not.toContain('secrets.lastUsed')
    })
  })

  describe('loading state', () => {
    it('shows spinner when loading', () => {
      const secret = createMockSecret()
      const wrapper = mountComponent({ secret, loading: true })

      expect(wrapper.find('.pi-spinner').exists()).toBe(true)
    })

    it('hides action buttons when loading', () => {
      const secret = createMockSecret()
      const wrapper = mountComponent({ secret, loading: true })

      expect(wrapper.find('.pi-pen-to-square').exists()).toBe(false)
      expect(wrapper.find('.pi-trash').exists()).toBe(false)
    })

    it('shows action buttons when not loading', () => {
      const secret = createMockSecret()
      const wrapper = mountComponent({ secret, loading: false })

      expect(wrapper.find('.pi-pen-to-square').exists()).toBe(true)
      expect(wrapper.find('.pi-trash').exists()).toBe(true)
    })
  })

  describe('disabled state', () => {
    it('disables buttons when disabled prop is true', () => {
      const secret = createMockSecret()
      const wrapper = mountComponent({ secret, disabled: true })

      const buttons = wrapper.findAll('button')
      buttons.forEach((button) => {
        expect(button.attributes('disabled')).toBeDefined()
      })
    })

    it('enables buttons when disabled prop is false', () => {
      const secret = createMockSecret()
      const wrapper = mountComponent({ secret, disabled: false })

      const buttons = wrapper.findAll('button')
      buttons.forEach((button) => {
        expect(button.attributes('disabled')).toBeUndefined()
      })
    })
  })

  describe('events', () => {
    it('emits edit event when edit button clicked', async () => {
      const secret = createMockSecret()
      const wrapper = mountComponent({ secret })

      const editButton = wrapper.findAll('button')[0]
      await editButton.trigger('click')

      expect(wrapper.emitted('edit')).toBeDefined()
      expect(wrapper.emitted('edit')!.length).toBeGreaterThanOrEqual(1)
    })

    it('emits delete event when delete button clicked', async () => {
      const secret = createMockSecret()
      const wrapper = mountComponent({ secret })

      const deleteButton = wrapper.findAll('button')[1]
      await deleteButton.trigger('click')

      expect(wrapper.emitted('delete')).toBeDefined()
      expect(wrapper.emitted('delete')!.length).toBeGreaterThanOrEqual(1)
    })
  })
})
