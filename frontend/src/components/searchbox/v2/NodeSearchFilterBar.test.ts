import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

import NodeSearchFilterBar from '@/components/searchbox/v2/NodeSearchFilterBar.vue'
import {
  createMockNodeDef,
  setupTestPinia,
  testI18n
} from '@/components/searchbox/v2/__test__/testUtils'
import { useNodeDefStore } from '@/stores/nodeDefStore'

vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: vi.fn(() => ({
    get: vi.fn(() => undefined),
    set: vi.fn()
  }))
}))

describe(NodeSearchFilterBar, () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    setupTestPinia()
    useNodeDefStore().updateNodeDefs([
      createMockNodeDef({
        name: 'ImageNode',
        input: { required: { image: ['IMAGE', {}] } },
        output: ['IMAGE']
      })
    ])
  })

  async function createWrapper(props = {}) {
    const wrapper = mount(NodeSearchFilterBar, {
      props,
      global: { plugins: [testI18n] }
    })
    await nextTick()
    return wrapper
  }

  it('should render Input, Output, and Source filter chips', async () => {
    const wrapper = await createWrapper()

    const buttons = wrapper.findAll('button')
    expect(buttons).toHaveLength(3)
    expect(buttons[0].text()).toBe('Input')
    expect(buttons[1].text()).toBe('Output')
    expect(buttons[2].text()).toBe('Source')
  })

  it('should mark active chip as pressed when activeChipKey matches', async () => {
    const wrapper = await createWrapper({ activeChipKey: 'input' })

    const inputBtn = wrapper.findAll('button').find((b) => b.text() === 'Input')
    expect(inputBtn?.attributes('aria-pressed')).toBe('true')
  })

  it('should not mark chips as pressed when activeChipKey does not match', async () => {
    const wrapper = await createWrapper({ activeChipKey: null })

    wrapper.findAll('button').forEach((btn) => {
      expect(btn.attributes('aria-pressed')).toBe('false')
    })
  })

  it('should emit selectChip with chip data when clicked', async () => {
    const wrapper = await createWrapper()

    const inputBtn = wrapper.findAll('button').find((b) => b.text() === 'Input')
    await inputBtn?.trigger('click')

    const emitted = wrapper.emitted('selectChip')!
    expect(emitted[0][0]).toMatchObject({
      key: 'input',
      label: 'Input',
      filter: expect.anything()
    })
  })
})
