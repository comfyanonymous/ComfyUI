import { flushPromises, shallowMount } from '@vue/test-utils'
import { describe, expect, it, vi, beforeEach } from 'vitest'

import SettingItem from '@/platform/settings/components/SettingItem.vue'
import type { SettingParams } from '@/platform/settings/types'
import { i18n } from '@/i18n'

/**
 * Verifies that SettingItem emits telemetry when its value changes
 * and suppresses telemetry when the value remains the same.
 */
const trackSettingChanged = vi.fn()
vi.mock('@/platform/telemetry', () => ({
  useTelemetry: vi.fn(() => ({
    trackSettingChanged
  }))
}))

const mockGet = vi.fn()
const mockSet = vi.fn()
vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: () => ({
    get: mockGet,
    set: mockSet
  })
}))

/**
 * Minimal stub for FormItem that allows emitting `update:form-value`.
 */
const FormItemUpdateStub = {
  template: '<div />',
  emits: ['update:form-value'],
  props: ['id', 'item', 'formValue']
}

describe('SettingItem (telemetry UI tracking)', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  const mountComponent = (setting: SettingParams) => {
    return shallowMount(SettingItem, {
      global: {
        plugins: [i18n],
        stubs: {
          FormItem: FormItemUpdateStub,
          Tag: true
        }
      },
      props: {
        setting
      }
    })
  }

  it('tracks telemetry when value changes via UI (uses normalized value)', async () => {
    const settingParams: SettingParams = {
      id: 'main.sub.setting.name',
      name: 'Telemetry Visible',
      type: 'text',
      defaultValue: 'default'
    }

    mockGet.mockReturnValueOnce('default').mockReturnValueOnce('normalized')
    mockSet.mockResolvedValue(undefined)

    const wrapper = mountComponent(settingParams)

    const newValue = 'newvalue'
    const formItem = wrapper.findComponent(FormItemUpdateStub)
    formItem.vm.$emit('update:form-value', newValue)

    await flushPromises()

    expect(trackSettingChanged).toHaveBeenCalledTimes(1)
    expect(trackSettingChanged).toHaveBeenCalledWith(
      expect.objectContaining({
        setting_id: 'main.sub.setting.name',
        previous_value: 'default',
        new_value: 'normalized'
      })
    )
  })

  it('does not track telemetry when normalized value does not change', async () => {
    const settingParams: SettingParams = {
      id: 'main.sub.setting.name',
      name: 'Telemetry Visible',
      type: 'text',
      defaultValue: 'same'
    }

    mockGet.mockReturnValueOnce('same').mockReturnValueOnce('same')
    mockSet.mockResolvedValue(undefined)

    const wrapper = mountComponent(settingParams)

    const unchangedValue = 'same'
    const formItem = wrapper.findComponent(FormItemUpdateStub)
    formItem.vm.$emit('update:form-value', unchangedValue)

    await flushPromises()

    expect(trackSettingChanged).not.toHaveBeenCalled()
  })
})
