import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { FilterChip } from '@/components/searchbox/v2/NodeSearchFilterBar.vue'
import NodeSearchInput from '@/components/searchbox/v2/NodeSearchInput.vue'
import {
  setupTestPinia,
  testI18n
} from '@/components/searchbox/v2/__test__/testUtils'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import type { FuseFilter, FuseFilterWithValue } from '@/utils/fuseUtil'

vi.mock('@/utils/litegraphUtil', () => ({
  getLinkTypeColor: vi.fn((type: string) =>
    type === 'IMAGE' ? '#64b5f6' : undefined
  )
}))

vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: vi.fn(() => ({
    get: vi.fn(),
    set: vi.fn()
  }))
}))

function createFilter(
  id: string,
  value: string
): FuseFilterWithValue<ComfyNodeDefImpl, string> {
  return {
    filterDef: {
      id,
      matches: vi.fn(() => true)
    } as Partial<FuseFilter<ComfyNodeDefImpl, string>> as FuseFilter<
      ComfyNodeDefImpl,
      string
    >,
    value
  }
}

function createActiveFilter(label: string): FilterChip {
  return {
    key: label.toLowerCase(),
    label,
    filter: {
      id: label.toLowerCase(),
      matches: vi.fn(() => true)
    } as Partial<FuseFilter<ComfyNodeDefImpl, string>> as FuseFilter<
      ComfyNodeDefImpl,
      string
    >
  }
}

describe('NodeSearchInput', () => {
  beforeEach(() => {
    setupTestPinia()
    vi.restoreAllMocks()
  })

  function createWrapper(
    props: Partial<{
      filters: FuseFilterWithValue<ComfyNodeDefImpl, string>[]
      activeFilter: FilterChip | null
      searchQuery: string
      filterQuery: string
    }> = {}
  ) {
    return mount(NodeSearchInput, {
      props: {
        filters: [],
        activeFilter: null,
        searchQuery: '',
        filterQuery: '',
        ...props
      },
      global: { plugins: [testI18n] }
    })
  }

  it('should route input to searchQuery when no active filter', async () => {
    const wrapper = createWrapper()
    await wrapper.find('input').setValue('test search')

    expect(wrapper.emitted('update:searchQuery')![0]).toEqual(['test search'])
  })

  it('should route input to filterQuery when active filter is set', async () => {
    const wrapper = createWrapper({
      activeFilter: createActiveFilter('Input')
    })
    await wrapper.find('input').setValue('IMAGE')

    expect(wrapper.emitted('update:filterQuery')![0]).toEqual(['IMAGE'])
    expect(wrapper.emitted('update:searchQuery')).toBeUndefined()
  })

  it('should show filter label placeholder when active filter is set', () => {
    const wrapper = createWrapper({
      activeFilter: createActiveFilter('Input')
    })

    expect(
      (wrapper.find('input').element as HTMLInputElement).placeholder
    ).toContain('input')
  })

  it('should show add node placeholder when no active filter', () => {
    const wrapper = createWrapper()

    expect(
      (wrapper.find('input').element as HTMLInputElement).placeholder
    ).toContain('Add a node')
  })

  it('should hide filter chips when active filter is set', () => {
    const wrapper = createWrapper({
      filters: [createFilter('input', 'IMAGE')],
      activeFilter: createActiveFilter('Input')
    })

    expect(wrapper.findAll('[data-testid="filter-chip"]')).toHaveLength(0)
  })

  it('should show filter chips when no active filter', () => {
    const wrapper = createWrapper({
      filters: [createFilter('input', 'IMAGE')]
    })

    expect(wrapper.findAll('[data-testid="filter-chip"]')).toHaveLength(1)
  })

  it('should emit cancelFilter when cancel button is clicked', async () => {
    const wrapper = createWrapper({
      activeFilter: createActiveFilter('Input')
    })

    await wrapper.find('[data-testid="cancel-filter"]').trigger('click')

    expect(wrapper.emitted('cancelFilter')).toHaveLength(1)
  })

  it('should emit selectCurrent on Enter', async () => {
    const wrapper = createWrapper()

    await wrapper.find('input').trigger('keydown', { key: 'Enter' })

    expect(wrapper.emitted('selectCurrent')).toHaveLength(1)
  })

  it('should emit navigateDown on ArrowDown', async () => {
    const wrapper = createWrapper()

    await wrapper.find('input').trigger('keydown', { key: 'ArrowDown' })

    expect(wrapper.emitted('navigateDown')).toHaveLength(1)
  })

  it('should emit navigateUp on ArrowUp', async () => {
    const wrapper = createWrapper()

    await wrapper.find('input').trigger('keydown', { key: 'ArrowUp' })

    expect(wrapper.emitted('navigateUp')).toHaveLength(1)
  })
})
