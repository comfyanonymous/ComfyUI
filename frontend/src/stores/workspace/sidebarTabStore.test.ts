import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { nextTick, ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useSidebarTabStore } from '@/stores/workspace/sidebarTabStore'

const { mockGetSetting, mockRegisterCommand, mockRegisterCommands } =
  vi.hoisted(() => ({
    mockGetSetting: vi.fn(),
    mockRegisterCommand: vi.fn(),
    mockRegisterCommands: vi.fn()
  }))

vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: () => ({
    get: mockGetSetting
  })
}))

vi.mock('@/stores/commandStore', () => ({
  useCommandStore: () => ({
    registerCommand: mockRegisterCommand,
    commands: []
  })
}))

vi.mock('@/stores/menuItemStore', () => ({
  useMenuItemStore: () => ({
    registerCommands: mockRegisterCommands
  })
}))

vi.mock('@/i18n', () => ({
  t: (key: string) => key,
  te: () => false
}))

vi.mock('@/composables/sidebarTabs/useAssetsSidebarTab', () => ({
  useAssetsSidebarTab: () => ({
    id: 'assets',
    title: 'assets',
    type: 'vue',
    component: {}
  })
}))

vi.mock('@/composables/sidebarTabs/useJobHistorySidebarTab', () => ({
  useJobHistorySidebarTab: () => ({
    id: 'job-history',
    title: 'job-history',
    type: 'vue',
    component: {}
  })
}))

vi.mock('@/composables/sidebarTabs/useNodeLibrarySidebarTab', () => ({
  useNodeLibrarySidebarTab: () => ({
    id: 'node-library',
    title: 'node-library',
    type: 'vue',
    component: {}
  })
}))

vi.mock('@/composables/sidebarTabs/useModelLibrarySidebarTab', () => ({
  useModelLibrarySidebarTab: () => ({
    id: 'model-library',
    title: 'model-library',
    type: 'vue',
    component: {}
  })
}))

vi.mock(
  '@/platform/workflow/management/composables/useWorkflowsSidebarTab',
  () => ({
    useWorkflowsSidebarTab: () => ({
      id: 'workflows',
      title: 'workflows',
      type: 'vue',
      component: {}
    })
  })
)

describe('useSidebarTabStore', () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    mockGetSetting.mockReset()
    mockRegisterCommand.mockClear()
    mockRegisterCommands.mockClear()
  })

  it('registers the job history tab when QPO V2 is enabled', () => {
    mockGetSetting.mockImplementation((key: string) =>
      key === 'Comfy.Queue.QPOV2' ? true : undefined
    )

    const store = useSidebarTabStore()
    store.registerCoreSidebarTabs()

    expect(store.sidebarTabs.map((tab) => tab.id)).toEqual([
      'job-history',
      'assets',
      'node-library',
      'model-library',
      'workflows'
    ])
    expect(mockRegisterCommand).toHaveBeenCalledTimes(5)
  })

  it('does not register the job history tab when QPO V2 is disabled', () => {
    mockGetSetting.mockImplementation((key: string) =>
      key === 'Comfy.Queue.QPOV2' ? false : undefined
    )

    const store = useSidebarTabStore()
    store.registerCoreSidebarTabs()

    expect(store.sidebarTabs.map((tab) => tab.id)).toEqual([
      'assets',
      'node-library',
      'model-library',
      'workflows'
    ])
    expect(mockRegisterCommand).toHaveBeenCalledTimes(4)
  })

  it('prepends the job history tab when QPO V2 is toggled on', async () => {
    const qpoV2Enabled = ref(false)
    mockGetSetting.mockImplementation((key: string) =>
      key === 'Comfy.Queue.QPOV2' ? qpoV2Enabled.value : undefined
    )

    const store = useSidebarTabStore()
    store.registerCoreSidebarTabs()

    qpoV2Enabled.value = true
    await nextTick()

    expect(store.sidebarTabs.map((tab) => tab.id)).toEqual([
      'job-history',
      'assets',
      'node-library',
      'model-library',
      'workflows'
    ])
    expect(mockRegisterCommand).toHaveBeenCalledTimes(5)
  })
})
