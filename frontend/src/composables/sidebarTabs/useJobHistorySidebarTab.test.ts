import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useJobHistorySidebarTab } from '@/composables/sidebarTabs/useJobHistorySidebarTab'

const { mockActiveJobsCount, mockActiveSidebarTabId } = vi.hoisted(() => ({
  mockActiveJobsCount: { value: 0 },
  mockActiveSidebarTabId: { value: null as string | null }
}))

vi.mock('@/components/sidebar/tabs/JobHistorySidebarTab.vue', () => ({
  default: {}
}))

vi.mock('@/stores/queueStore', () => ({
  useQueueStore: () => ({
    activeJobsCount: mockActiveJobsCount.value
  })
}))

vi.mock('@/stores/workspace/sidebarTabStore', () => ({
  useSidebarTabStore: () => ({
    activeSidebarTabId: mockActiveSidebarTabId.value
  })
}))

describe('useJobHistorySidebarTab', () => {
  beforeEach(() => {
    mockActiveSidebarTabId.value = null
    mockActiveJobsCount.value = 0
  })

  it('shows active jobs count while the panel is closed', () => {
    mockActiveSidebarTabId.value = 'assets'
    mockActiveJobsCount.value = 3

    const sidebarTab = useJobHistorySidebarTab()

    expect(typeof sidebarTab.iconBadge).toBe('function')
    expect((sidebarTab.iconBadge as () => string | null)()).toBe('3')
  })

  it('hides badge while the job history panel is open', () => {
    mockActiveSidebarTabId.value = 'job-history'
    mockActiveJobsCount.value = 3

    const sidebarTab = useJobHistorySidebarTab()

    expect((sidebarTab.iconBadge as () => string | null)()).toBeNull()
  })

  it('hides badge when there are no active jobs', () => {
    mockActiveSidebarTabId.value = null
    mockActiveJobsCount.value = 0

    const sidebarTab = useJobHistorySidebarTab()

    expect((sidebarTab.iconBadge as () => string | null)()).toBeNull()
  })
})
