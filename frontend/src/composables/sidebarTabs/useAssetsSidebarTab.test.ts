import { describe, expect, it, vi } from 'vitest'

import { useAssetsSidebarTab } from '@/composables/sidebarTabs/useAssetsSidebarTab'

const { mockGetSetting, mockUnseenAddedAssetsCount } = vi.hoisted(() => ({
  mockGetSetting: vi.fn(),
  mockUnseenAddedAssetsCount: { value: 0 }
}))

vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: () => ({
    get: mockGetSetting
  })
}))

vi.mock('@/components/sidebar/tabs/AssetsSidebarTab.vue', () => ({
  default: {}
}))

vi.mock('@/stores/workspace/assetsSidebarBadgeStore', () => ({
  useAssetsSidebarBadgeStore: () => ({
    unseenAddedAssetsCount: mockUnseenAddedAssetsCount.value
  })
}))

describe('useAssetsSidebarTab', () => {
  it('hides icon badge when QPO V2 is disabled', () => {
    mockGetSetting.mockReturnValue(false)
    mockUnseenAddedAssetsCount.value = 3

    const sidebarTab = useAssetsSidebarTab()

    expect(typeof sidebarTab.iconBadge).toBe('function')
    expect((sidebarTab.iconBadge as () => string | null)()).toBeNull()
  })

  it('shows unseen added assets count when QPO V2 is enabled', () => {
    mockGetSetting.mockReturnValue(true)
    mockUnseenAddedAssetsCount.value = 3

    const sidebarTab = useAssetsSidebarTab()

    expect(typeof sidebarTab.iconBadge).toBe('function')
    expect((sidebarTab.iconBadge as () => string | null)()).toBe('3')
  })

  it('hides badge when there are no unseen added assets', () => {
    mockGetSetting.mockReturnValue(true)
    mockUnseenAddedAssetsCount.value = 0

    const sidebarTab = useAssetsSidebarTab()

    expect((sidebarTab.iconBadge as () => string | null)()).toBeNull()
  })
})
