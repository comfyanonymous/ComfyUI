import { compare, valid } from 'semver'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick, ref } from 'vue'

import type { components } from '@/types/comfyRegistryTypes'
import { useInstalledPacks } from '@/workbench/extensions/manager/composables/nodePack/useInstalledPacks'
import { useUpdateAvailableNodes } from '@/workbench/extensions/manager/composables/nodePack/useUpdateAvailableNodes'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'

// Mock Vue's onMounted to execute immediately for testing
vi.mock('vue', async () => {
  const actual = await vi.importActual('vue')
  return {
    ...actual,
    onMounted: (cb: () => void) => cb()
  }
})

// Mock the dependencies
vi.mock(
  '@/workbench/extensions/manager/composables/nodePack/useInstalledPacks',
  () => ({
    useInstalledPacks: vi.fn()
  })
)

vi.mock('@/workbench/extensions/manager/stores/comfyManagerStore', () => ({
  useComfyManagerStore: vi.fn()
}))

vi.mock('semver', () => ({
  compare: vi.fn(),
  valid: vi.fn()
}))

const mockUseInstalledPacks = vi.mocked(useInstalledPacks)
const mockUseComfyManagerStore = vi.mocked(useComfyManagerStore)

const mockSemverCompare = vi.mocked(compare)
const mockSemverValid = vi.mocked(valid)

type InstalledPacksReturn = ReturnType<typeof useInstalledPacks>
type ManagerStoreReturn = ReturnType<typeof useComfyManagerStore>

function createMockInstalledPacksReturn(
  overrides: Partial<InstalledPacksReturn> = {}
): InstalledPacksReturn {
  return {
    installedPacks: ref([]),
    isLoading: ref(false),
    isReady: ref(false),
    error: ref(null),
    startFetchInstalled: vi.fn(),
    installedPacksWithVersions: ref([]),
    filterInstalledPack: vi.fn(),
    ...overrides
  } as Partial<InstalledPacksReturn> as InstalledPacksReturn
}

function createMockManagerStoreReturn(
  overrides: Partial<ManagerStoreReturn> = {}
): ManagerStoreReturn {
  return {
    ...overrides
  } as Partial<ManagerStoreReturn> as ManagerStoreReturn
}

describe('useUpdateAvailableNodes', () => {
  const mockInstalledPacks = [
    {
      id: 'pack-1',
      name: 'Outdated Pack',
      latest_version: { version: '2.0.0' }
    } as components['schemas']['Node'],
    {
      id: 'pack-2',
      name: 'Up to Date Pack',
      latest_version: { version: '1.0.0' }
    } as components['schemas']['Node'],
    {
      id: 'pack-3',
      name: 'Nightly Pack',
      latest_version: { version: '1.5.0' }
    } as components['schemas']['Node'],
    {
      id: 'pack-4',
      name: 'No Latest Version',
      latest_version: undefined
    } as components['schemas']['Node']
  ]

  const mockStartFetchInstalled = vi.fn()
  const mockIsPackInstalled = vi.fn()
  const mockGetInstalledPackVersion = vi.fn()
  const mockIsPackEnabled = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()

    // Default setup
    mockIsPackInstalled.mockReturnValue(true)
    mockIsPackEnabled.mockReturnValue(true) // Default: all packs are enabled
    mockGetInstalledPackVersion.mockImplementation((id: string) => {
      switch (id) {
        case 'pack-1':
          return '1.0.0' // outdated
        case 'pack-2':
          return '1.0.0' // up to date
        case 'pack-3':
          return 'nightly-abc123' // nightly
        case 'pack-4':
          return '1.0.0' // no latest version
        default:
          return '1.0.0'
      }
    })

    mockSemverValid.mockImplementation((version) => {
      return version &&
        typeof version === 'string' &&
        !version.includes('nightly')
        ? version
        : null
    })

    mockSemverCompare.mockImplementation((latest, installed) => {
      if (latest === '2.0.0' && installed === '1.0.0') return 1 // outdated
      if (latest === '1.0.0' && installed === '1.0.0') return 0 // up to date
      return 0
    })

    mockUseComfyManagerStore.mockReturnValue(
      createMockManagerStoreReturn({
        isPackInstalled: mockIsPackInstalled,
        getInstalledPackVersion: mockGetInstalledPackVersion,
        isPackEnabled: mockIsPackEnabled
      })
    )

    mockUseInstalledPacks.mockReturnValue(
      createMockInstalledPacksReturn({
        startFetchInstalled: mockStartFetchInstalled
      })
    )
  })

  describe('core filtering logic', () => {
    it('identifies outdated packs correctly', () => {
      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          installedPacks: ref(mockInstalledPacks),
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      const { updateAvailableNodePacks } = useUpdateAvailableNodes()

      // Should only include pack-1 (outdated)
      expect(updateAvailableNodePacks.value).toHaveLength(1)
      expect(updateAvailableNodePacks.value[0].id).toBe('pack-1')
    })

    it('excludes up-to-date packs', () => {
      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          installedPacks: ref([mockInstalledPacks[1]]), // pack-2: up to date
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      const { updateAvailableNodePacks } = useUpdateAvailableNodes()

      expect(updateAvailableNodePacks.value).toHaveLength(0)
    })

    it('excludes nightly packs from updates', () => {
      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          installedPacks: ref([mockInstalledPacks[2]]), // pack-3: nightly
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      const { updateAvailableNodePacks } = useUpdateAvailableNodes()

      expect(updateAvailableNodePacks.value).toHaveLength(0)
    })

    it('excludes packs with no latest version', () => {
      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          installedPacks: ref([mockInstalledPacks[3]]), // pack-4: no latest version
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      const { updateAvailableNodePacks } = useUpdateAvailableNodes()

      expect(updateAvailableNodePacks.value).toHaveLength(0)
    })

    it('excludes uninstalled packs', () => {
      mockIsPackInstalled.mockReturnValue(false)
      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          installedPacks: ref(mockInstalledPacks),
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      const { updateAvailableNodePacks } = useUpdateAvailableNodes()

      expect(updateAvailableNodePacks.value).toHaveLength(0)
    })

    it('returns empty array when no installed packs exist', () => {
      const { updateAvailableNodePacks } = useUpdateAvailableNodes()

      expect(updateAvailableNodePacks.value).toEqual([])
    })
  })

  describe('hasUpdateAvailable computed', () => {
    it('returns true when updates are available', () => {
      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          installedPacks: ref([mockInstalledPacks[0]]), // pack-1: outdated
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      const { hasUpdateAvailable } = useUpdateAvailableNodes()

      expect(hasUpdateAvailable.value).toBe(true)
    })

    it('returns false when no updates are available', () => {
      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          installedPacks: ref([mockInstalledPacks[1]]), // pack-2: up to date
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      const { hasUpdateAvailable } = useUpdateAvailableNodes()

      expect(hasUpdateAvailable.value).toBe(false)
    })
  })

  describe('automatic data fetching', () => {
    it('fetches installed packs automatically when none exist', () => {
      useUpdateAvailableNodes()

      expect(mockStartFetchInstalled).toHaveBeenCalledOnce()
    })

    it('does not fetch when packs already exist', () => {
      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          installedPacks: ref(mockInstalledPacks),
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      useUpdateAvailableNodes()

      expect(mockStartFetchInstalled).not.toHaveBeenCalled()
    })

    it('does not fetch when already loading', () => {
      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          isLoading: ref(true),
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      useUpdateAvailableNodes()

      expect(mockStartFetchInstalled).not.toHaveBeenCalled()
    })
  })

  describe('state management', () => {
    it('exposes loading state from useInstalledPacks', () => {
      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          isLoading: ref(true),
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      const { isLoading } = useUpdateAvailableNodes()

      expect(isLoading.value).toBe(true)
    })

    it('exposes error state from useInstalledPacks', () => {
      const testError = 'Failed to fetch installed packs'
      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          error: ref(testError),
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      const { error } = useUpdateAvailableNodes()

      expect(error.value).toBe(testError)
    })
  })

  describe('reactivity', () => {
    it('updates when installed packs change', async () => {
      const installedPacksRef = ref<components['schemas']['Node'][]>([])
      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          installedPacks: installedPacksRef,
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      const { updateAvailableNodePacks, hasUpdateAvailable } =
        useUpdateAvailableNodes()

      // Initially empty
      expect(updateAvailableNodePacks.value).toEqual([])
      expect(hasUpdateAvailable.value).toBe(false)

      // Update installed packs
      installedPacksRef.value = [mockInstalledPacks[0]]
      await nextTick()

      // Should update available updates
      expect(updateAvailableNodePacks.value).toHaveLength(1)
      expect(hasUpdateAvailable.value).toBe(true)
    })
  })

  describe('version comparison logic', () => {
    it('calls compareVersions with correct parameters', () => {
      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          installedPacks: ref([mockInstalledPacks[0]]), // pack-1
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      const { updateAvailableNodePacks } = useUpdateAvailableNodes()

      // Access the computed to trigger the logic
      expect(updateAvailableNodePacks.value).toBeDefined()

      expect(mockSemverCompare).toHaveBeenCalledWith('2.0.0', '1.0.0')
    })

    it('calls semver.valid to check nightly versions', () => {
      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          installedPacks: ref([mockInstalledPacks[2]]), // pack-3: nightly
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      const { updateAvailableNodePacks } = useUpdateAvailableNodes()

      // Access the computed to trigger the logic
      expect(updateAvailableNodePacks.value).toBeDefined()

      expect(mockSemverValid).toHaveBeenCalledWith('nightly-abc123')
    })

    it('calls isPackInstalled for each pack', () => {
      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          installedPacks: ref(mockInstalledPacks),
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      const { updateAvailableNodePacks } = useUpdateAvailableNodes()

      // Access the computed to trigger the logic
      expect(updateAvailableNodePacks.value).toBeDefined()

      expect(mockIsPackInstalled).toHaveBeenCalledWith('pack-1')
      expect(mockIsPackInstalled).toHaveBeenCalledWith('pack-2')
      expect(mockIsPackInstalled).toHaveBeenCalledWith('pack-3')
      expect(mockIsPackInstalled).toHaveBeenCalledWith('pack-4')
    })
  })

  describe('enabledUpdateAvailableNodePacks', () => {
    it('returns only enabled packs with updates', () => {
      mockIsPackEnabled.mockImplementation((id: string) => {
        // pack-1 is disabled
        return id !== 'pack-1'
      })

      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          installedPacks: ref([mockInstalledPacks[0], mockInstalledPacks[1]]),
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      const { updateAvailableNodePacks, enabledUpdateAvailableNodePacks } =
        useUpdateAvailableNodes()

      // pack-1 has updates but is disabled
      expect(updateAvailableNodePacks.value).toHaveLength(1)
      expect(updateAvailableNodePacks.value[0].id).toBe('pack-1')

      // enabledUpdateAvailableNodePacks should be empty
      expect(enabledUpdateAvailableNodePacks.value).toHaveLength(0)
    })

    it('returns all packs when all are enabled', () => {
      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          installedPacks: ref([mockInstalledPacks[0]]), // pack-1: outdated
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      const { updateAvailableNodePacks, enabledUpdateAvailableNodePacks } =
        useUpdateAvailableNodes()

      expect(updateAvailableNodePacks.value).toHaveLength(1)
      expect(enabledUpdateAvailableNodePacks.value).toHaveLength(1)
      expect(enabledUpdateAvailableNodePacks.value[0].id).toBe('pack-1')
    })
  })

  describe('hasDisabledUpdatePacks', () => {
    it('returns true when there are disabled packs with updates', () => {
      mockIsPackEnabled.mockImplementation((id: string) => {
        // pack-1 is disabled
        return id !== 'pack-1'
      })

      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          installedPacks: ref([mockInstalledPacks[0]]), // pack-1: outdated
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      const { hasDisabledUpdatePacks } = useUpdateAvailableNodes()

      expect(hasDisabledUpdatePacks.value).toBe(true)
    })

    it('returns false when all packs with updates are enabled', () => {
      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          installedPacks: ref([mockInstalledPacks[0]]), // pack-1: outdated
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      const { hasDisabledUpdatePacks } = useUpdateAvailableNodes()

      expect(hasDisabledUpdatePacks.value).toBe(false)
    })

    it('returns false when no packs have updates', () => {
      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          installedPacks: ref([mockInstalledPacks[1]]), // pack-2: up to date
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      const { hasDisabledUpdatePacks } = useUpdateAvailableNodes()

      expect(hasDisabledUpdatePacks.value).toBe(false)
    })
  })

  describe('hasUpdateAvailable with disabled packs', () => {
    it('returns false when only disabled packs have updates', () => {
      mockIsPackEnabled.mockReturnValue(false) // All packs disabled

      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          installedPacks: ref([mockInstalledPacks[0]]), // pack-1: outdated
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      const { hasUpdateAvailable } = useUpdateAvailableNodes()

      expect(hasUpdateAvailable.value).toBe(false)
    })

    it('returns true when at least one enabled pack has updates', () => {
      mockIsPackEnabled.mockImplementation((id: string) => {
        // Only pack-1 is enabled
        return id === 'pack-1'
      })

      mockUseInstalledPacks.mockReturnValue(
        createMockInstalledPacksReturn({
          installedPacks: ref([mockInstalledPacks[0]]), // pack-1: outdated
          startFetchInstalled: mockStartFetchInstalled
        })
      )

      const { hasUpdateAvailable } = useUpdateAvailableNodes()

      expect(hasUpdateAvailable.value).toBe(true)
    })
  })
})
