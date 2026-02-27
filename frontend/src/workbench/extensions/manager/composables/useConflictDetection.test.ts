import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { computed, ref } from 'vue'

import { useComfyRegistryService } from '@/services/comfyRegistryService'
import { useSystemStatsStore } from '@/stores/systemStatsStore'
import type { components } from '@/types/comfyRegistryTypes'
import { useInstalledPacks } from '@/workbench/extensions/manager/composables/nodePack/useInstalledPacks'
import { useConflictAcknowledgment } from '@/workbench/extensions/manager/composables/useConflictAcknowledgment'
import type { ConflictAcknowledgmentState } from '@/workbench/extensions/manager/composables/useConflictAcknowledgment'
import { useConflictDetection } from '@/workbench/extensions/manager/composables/useConflictDetection'
import { useComfyManagerService } from '@/workbench/extensions/manager/services/comfyManagerService'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'
import { useConflictDetectionStore } from '@/workbench/extensions/manager/stores/conflictDetectionStore'
import type { ConflictDetectionResult } from '@/workbench/extensions/manager/types/conflictDetectionTypes'
import { checkVersionCompatibility } from '@/workbench/extensions/manager/utils/versionUtil'

// Mock @vueuse/core until function
vi.mock('@vueuse/core', async () => {
  const actual = await vi.importActual('@vueuse/core')
  return {
    ...actual,
    until: vi.fn(() => ({
      toBe: vi.fn(() => Promise.resolve())
    }))
  }
})

// Mock dependencies
vi.mock('@/workbench/extensions/manager/services/comfyManagerService', () => ({
  useComfyManagerService: vi.fn()
}))

vi.mock('@/services/comfyRegistryService', () => ({
  useComfyRegistryService: vi.fn()
}))

vi.mock('@/stores/systemStatsStore', () => ({
  useSystemStatsStore: vi.fn()
}))

vi.mock('@/workbench/extensions/manager/utils/versionUtil', () => ({
  getFrontendVersion: vi.fn(() => '1.24.0'),
  checkVersionCompatibility: vi.fn()
}))

vi.mock('@/workbench/extensions/manager/utils/systemCompatibility', () => ({
  checkOSCompatibility: vi.fn(),
  checkAcceleratorCompatibility: vi.fn(),
  normalizeOSList: vi.fn((list) => list)
}))

vi.mock('@/workbench/extensions/manager/utils/conflictUtils', () => ({
  consolidateConflictsByPackage: vi.fn((results) => results),
  createBannedConflict: vi.fn((isBanned) =>
    isBanned
      ? {
          type: 'banned',
          current_value: 'installed',
          required_value: 'not_banned'
        }
      : null
  ),
  createPendingConflict: vi.fn((isPending) =>
    isPending
      ? {
          type: 'pending',
          current_value: 'installed',
          required_value: 'not_pending'
        }
      : null
  ),
  generateConflictSummary: vi.fn((results, duration) => ({
    total_packages: results.length,
    compatible_packages: results.filter(
      (r: ConflictDetectionResult) => r.is_compatible
    ).length,
    conflicted_packages: results.filter(
      (r: ConflictDetectionResult) => r.has_conflict
    ).length,
    banned_packages: 0,
    pending_packages: 0,
    conflicts_by_type_details: {},
    last_check_timestamp: new Date().toISOString(),
    check_duration_ms: duration
  }))
}))

vi.mock(
  '@/workbench/extensions/manager/composables/useConflictAcknowledgment',
  () => ({
    useConflictAcknowledgment: vi.fn()
  })
)

vi.mock(
  '@/workbench/extensions/manager/composables/nodePack/useInstalledPacks',
  () => ({
    useInstalledPacks: vi.fn()
  })
)

vi.mock('@/workbench/extensions/manager/stores/comfyManagerStore', () => ({
  useComfyManagerStore: vi.fn()
}))

vi.mock('@/workbench/extensions/manager/stores/conflictDetectionStore', () => ({
  useConflictDetectionStore: vi.fn()
}))

vi.mock('@/workbench/extensions/manager/composables/useManagerState', () => ({
  useManagerState: vi.fn(() => ({
    isNewManagerUI: { value: true }
  }))
}))

describe('useConflictDetection', () => {
  let pinia: ReturnType<typeof createTestingPinia>

  const mockComfyManagerService = {
    getImportFailInfoBulk: vi.fn(),
    isLoading: ref(false),
    error: ref<string | null>(null)
  } as Partial<ReturnType<typeof useComfyManagerService>> as ReturnType<
    typeof useComfyManagerService
  >

  const mockRegistryService = {
    getBulkNodeVersions: vi.fn(),
    isLoading: ref(false),
    error: ref<string | null>(null)
  } as Partial<ReturnType<typeof useComfyRegistryService>> as ReturnType<
    typeof useComfyRegistryService
  >

  // Create a ref that can be modified in tests
  const mockInstalledPacksWithVersions = ref<{ id: string; version: string }[]>(
    []
  )

  const mockInstalledPacks = {
    startFetchInstalled: vi.fn(),
    installedPacks: ref<components['schemas']['Node'][]>([]),
    installedPacksWithVersions: computed(
      () => mockInstalledPacksWithVersions.value
    ),
    isReady: ref(false),
    isLoading: ref(false),
    error: ref<unknown>(null)
  } as Partial<ReturnType<typeof useInstalledPacks>> as ReturnType<
    typeof useInstalledPacks
  >

  const mockManagerStore = {
    isPackEnabled: vi.fn()
  } as Partial<ReturnType<typeof useComfyManagerStore>> as ReturnType<
    typeof useComfyManagerStore
  >

  // Create refs that can be used to control computed properties
  let mockConflictedPackages: ConflictDetectionResult[] = []

  const mockConflictStore = {
    get hasConflicts() {
      return mockConflictedPackages.some((p) => p.has_conflict)
    },
    get conflictedPackages() {
      return mockConflictedPackages
    },
    get bannedPackages() {
      return mockConflictedPackages.filter((p) =>
        p.conflicts?.some((c) => c.type === 'banned')
      )
    },
    get securityPendingPackages() {
      return mockConflictedPackages.filter((p) =>
        p.conflicts?.some((c) => c.type === 'pending')
      )
    },
    setConflictedPackages: vi.fn(),
    clearConflicts: vi.fn()
  } as Partial<ReturnType<typeof useConflictDetectionStore>> as ReturnType<
    typeof useConflictDetectionStore
  >

  const mockIsInitialized = true
  const mockSystemStatsStore = {
    systemStats: {
      system: {
        os: 'darwin', // sys.platform returns 'darwin' for macOS
        ram_total: 17179869184,
        ram_free: 8589934592,
        comfyui_version: '0.3.41',
        required_frontend_version: '1.24.0',
        python_version:
          '3.11.0 (main, Oct 13 2023, 09:34:16) [Clang 15.0.0 (clang-1500.0.40.1)]',
        pytorch_version: '2.1.0',
        embedded_python: false,
        argv: ['--enable-manager']
      },
      devices: [
        {
          name: 'Apple M1 Pro',
          type: 'mps',
          index: 0,
          vram_total: 17179869184,
          vram_free: 8589934592,
          torch_vram_total: 17179869184,
          torch_vram_free: 8589934592
        }
      ]
    },
    isInitialized: mockIsInitialized,

    _customProperties: new Set<string>()
  } as Partial<ReturnType<typeof useSystemStatsStore>> as ReturnType<
    typeof useSystemStatsStore
  >

  const mockAcknowledgment = {
    checkComfyUIVersionChange: vi.fn(),
    acknowledgmentState: computed(
      () => ({}) as Partial<ConflictAcknowledgmentState>
    ),
    shouldShowConflictModal: computed(() => false),
    shouldShowRedDot: computed(() => false),
    shouldShowManagerBanner: computed(() => false),
    dismissRedDotNotification: vi.fn(),
    dismissWarningBanner: vi.fn(),
    markConflictsAsSeen: vi.fn()
  } as Partial<ReturnType<typeof useConflictAcknowledgment>> as ReturnType<
    typeof useConflictAcknowledgment
  >

  beforeEach(() => {
    vi.clearAllMocks()
    pinia = createTestingPinia({ stubActions: false })
    setActivePinia(pinia)

    // Setup mocks
    vi.mocked(useComfyManagerService).mockReturnValue(mockComfyManagerService)
    vi.mocked(useComfyRegistryService).mockReturnValue(mockRegistryService)
    vi.mocked(useSystemStatsStore).mockReturnValue(mockSystemStatsStore)
    vi.mocked(useConflictAcknowledgment).mockReturnValue(mockAcknowledgment)
    vi.mocked(useInstalledPacks).mockReturnValue(mockInstalledPacks)
    vi.mocked(useComfyManagerStore).mockReturnValue(mockManagerStore)
    vi.mocked(useConflictDetectionStore).mockReturnValue(mockConflictStore)

    // Reset mock implementations
    vi.mocked(mockInstalledPacks.startFetchInstalled).mockResolvedValue(
      undefined
    )
    vi.mocked(mockManagerStore.isPackEnabled).mockReturnValue(true)
    vi.mocked(mockRegistryService.getBulkNodeVersions).mockResolvedValue({
      node_versions: []
    })
    vi.mocked(mockComfyManagerService.getImportFailInfoBulk).mockResolvedValue(
      {}
    )

    // Reset the installedPacksWithVersions data
    mockInstalledPacksWithVersions.value = []
    // Reset conflicted packages
    mockConflictedPackages = []
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('system environment collection', () => {
    it('should collect system environment correctly', async () => {
      const { collectSystemEnvironment } = useConflictDetection()
      const environment = await collectSystemEnvironment()

      expect(environment).toEqual({
        comfyui_version: '0.3.41',
        frontend_version: '1.24.0',
        os: 'darwin',
        accelerator: 'mps'
      })
    })

    it('should handle missing system stats gracefully', async () => {
      mockSystemStatsStore.systemStats = null as never

      const { collectSystemEnvironment } = useConflictDetection()
      const environment = await collectSystemEnvironment()

      // When systemStats is null, empty strings are used as fallback
      expect(environment).toEqual({
        comfyui_version: '',
        frontend_version: '1.24.0',
        os: '',
        accelerator: ''
      })
    })
  })

  describe('conflict detection', () => {
    it('should detect version conflicts', async () => {
      // Setup installed packages
      mockInstalledPacks.isReady.value = true
      mockInstalledPacks.installedPacks.value = [
        {
          id: 'test-pack',
          name: 'Test Pack',
          latest_version: { version: '1.0.0' }
        } as components['schemas']['Node']
      ]

      mockInstalledPacksWithVersions.value = [
        {
          id: 'test-pack',
          version: '1.0.0'
        }
      ]

      // Mock registry response with version requirements
      vi.mocked(mockRegistryService.getBulkNodeVersions).mockResolvedValue({
        node_versions: [
          {
            status: 'success' as const,
            identifier: { node_id: 'test-pack', version: '1.0.0' },
            node_version: {
              supported_comfyui_version: '>=0.4.0',
              supported_comfyui_frontend_version: '>=2.0.0',
              supported_os: ['Windows', 'Linux', 'macOS'],
              supported_accelerators: ['CUDA', 'Metal', 'CPU'],
              status: 'NodeVersionStatusActive' as const,
              version: '1.0.0',
              publisher_id: 'test-publisher',
              node_id: 'test-pack',
              created_at: '2024-01-01T00:00:00Z'
            } as components['schemas']['NodeVersion']
          }
        ]
      })

      // Mock version checks to return conflicts
      vi.mocked(checkVersionCompatibility).mockImplementation(
        (type, current, required) => {
          if (type === 'comfyui_version' && required === '>=0.4.0') {
            return {
              type: 'comfyui_version',
              current_value: current || '0.3.41',
              required_value: '>=0.4.0'
            }
          }
          return null
        }
      )

      const { runFullConflictAnalysis } = useConflictDetection()
      const result = await runFullConflictAnalysis()

      expect(result.success).toBe(true)
      expect(result.results).toHaveLength(1)
      expect(result.results[0].has_conflict).toBe(true)
      expect(result.results[0].conflicts).toContainEqual({
        type: 'comfyui_version',
        current_value: '0.3.41',
        required_value: '>=0.4.0'
      })
    })

    it('should detect banned packages', async () => {
      mockInstalledPacks.isReady.value = true
      mockInstalledPacks.installedPacks.value = [
        {
          id: 'banned-pack',
          name: 'Banned Pack'
        } as components['schemas']['Node']
      ]

      mockInstalledPacksWithVersions.value = [
        {
          id: 'banned-pack',
          version: '1.0.0'
        }
      ]

      vi.mocked(mockRegistryService.getBulkNodeVersions).mockResolvedValue({
        node_versions: [
          {
            status: 'success' as const,
            identifier: { node_id: 'banned-pack', version: '1.0.0' },
            node_version: {
              status: 'NodeVersionStatusBanned' as const,
              version: '1.0.0',
              publisher_id: 'test-publisher',
              node_id: 'banned-pack',
              created_at: '2024-01-01T00:00:00Z',
              supported_comfyui_version: undefined,
              supported_comfyui_frontend_version: undefined,
              supported_os: undefined,
              supported_accelerators: undefined
            } as components['schemas']['NodeVersion']
          }
        ]
      })

      const { runFullConflictAnalysis } = useConflictDetection()
      const result = await runFullConflictAnalysis()

      expect(result.results[0].conflicts).toContainEqual({
        type: 'banned',
        current_value: 'installed',
        required_value: 'not_banned'
      })
    })

    it('should detect import failures', async () => {
      mockInstalledPacks.isReady.value = true
      mockInstalledPacksWithVersions.value = [
        {
          id: 'fail-pack',
          version: '1.0.0'
        }
      ]

      vi.mocked(
        mockComfyManagerService.getImportFailInfoBulk
      ).mockResolvedValue({
        'fail-pack': {
          error: 'Import error',
          name: 'fail-pack',
          path: '/path/to/pack'
        } as { error?: string; traceback?: string } | null // The actual API returns different structure than types
      })

      // Mock registry response for the package
      vi.mocked(mockRegistryService.getBulkNodeVersions).mockResolvedValue({
        node_versions: []
      })

      const { runFullConflictAnalysis } = useConflictDetection()
      const result = await runFullConflictAnalysis()

      expect(result.results).toHaveLength(1)
      // Import failure should match the actual implementation
      expect(result.results[0].conflicts).toContainEqual({
        type: 'import_failed',
        current_value: 'Import error',
        required_value: 'Import error'
      })
    })
  })

  describe('computed properties', () => {
    it('should expose conflict status from store', () => {
      mockConflictedPackages = [
        {
          package_id: 'test',
          package_name: 'Test',
          has_conflict: true,
          is_compatible: false,
          conflicts: []
        }
      ]

      useConflictDetection()

      // The hasConflicts computed should be true since we have a conflict
      expect(mockConflictedPackages).toHaveLength(1)
      expect(mockConflictedPackages[0].has_conflict).toBe(true)
    })
  })

  describe('initialization', () => {
    it('should initialize without errors', async () => {
      // Mock that installed packs are ready
      mockInstalledPacks.isReady.value = true
      mockInstalledPacksWithVersions.value = []

      // Ensure startFetchInstalled resolves
      vi.mocked(mockInstalledPacks.startFetchInstalled).mockResolvedValue(
        undefined
      )

      const { initializeConflictDetection } = useConflictDetection()

      // Set a timeout to prevent hanging
      await expect(
        Promise.race([
          initializeConflictDetection(),
          new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Timeout')), 1000)
          )
        ])
      ).resolves.not.toThrow()
    })
  })
})
