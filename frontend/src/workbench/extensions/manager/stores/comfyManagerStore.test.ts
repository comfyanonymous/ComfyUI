import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick, ref } from 'vue'

import { useComfyManagerService } from '@/workbench/extensions/manager/services/comfyManagerService'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'
import type { components as ManagerComponents } from '@/workbench/extensions/manager/types/generatedManagerTypes'

type InstalledPacksResponse =
  ManagerComponents['schemas']['InstalledPacksResponse']
type ManagerChannel = ManagerComponents['schemas']['ManagerChannel']
type ManagerDatabaseSource =
  ManagerComponents['schemas']['ManagerDatabaseSource']
type ManagerPackInstalled = ManagerComponents['schemas']['ManagerPackInstalled']

vi.mock('@/workbench/extensions/manager/services/comfyManagerService', () => ({
  useComfyManagerService: vi.fn()
}))

vi.mock('@/workbench/extensions/manager/composables/useManagerQueue', () => {
  const enqueueTaskMock = vi.fn()

  return {
    useManagerQueue: () => ({
      statusMessage: ref(''),
      allTasksDone: ref(false),
      enqueueTask: enqueueTaskMock,
      isProcessingTasks: ref(false)
    }),
    enqueueTask: enqueueTaskMock
  }
})

vi.mock('@/composables/useServerLogs', () => ({
  useServerLogs: () => ({
    startListening: vi.fn(),
    stopListening: vi.fn(),
    logs: ref([])
  })
}))

vi.mock('vue-i18n', () => ({
  useI18n: () => ({
    t: vi.fn((key) => key)
  }),
  createI18n: vi.fn(() => ({
    global: {
      t: vi.fn((key) => key)
    }
  }))
}))

interface EnabledDisabledTestCase {
  desc: string
  installed: Record<string, ManagerPackInstalled>
  expectState: 'enabled' | 'disabled'
  /** @default 'name' */
  packName?: string
}

describe('useComfyManagerStore', () => {
  let mockManagerService: ReturnType<typeof useComfyManagerService>

  const triggerPacksChange = async (
    installedPacks: InstalledPacksResponse,
    store: ReturnType<typeof useComfyManagerStore>
  ) => {
    // Simulate change in value to properly trigger watchers. Required even for immediate watchers.
    store.installedPacks = {}
    await nextTick()
    store.installedPacks = installedPacks
  }

  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    vi.clearAllMocks()
    mockManagerService = {
      isLoading: ref(false),
      error: ref(null),
      startQueue: vi.fn().mockResolvedValue(null),
      getQueueStatus: vi.fn().mockResolvedValue(null),
      getTaskHistory: vi.fn().mockResolvedValue(null),
      listInstalledPacks: vi.fn().mockResolvedValue({}),
      getImportFailInfo: vi.fn().mockResolvedValue(null),
      getImportFailInfoBulk: vi.fn().mockResolvedValue({}),
      installPack: vi.fn().mockResolvedValue(null),
      uninstallPack: vi.fn().mockResolvedValue(null),
      enablePack: vi.fn().mockResolvedValue(null),
      disablePack: vi.fn().mockResolvedValue(null),
      updatePack: vi.fn().mockResolvedValue(null),
      updateAllPacks: vi.fn().mockResolvedValue(null),
      updateComfyUI: vi.fn().mockResolvedValue(null),
      rebootComfyUI: vi.fn().mockResolvedValue(null),
      isLegacyManagerUI: vi.fn().mockResolvedValue(false)
    }

    vi.mocked(useComfyManagerService).mockReturnValue(mockManagerService)
  })

  const testCases: EnabledDisabledTestCase[] = [
    {
      desc: 'Two enabled versions',
      installed: {
        'name@1_0_2': {
          enabled: true,
          cnr_id: 'name',
          ver: '1.0.2',
          aux_id: undefined
        },
        name: { enabled: true, cnr_id: 'name', ver: '1.0.0', aux_id: undefined }
      },
      expectState: 'enabled'
    },
    {
      desc: 'Two disabled versions',
      installed: {
        'name@1_0_2': {
          enabled: false,
          cnr_id: 'name',
          ver: '1.0.2',
          aux_id: undefined
        },
        name: {
          enabled: false,
          cnr_id: 'name',
          ver: '1.0.0',
          aux_id: undefined
        }
      },
      expectState: 'disabled'
    },
    {
      desc: 'Enabled version and pinned disabled version',
      installed: {
        'name@1_0_2': {
          enabled: false,
          cnr_id: 'name',
          ver: '1.0.2',
          aux_id: undefined
        },
        name: { enabled: true, cnr_id: 'name', ver: '1.0.0', aux_id: undefined }
      },
      expectState: 'enabled'
    },
    {
      desc: 'Disabled version and pinned enabled version',
      installed: {
        'name@1_0_2': {
          enabled: true,
          cnr_id: 'name',
          ver: '1.0.2',
          aux_id: undefined
        },
        name: {
          enabled: false,
          cnr_id: 'name',
          ver: '1.0.0',
          aux_id: undefined
        }
      },
      expectState: 'enabled'
    },
    {
      desc: 'Pinned enabled version, Pinned disabled version',
      installed: {
        'name@1_0_2': {
          enabled: false,
          cnr_id: 'name',
          ver: '1.0.2',
          aux_id: undefined
        },
        'name@1_0_3': {
          enabled: true,
          cnr_id: 'name',
          ver: '1.0.3',
          aux_id: undefined
        }
      },
      expectState: 'enabled'
    },
    {
      desc: 'Two enabled non-CNR versions',
      packName: 'author/name',
      installed: {
        'author/name@1_0_2': {
          enabled: true,
          aux_id: 'author/name',
          cnr_id: undefined,
          ver: '1.0.2'
        },
        'author/name': {
          enabled: true,
          aux_id: 'author/name',
          cnr_id: undefined,
          ver: '1.0.0'
        }
      },
      expectState: 'enabled'
    },
    {
      desc: 'Two disabled non-CNR versions',
      packName: 'author/name',
      installed: {
        'author/name@1_0_2': {
          enabled: false,
          aux_id: 'author/name',
          cnr_id: undefined,
          ver: '1.0.2'
        },
        'author/name': {
          enabled: false,
          aux_id: 'author/name',
          cnr_id: undefined,
          ver: '1.0.0'
        }
      },
      expectState: 'disabled'
    },
    {
      desc: 'Non-CNR disabled version, CNR enabled version',
      packName: 'author/name',
      installed: {
        'author/name': {
          enabled: false,
          aux_id: 'author/name',
          cnr_id: undefined,
          ver: '1.0.0'
        },
        'author/name@1_0_2': {
          enabled: true,
          cnr_id: 'author/name',
          ver: '1.0.2',
          aux_id: undefined
        }
      },
      expectState: 'enabled'
    },
    {
      desc: 'Disabled non-CNR version, CNR disabled version',
      packName: 'author/name',
      installed: {
        'author/name': {
          enabled: false,
          aux_id: 'author/name',
          cnr_id: undefined, // non-CNR pack
          ver: '1.0.0'
        },
        'author/name@1_0_2': {
          enabled: false,
          cnr_id: 'author/name', // CNR pack
          ver: '1.0.2',
          aux_id: undefined
        }
      },
      expectState: 'disabled'
    },
    {
      desc: 'Enabled non-CNR version, two versions',
      packName: 'author/name',
      installed: {
        'author/name': {
          enabled: true,
          aux_id: 'author/name',
          cnr_id: undefined,
          ver: '1.0.0'
        },
        'author/name@1_0_2': {
          enabled: true,
          cnr_id: 'author/name',
          ver: '1.0.2',
          aux_id: undefined
        }
      },
      expectState: 'enabled'
    },
    {
      desc: 'Enabled CNR version',
      packName: 'name',
      installed: {
        name: { enabled: true, cnr_id: 'name', ver: '1.0.0', aux_id: undefined }
      },
      expectState: 'enabled'
    },
    {
      desc: 'Disabled CNR version',
      packName: 'name',
      installed: {
        name: {
          enabled: false,
          cnr_id: 'name',
          ver: '1.0.0',
          aux_id: undefined
        }
      },
      expectState: 'disabled'
    },
    {
      desc: 'Enabled non-CNR version',
      packName: 'author/name',
      installed: {
        'author/name': {
          enabled: true,
          aux_id: 'author/name',
          cnr_id: undefined,
          ver: '1.0.0'
        }
      },
      expectState: 'enabled'
    },
    {
      desc: 'Disabled non-CNR version',
      packName: 'author/name',
      installed: {
        'author/name': {
          enabled: false,
          aux_id: 'author/name',
          cnr_id: undefined,
          ver: '1.0.0'
        }
      },
      expectState: 'disabled'
    },
    {
      desc: 'Pack not installed',
      installed: {
        'a different pack': {
          enabled: true,
          cnr_id: 'a different pack',
          ver: '1.0.0',
          aux_id: undefined
        }
      },
      expectState: 'disabled'
    }
  ]

  describe('isPackEnabled', () => {
    it.each(testCases)(
      '$expectState when $desc',
      async ({ installed, expectState, packName }) => {
        packName ??= 'name'

        const store = useComfyManagerStore()
        await triggerPacksChange(installed, store)

        const enabled = expectState === 'enabled'
        expect(store.isPackEnabled(packName)).toBe(enabled)
      }
    )
  })

  describe.skip('isPackInstalling', () => {
    it('should return false for packs not being installed', () => {
      const store = useComfyManagerStore()
      expect(store.isPackInstalling('test-pack')).toBe(false)
      expect(store.isPackInstalling(undefined)).toBe(false)
      expect(store.isPackInstalling('')).toBe(false)
    })

    it('should track pack as installing when installPack is called', async () => {
      const store = useComfyManagerStore()

      // Call installPack
      await store.installPack.call({
        id: 'test-pack',
        repository: 'https://github.com/test/test-pack',
        channel: 'dev' as ManagerChannel,
        mode: 'cache' as ManagerDatabaseSource,
        selected_version: 'latest',
        version: 'latest'
      })

      // Check that the pack is marked as installing
      expect(store.isPackInstalling('test-pack')).toBe(true)
    })

    it('should remove pack from installing list when explicitly removed', async () => {
      const store = useComfyManagerStore()

      // Call installPack
      await store.installPack.call({
        id: 'test-pack',
        repository: 'https://github.com/test/test-pack',
        channel: 'dev' as ManagerChannel,
        mode: 'cache' as ManagerDatabaseSource,
        selected_version: 'latest',
        version: 'latest'
      })

      // Verify pack is installing
      expect(store.isPackInstalling('test-pack')).toBe(true)

      // Call installPack again for another pack to demonstrate multiple installs
      await store.installPack.call({
        id: 'another-pack',
        repository: 'https://github.com/test/another-pack',
        channel: 'dev' as ManagerChannel,
        mode: 'cache' as ManagerDatabaseSource,
        selected_version: 'latest',
        version: 'latest'
      })

      // Both should be installing
      expect(store.isPackInstalling('test-pack')).toBe(true)
      expect(store.isPackInstalling('another-pack')).toBe(true)
    })

    it('should track multiple packs installing independently', async () => {
      const store = useComfyManagerStore()

      // Install pack 1
      await store.installPack.call({
        id: 'pack-1',
        repository: 'https://github.com/test/pack-1',
        channel: 'dev' as ManagerChannel,
        mode: 'cache' as ManagerDatabaseSource,
        selected_version: 'latest',
        version: 'latest'
      })

      // Install pack 2
      await store.installPack.call({
        id: 'pack-2',
        repository: 'https://github.com/test/pack-2',
        channel: 'dev' as ManagerChannel,
        mode: 'cache' as ManagerDatabaseSource,
        selected_version: 'latest',
        version: 'latest'
      })

      // Both should be installing
      expect(store.isPackInstalling('pack-1')).toBe(true)
      expect(store.isPackInstalling('pack-2')).toBe(true)
      expect(store.isPackInstalling('pack-3')).toBe(false)
    })
  })

  describe('refreshInstalledList with pack ID normalization', () => {
    it('normalizes pack IDs by removing version suffixes', async () => {
      const mockPacks = {
        'ComfyUI-GGUF@1_1_4': {
          enabled: false,
          cnr_id: 'ComfyUI-GGUF',
          ver: '1.1.4',
          aux_id: undefined
        },
        'ComfyUI-Manager': {
          enabled: true,
          cnr_id: 'ComfyUI-Manager',
          ver: '2.0.0',
          aux_id: undefined
        }
      }

      vi.mocked(mockManagerService.listInstalledPacks).mockResolvedValue(
        mockPacks
      )

      const store = useComfyManagerStore()
      await store.refreshInstalledList()

      // Both packs should be accessible by their base name
      expect(store.installedPacks['ComfyUI-GGUF']).toEqual({
        enabled: false,
        cnr_id: 'ComfyUI-GGUF',
        ver: '1.1.4',
        aux_id: undefined
      })
      expect(store.installedPacks['ComfyUI-Manager']).toEqual({
        enabled: true,
        cnr_id: 'ComfyUI-Manager',
        ver: '2.0.0',
        aux_id: undefined
      })

      // Version suffixed keys should not exist
      expect(store.installedPacks['ComfyUI-GGUF@1_1_4']).toBeUndefined()
    })

    it('handles duplicate keys after normalization', async () => {
      const mockPacks = {
        'test-pack': {
          enabled: true,
          cnr_id: 'test-pack',
          ver: '1.0.0',
          aux_id: undefined
        },
        'test-pack@1_1_0': {
          enabled: false,
          cnr_id: 'test-pack',
          ver: '1.1.0',
          aux_id: undefined
        }
      }

      vi.mocked(mockManagerService.listInstalledPacks).mockResolvedValue(
        mockPacks
      )

      const store = useComfyManagerStore()
      await store.refreshInstalledList()

      // The normalized key should exist (last one wins with mapKeys)
      expect(store.installedPacks['test-pack']).toBeDefined()
      expect(store.installedPacks['test-pack'].ver).toBe('1.1.0')
    })

    it('preserves version information for disabled packs', async () => {
      const mockPacks = {
        'disabled-pack@2_0_0': {
          enabled: false,
          cnr_id: 'disabled-pack',
          ver: '2.0.0',
          aux_id: undefined
        }
      }

      vi.mocked(mockManagerService.listInstalledPacks).mockResolvedValue(
        mockPacks
      )

      const store = useComfyManagerStore()
      await store.refreshInstalledList()

      // Pack should be accessible by base name with version preserved
      expect(store.getInstalledPackVersion('disabled-pack')).toBe('2.0.0')
      expect(store.isPackInstalled('disabled-pack')).toBe(true)
    })
  })
})
