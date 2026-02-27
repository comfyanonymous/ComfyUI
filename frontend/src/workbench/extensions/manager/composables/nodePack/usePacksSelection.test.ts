import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

import type { components } from '@/types/comfyRegistryTypes'
import { usePacksSelection } from '@/workbench/extensions/manager/composables/nodePack/usePacksSelection'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'

vi.mock('vue-i18n', async () => {
  const actual = await vi.importActual('vue-i18n')
  return {
    ...actual,
    useI18n: () => ({
      t: vi.fn((key) => key)
    })
  }
})

type NodePack = components['schemas']['Node']

describe('usePacksSelection', () => {
  let managerStore: ReturnType<typeof useComfyManagerStore>
  let mockIsPackInstalled: (packName: string | undefined) => boolean

  const createMockPack = (id: string): NodePack => ({
    id,
    name: `Pack ${id}`,
    description: `Description for pack ${id}`,
    category: 'Nodes',
    author: 'Test Author',
    license: 'MIT',
    repository: 'https://github.com/test/pack',
    tags: [],
    status: 'NodeStatusActive'
  })

  beforeEach(() => {
    vi.clearAllMocks()
    const pinia = createTestingPinia({ stubActions: false })
    setActivePinia(pinia)

    managerStore = useComfyManagerStore()

    // Mock the isPackInstalled method
    mockIsPackInstalled = vi.fn()
    managerStore.isPackInstalled = mockIsPackInstalled
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('installedPacks', () => {
    it('should filter and return only installed packs', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2'),
        createMockPack('pack3')
      ])

      vi.mocked(mockIsPackInstalled).mockImplementation((id) => {
        return id === 'pack1' || id === 'pack3'
      })

      const { installedPacks } = usePacksSelection(nodePacks)

      expect(installedPacks.value).toHaveLength(2)
      expect(installedPacks.value[0].id).toBe('pack1')
      expect(installedPacks.value[1].id).toBe('pack3')
      expect(mockIsPackInstalled).toHaveBeenCalledTimes(3)
    })

    it('should return empty array when no packs are installed', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2')
      ])

      vi.mocked(mockIsPackInstalled).mockReturnValue(false)

      const { installedPacks } = usePacksSelection(nodePacks)

      expect(installedPacks.value).toHaveLength(0)
    })

    it('should update when nodePacks ref changes', () => {
      const nodePacks = ref<NodePack[]>([createMockPack('pack1')])
      vi.mocked(mockIsPackInstalled).mockReturnValue(true)

      const { installedPacks } = usePacksSelection(nodePacks)
      expect(installedPacks.value).toHaveLength(1)

      // Add more packs
      nodePacks.value = [
        createMockPack('pack1'),
        createMockPack('pack2'),
        createMockPack('pack3')
      ]

      expect(installedPacks.value).toHaveLength(3)
    })
  })

  describe('notInstalledPacks', () => {
    it('should filter and return only not installed packs', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2'),
        createMockPack('pack3')
      ])

      vi.mocked(mockIsPackInstalled).mockImplementation((id) => {
        return id === 'pack1'
      })

      const { notInstalledPacks } = usePacksSelection(nodePacks)

      expect(notInstalledPacks.value).toHaveLength(2)
      expect(notInstalledPacks.value[0].id).toBe('pack2')
      expect(notInstalledPacks.value[1].id).toBe('pack3')
    })

    it('should return all packs when none are installed', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2')
      ])

      vi.mocked(mockIsPackInstalled).mockReturnValue(false)

      const { notInstalledPacks } = usePacksSelection(nodePacks)

      expect(notInstalledPacks.value).toHaveLength(2)
    })
  })

  describe('isAllInstalled', () => {
    it('should return true when all packs are installed', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2')
      ])

      vi.mocked(mockIsPackInstalled).mockReturnValue(true)

      const { isAllInstalled } = usePacksSelection(nodePacks)

      expect(isAllInstalled.value).toBe(true)
    })

    it('should return false when not all packs are installed', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2')
      ])

      vi.mocked(mockIsPackInstalled).mockImplementation((id) => id === 'pack1')

      const { isAllInstalled } = usePacksSelection(nodePacks)

      expect(isAllInstalled.value).toBe(false)
    })

    it('should return true for empty array', () => {
      const nodePacks = ref<NodePack[]>([])

      const { isAllInstalled } = usePacksSelection(nodePacks)

      expect(isAllInstalled.value).toBe(true)
    })
  })

  describe('isNoneInstalled', () => {
    it('should return true when no packs are installed', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2')
      ])

      vi.mocked(mockIsPackInstalled).mockReturnValue(false)

      const { isNoneInstalled } = usePacksSelection(nodePacks)

      expect(isNoneInstalled.value).toBe(true)
    })

    it('should return false when some packs are installed', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2')
      ])

      vi.mocked(mockIsPackInstalled).mockImplementation((id) => id === 'pack1')

      const { isNoneInstalled } = usePacksSelection(nodePacks)

      expect(isNoneInstalled.value).toBe(false)
    })

    it('should return true for empty array', () => {
      const nodePacks = ref<NodePack[]>([])

      const { isNoneInstalled } = usePacksSelection(nodePacks)

      expect(isNoneInstalled.value).toBe(true)
    })
  })

  describe('isMixed', () => {
    it('should return true when some but not all packs are installed', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2'),
        createMockPack('pack3')
      ])

      vi.mocked(mockIsPackInstalled).mockImplementation((id) => {
        return id === 'pack1' || id === 'pack2'
      })

      const { isMixed } = usePacksSelection(nodePacks)

      expect(isMixed.value).toBe(true)
    })

    it('should return false when all packs are installed', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2')
      ])

      vi.mocked(mockIsPackInstalled).mockReturnValue(true)

      const { isMixed } = usePacksSelection(nodePacks)

      expect(isMixed.value).toBe(false)
    })

    it('should return false when no packs are installed', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2')
      ])

      vi.mocked(mockIsPackInstalled).mockReturnValue(false)

      const { isMixed } = usePacksSelection(nodePacks)

      expect(isMixed.value).toBe(false)
    })

    it('should return false for empty array', () => {
      const nodePacks = ref<NodePack[]>([])

      const { isMixed } = usePacksSelection(nodePacks)

      expect(isMixed.value).toBe(false)
    })
  })

  describe('selectionState', () => {
    it('should return "all-installed" when all packs are installed', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2')
      ])

      vi.mocked(mockIsPackInstalled).mockReturnValue(true)

      const { selectionState } = usePacksSelection(nodePacks)

      expect(selectionState.value).toBe('all-installed')
    })

    it('should return "none-installed" when no packs are installed', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2')
      ])

      vi.mocked(mockIsPackInstalled).mockReturnValue(false)

      const { selectionState } = usePacksSelection(nodePacks)

      expect(selectionState.value).toBe('none-installed')
    })

    it('should return "mixed" when some packs are installed', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2'),
        createMockPack('pack3')
      ])

      vi.mocked(mockIsPackInstalled).mockImplementation((id) => id === 'pack1')

      const { selectionState } = usePacksSelection(nodePacks)

      expect(selectionState.value).toBe('mixed')
    })

    it('should update when installation status changes', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2')
      ])

      vi.mocked(mockIsPackInstalled).mockReturnValue(false)

      const { selectionState } = usePacksSelection(nodePacks)
      expect(selectionState.value).toBe('none-installed')

      // Change mock to simulate installation
      vi.mocked(mockIsPackInstalled).mockReturnValue(true)

      // Force reactivity update
      nodePacks.value = [...nodePacks.value]

      expect(selectionState.value).toBe('all-installed')
    })
  })

  describe('edge cases', () => {
    it('should handle packs with undefined ids', () => {
      const nodePacks = ref<NodePack[]>([
        { ...createMockPack('pack1'), id: undefined },
        createMockPack('pack2')
      ])

      vi.mocked(mockIsPackInstalled).mockImplementation((id) => id === 'pack2')

      const { installedPacks, notInstalledPacks } = usePacksSelection(nodePacks)

      expect(installedPacks.value).toHaveLength(1)
      expect(installedPacks.value[0].id).toBe('pack2')
      expect(notInstalledPacks.value).toHaveLength(1)
    })

    it('should handle dynamic changes to pack installation status', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2')
      ])

      const installationStatus: Record<string, boolean> = {
        pack1: false,
        pack2: false
      }

      vi.mocked(mockIsPackInstalled).mockImplementation(
        (id) => (id && installationStatus[id]) || false
      )

      const { installedPacks, notInstalledPacks, selectionState } =
        usePacksSelection(nodePacks)

      expect(selectionState.value).toBe('none-installed')
      expect(installedPacks.value).toHaveLength(0)
      expect(notInstalledPacks.value).toHaveLength(2)

      // Simulate installing pack1
      installationStatus.pack1 = true
      nodePacks.value = [...nodePacks.value] // Trigger reactivity

      expect(selectionState.value).toBe('mixed')
      expect(installedPacks.value).toHaveLength(1)
      expect(notInstalledPacks.value).toHaveLength(1)

      // Simulate installing pack2
      installationStatus.pack2 = true
      nodePacks.value = [...nodePacks.value] // Trigger reactivity

      expect(selectionState.value).toBe('all-installed')
      expect(installedPacks.value).toHaveLength(2)
      expect(notInstalledPacks.value).toHaveLength(0)
    })
  })
})
