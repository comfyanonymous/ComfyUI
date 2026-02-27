import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

import type { components } from '@/types/comfyRegistryTypes'
import { usePacksStatus } from '@/workbench/extensions/manager/composables/nodePack/usePacksStatus'
import { useConflictDetectionStore } from '@/workbench/extensions/manager/stores/conflictDetectionStore'
import type { ConflictDetectionResult } from '@/workbench/extensions/manager/types/conflictDetectionTypes'

type NodePack = components['schemas']['Node']
type NodeStatus = components['schemas']['NodeStatus']
type NodeVersionStatus = components['schemas']['NodeVersionStatus']

describe('usePacksStatus', () => {
  let conflictDetectionStore: ReturnType<typeof useConflictDetectionStore>

  const createMockPack = (
    id: string,
    status?: NodeStatus | NodeVersionStatus
  ): NodePack => ({
    id,
    name: `Pack ${id}`,
    description: `Description for pack ${id}`,
    category: 'Nodes',
    author: 'Test Author',
    license: 'MIT',
    repository: 'https://github.com/test/pack',
    tags: [],
    status: (status || 'NodeStatusActive') as NodeStatus
  })

  const createMockConflict = (
    packageId: string,
    type: 'import_failed' | 'banned' | 'pending' = 'import_failed'
  ): ConflictDetectionResult => ({
    package_id: packageId,
    package_name: `Pack ${packageId}`,
    has_conflict: true,
    conflicts: [
      {
        type,
        current_value: 'current',
        required_value: 'required'
      }
    ],
    is_compatible: false
  })

  beforeEach(() => {
    vi.clearAllMocks()
    setActivePinia(createTestingPinia({ stubActions: false }))
    conflictDetectionStore = useConflictDetectionStore()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('hasImportFailed', () => {
    it('should return true when at least one pack has import_failed conflict', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2'),
        createMockPack('pack3')
      ])

      // Set up mock conflicts
      conflictDetectionStore.setConflictedPackages([
        createMockConflict('pack2', 'import_failed'),
        createMockConflict('pack3', 'banned')
      ])

      const { hasImportFailed } = usePacksStatus(nodePacks)

      expect(hasImportFailed.value).toBe(true)
    })

    it('should return false when no pack has import_failed conflict', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2')
      ])

      // Set up mock conflicts with no import_failed
      conflictDetectionStore.setConflictedPackages([
        createMockConflict('pack1', 'pending'),
        createMockConflict('pack2', 'banned')
      ])

      const { hasImportFailed } = usePacksStatus(nodePacks)

      expect(hasImportFailed.value).toBe(false)
    })

    it('should return false when no conflicts exist', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2')
      ])

      conflictDetectionStore.setConflictedPackages([])

      const { hasImportFailed } = usePacksStatus(nodePacks)

      expect(hasImportFailed.value).toBe(false)
    })

    it('should handle packs without ids', () => {
      const nodePacks = ref<NodePack[]>([
        { ...createMockPack('pack1'), id: undefined },
        createMockPack('pack2')
      ])

      conflictDetectionStore.setConflictedPackages([
        createMockConflict('pack2', 'import_failed')
      ])

      const { hasImportFailed } = usePacksStatus(nodePacks)

      expect(hasImportFailed.value).toBe(true)
    })

    it('should update when conflicts change', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2')
      ])

      conflictDetectionStore.setConflictedPackages([])

      const { hasImportFailed } = usePacksStatus(nodePacks)
      expect(hasImportFailed.value).toBe(false)

      // Add import_failed conflict
      conflictDetectionStore.setConflictedPackages([
        createMockConflict('pack1', 'import_failed')
      ])

      expect(hasImportFailed.value).toBe(true)
    })
  })

  describe('overallStatus', () => {
    it('should prioritize banned status over all others', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1', 'NodeStatusActive'),
        createMockPack('pack2', 'NodeStatusBanned'),
        createMockPack('pack3', 'NodeVersionStatusDeleted')
      ])

      const { overallStatus } = usePacksStatus(nodePacks)

      expect(overallStatus.value).toBe('NodeStatusBanned')
    })

    it('should prioritize version banned over deleted and active', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1', 'NodeStatusActive'),
        createMockPack('pack2', 'NodeVersionStatusBanned'),
        createMockPack('pack3', 'NodeVersionStatusDeleted')
      ])

      const { overallStatus } = usePacksStatus(nodePacks)

      expect(overallStatus.value).toBe('NodeVersionStatusBanned')
    })

    it('should prioritize deleted status appropriately', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1', 'NodeStatusActive'),
        createMockPack('pack2', 'NodeStatusDeleted'),
        createMockPack('pack3', 'NodeVersionStatusActive')
      ])

      const { overallStatus } = usePacksStatus(nodePacks)

      expect(overallStatus.value).toBe('NodeStatusDeleted')
    })

    it('should prioritize version deleted over flagged and active', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1', 'NodeVersionStatusFlagged'),
        createMockPack('pack2', 'NodeVersionStatusDeleted'),
        createMockPack('pack3', 'NodeVersionStatusActive')
      ])

      const { overallStatus } = usePacksStatus(nodePacks)

      expect(overallStatus.value).toBe('NodeVersionStatusDeleted')
    })

    it('should prioritize flagged status over pending and active', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1', 'NodeVersionStatusPending'),
        createMockPack('pack2', 'NodeVersionStatusFlagged'),
        createMockPack('pack3', 'NodeVersionStatusActive')
      ])

      const { overallStatus } = usePacksStatus(nodePacks)

      expect(overallStatus.value).toBe('NodeVersionStatusFlagged')
    })

    it('should prioritize pending status over active', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1', 'NodeVersionStatusActive'),
        createMockPack('pack2', 'NodeVersionStatusPending'),
        createMockPack('pack3', 'NodeStatusActive')
      ])

      const { overallStatus } = usePacksStatus(nodePacks)

      expect(overallStatus.value).toBe('NodeVersionStatusPending')
    })

    it('should return NodeStatusActive when all packs are active', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1', 'NodeStatusActive'),
        createMockPack('pack2', 'NodeStatusActive')
      ])

      const { overallStatus } = usePacksStatus(nodePacks)

      expect(overallStatus.value).toBe('NodeStatusActive')
    })

    it('should return NodeStatusActive as default when all packs have no status', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2')
      ])

      const { overallStatus } = usePacksStatus(nodePacks)

      // Since createMockPack sets status to 'NodeStatusActive' by default
      expect(overallStatus.value).toBe('NodeStatusActive')
    })

    it('should handle empty pack array', () => {
      const nodePacks = ref<NodePack[]>([])

      const { overallStatus } = usePacksStatus(nodePacks)

      expect(overallStatus.value).toBe('NodeVersionStatusActive')
    })

    it('should update when pack statuses change', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1', 'NodeStatusActive'),
        createMockPack('pack2', 'NodeStatusActive')
      ])

      const { overallStatus } = usePacksStatus(nodePacks)
      expect(overallStatus.value).toBe('NodeStatusActive')

      // Change one pack to banned
      nodePacks.value = [
        createMockPack('pack1', 'NodeStatusBanned'),
        createMockPack('pack2', 'NodeStatusActive')
      ]

      expect(overallStatus.value).toBe('NodeStatusBanned')
    })
  })

  describe('integration with import failures', () => {
    it('should return NodeVersionStatusActive when import failures exist (handled separately)', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1', 'NodeStatusActive'),
        createMockPack('pack2', 'NodeStatusActive')
      ])

      conflictDetectionStore.setConflictedPackages([
        createMockConflict('pack1', 'import_failed')
      ])

      const { hasImportFailed, overallStatus } = usePacksStatus(nodePacks)

      expect(hasImportFailed.value).toBe(true)
      // When import failed exists, it returns NodeVersionStatusActive
      expect(overallStatus.value).toBe('NodeVersionStatusActive')
    })

    it('should return NodeVersionStatusActive when import failures exist even with banned status', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1', 'NodeStatusBanned'),
        createMockPack('pack2', 'NodeStatusActive')
      ])

      conflictDetectionStore.setConflictedPackages([
        createMockConflict('pack2', 'import_failed')
      ])

      const { hasImportFailed, overallStatus } = usePacksStatus(nodePacks)

      expect(hasImportFailed.value).toBe(true)
      // Import failed takes priority and returns NodeVersionStatusActive
      expect(overallStatus.value).toBe('NodeVersionStatusActive')
    })
  })

  describe('edge cases', () => {
    it('should handle multiple conflicts per package', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2')
      ])

      conflictDetectionStore.setConflictedPackages([
        {
          package_id: 'pack1',
          package_name: 'Pack pack1',
          has_conflict: true,
          conflicts: [
            {
              type: 'pending',
              current_value: 'current1',
              required_value: 'required1'
            },
            {
              type: 'import_failed',
              current_value: 'current2',
              required_value: 'required2'
            }
          ],
          is_compatible: false
        }
      ])

      const { hasImportFailed } = usePacksStatus(nodePacks)

      expect(hasImportFailed.value).toBe(true)
    })

    it('should handle packs with no conflicts in store', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1'),
        createMockPack('pack2')
      ])

      const { hasImportFailed } = usePacksStatus(nodePacks)

      expect(hasImportFailed.value).toBe(false)
    })

    it('should handle mixed status types correctly', () => {
      const nodePacks = ref<NodePack[]>([
        createMockPack('pack1', 'NodeStatusBanned'),
        createMockPack('pack2', 'NodeVersionStatusBanned'),
        createMockPack('pack3', 'NodeStatusDeleted'),
        createMockPack('pack4', 'NodeVersionStatusDeleted'),
        createMockPack('pack5', 'NodeVersionStatusFlagged'),
        createMockPack('pack6', 'NodeVersionStatusPending'),
        createMockPack('pack7', 'NodeStatusActive'),
        createMockPack('pack8', 'NodeVersionStatusActive')
      ])

      const { overallStatus } = usePacksStatus(nodePacks)

      // Should return the highest priority status (NodeStatusBanned)
      expect(overallStatus.value).toBe('NodeStatusBanned')
    })

    it('should be reactive to nodePacks changes', () => {
      const nodePacks = ref<NodePack[]>([])

      const { overallStatus } = usePacksStatus(nodePacks)
      expect(overallStatus.value).toBe('NodeVersionStatusActive')

      // Add packs
      nodePacks.value = [
        createMockPack('pack1', 'NodeStatusDeleted'),
        createMockPack('pack2', 'NodeStatusActive')
      ]

      expect(overallStatus.value).toBe('NodeStatusDeleted')

      // Add a higher priority status
      nodePacks.value.push(createMockPack('pack3', 'NodeStatusBanned'))

      expect(overallStatus.value).toBe('NodeStatusBanned')
    })
  })
})
