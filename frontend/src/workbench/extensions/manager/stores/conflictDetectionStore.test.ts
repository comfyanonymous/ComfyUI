import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useConflictDetectionStore } from '@/workbench/extensions/manager/stores/conflictDetectionStore'
import type { ConflictDetectionResult } from '@/workbench/extensions/manager/types/conflictDetectionTypes'

describe('useConflictDetectionStore', () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
  })

  const mockConflictedPackages: ConflictDetectionResult[] = [
    {
      package_id: 'ComfyUI-Manager',
      package_name: 'ComfyUI-Manager',
      has_conflict: true,
      is_compatible: false,
      conflicts: [
        {
          type: 'pending',
          current_value: 'no_registry_data',
          required_value: 'registry_data_available'
        }
      ]
    },
    {
      package_id: 'comfyui-easy-use',
      package_name: 'comfyui-easy-use',
      has_conflict: true,
      is_compatible: false,
      conflicts: [
        {
          type: 'comfyui_version',
          current_value: '0.3.43',
          required_value: '<0.3.40'
        }
      ]
    },
    {
      package_id: 'img2colors-comfyui-node',
      package_name: 'img2colors-comfyui-node',
      has_conflict: true,
      is_compatible: false,
      conflicts: [
        {
          type: 'banned',
          current_value: 'installed',
          required_value: 'not_banned'
        }
      ]
    }
  ]

  describe('initial state', () => {
    it('should have empty initial state', () => {
      const store = useConflictDetectionStore()

      expect(store.conflictedPackages).toEqual([])
      expect(store.isDetecting).toBe(false)
      expect(store.lastDetectionTime).toBeNull()
      expect(store.hasConflicts).toBe(false)
    })
  })

  describe('setConflictedPackages', () => {
    it('should set conflicted packages', () => {
      const store = useConflictDetectionStore()

      store.setConflictedPackages(mockConflictedPackages)

      expect(store.conflictedPackages).toEqual(mockConflictedPackages)
      expect(store.conflictedPackages).toHaveLength(3)
    })

    it('should update hasConflicts computed property', () => {
      const store = useConflictDetectionStore()

      expect(store.hasConflicts).toBe(false)

      store.setConflictedPackages(mockConflictedPackages)

      expect(store.hasConflicts).toBe(true)
    })
  })

  describe('getConflictsForPackageByID', () => {
    it('should find package by exact ID match', () => {
      const store = useConflictDetectionStore()
      store.setConflictedPackages(mockConflictedPackages)

      const result = store.getConflictsForPackageByID('ComfyUI-Manager')

      expect(result).toBeDefined()
      expect(result?.package_id).toBe('ComfyUI-Manager')
      expect(result?.conflicts).toHaveLength(1)
    })

    it('should return undefined for non-existent package', () => {
      const store = useConflictDetectionStore()
      store.setConflictedPackages(mockConflictedPackages)

      const result = store.getConflictsForPackageByID('non-existent-package')

      expect(result).toBeUndefined()
    })
  })

  describe('bannedPackages', () => {
    it('should filter packages with banned conflicts', () => {
      const store = useConflictDetectionStore()
      store.setConflictedPackages(mockConflictedPackages)

      const bannedPackages = store.bannedPackages

      expect(bannedPackages).toHaveLength(1)
      expect(bannedPackages[0].package_id).toBe('img2colors-comfyui-node')
    })

    it('should return empty array when no banned packages', () => {
      const store = useConflictDetectionStore()
      const noBannedPackages = mockConflictedPackages.filter(
        (pkg) => !pkg.conflicts.some((c) => c.type === 'banned')
      )
      store.setConflictedPackages(noBannedPackages)

      const bannedPackages = store.bannedPackages

      expect(bannedPackages).toHaveLength(0)
    })
  })

  describe('securityPendingPackages', () => {
    it('should filter packages with pending conflicts', () => {
      const store = useConflictDetectionStore()
      store.setConflictedPackages(mockConflictedPackages)

      const securityPendingPackages = store.securityPendingPackages

      expect(securityPendingPackages).toHaveLength(1)
      expect(securityPendingPackages[0].package_id).toBe('ComfyUI-Manager')
    })
  })

  describe('clearConflicts', () => {
    it('should clear all conflicted packages', () => {
      const store = useConflictDetectionStore()
      store.setConflictedPackages(mockConflictedPackages)

      expect(store.conflictedPackages).toHaveLength(3)
      expect(store.hasConflicts).toBe(true)

      store.clearConflicts()

      expect(store.conflictedPackages).toEqual([])
      expect(store.hasConflicts).toBe(false)
    })
  })

  describe('detection state management', () => {
    it('should set detecting state', () => {
      const store = useConflictDetectionStore()

      expect(store.isDetecting).toBe(false)

      store.setDetecting(true)

      expect(store.isDetecting).toBe(true)

      store.setDetecting(false)

      expect(store.isDetecting).toBe(false)
    })

    it('should set last detection time', () => {
      const store = useConflictDetectionStore()
      const timestamp = '2024-01-01T00:00:00Z'

      expect(store.lastDetectionTime).toBeNull()

      store.setLastDetectionTime(timestamp)

      expect(store.lastDetectionTime).toBe(timestamp)
    })
  })

  describe('reactivity', () => {
    it('should update computed properties when conflicted packages change', () => {
      const store = useConflictDetectionStore()

      // Initially no conflicts
      expect(store.hasConflicts).toBe(false)
      expect(store.bannedPackages).toHaveLength(0)

      // Add conflicts
      store.setConflictedPackages(mockConflictedPackages)

      // Computed properties should update
      expect(store.hasConflicts).toBe(true)
      expect(store.bannedPackages).toHaveLength(1)
      expect(store.securityPendingPackages).toHaveLength(1)

      // Clear conflicts
      store.clearConflicts()

      // Computed properties should update again
      expect(store.hasConflicts).toBe(false)
      expect(store.bannedPackages).toHaveLength(0)
      expect(store.securityPendingPackages).toHaveLength(0)
    })
  })

  describe('edge cases', () => {
    it('should handle empty conflicts array', () => {
      const store = useConflictDetectionStore()
      store.setConflictedPackages([])

      expect(store.conflictedPackages).toEqual([])
      expect(store.hasConflicts).toBe(false)
      expect(store.bannedPackages).toHaveLength(0)
      expect(store.securityPendingPackages).toHaveLength(0)
    })

    it('should handle packages with multiple conflict types', () => {
      const store = useConflictDetectionStore()
      const multiConflictPackage: ConflictDetectionResult = {
        package_id: 'multi-conflict-package',
        package_name: 'Multi Conflict Package',
        has_conflict: true,
        is_compatible: false,
        conflicts: [
          {
            type: 'banned',
            current_value: 'installed',
            required_value: 'not_banned'
          },
          {
            type: 'pending',
            current_value: 'no_registry_data',
            required_value: 'registry_data_available'
          }
        ]
      }

      store.setConflictedPackages([multiConflictPackage])

      // Should appear in both banned and security pending
      expect(store.bannedPackages).toHaveLength(1)
      expect(store.securityPendingPackages).toHaveLength(1)
      expect(store.bannedPackages[0].package_id).toBe('multi-conflict-package')
      expect(store.securityPendingPackages[0].package_id).toBe(
        'multi-conflict-package'
      )
    })

    it('should handle packages with has_conflict false', () => {
      const store = useConflictDetectionStore()
      const noConflictPackage: ConflictDetectionResult = {
        package_id: 'no-conflict-package',
        package_name: 'No Conflict Package',
        has_conflict: false,
        is_compatible: true,
        conflicts: []
      }

      store.setConflictedPackages([noConflictPackage])

      // hasConflicts should check has_conflict property
      expect(store.hasConflicts).toBe(false)
    })
  })
})
