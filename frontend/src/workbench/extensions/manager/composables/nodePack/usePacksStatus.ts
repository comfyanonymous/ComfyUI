import { computed } from 'vue'
import type { Ref } from 'vue'

import type { components } from '@/types/comfyRegistryTypes'
import { useConflictDetectionStore } from '@/workbench/extensions/manager/stores/conflictDetectionStore'

type NodePack = components['schemas']['Node']
type NodeStatus = components['schemas']['NodeStatus']
type NodeVersionStatus = components['schemas']['NodeVersionStatus']

const STATUS_PRIORITY = [
  'NodeStatusBanned',
  'NodeVersionStatusBanned',
  'NodeStatusDeleted',
  'NodeVersionStatusDeleted',
  'NodeVersionStatusFlagged',
  'NodeVersionStatusPending',
  'NodeStatusActive',
  'NodeVersionStatusActive'
] as const

/**
 * Composable for managing package status with priority
 * Handles import failures and determines the most important status
 */
export function usePacksStatus(nodePacks: Ref<NodePack[]>) {
  const conflictDetectionStore = useConflictDetectionStore()

  const hasImportFailed = computed(() => {
    return nodePacks.value.some((pack) => {
      if (!pack.id) return false
      const conflicts = conflictDetectionStore.getConflictsForPackageByID(
        pack.id
      )
      return (
        conflicts?.conflicts?.some((c) => c.type === 'import_failed') || false
      )
    })
  })

  const overallStatus = computed<NodeStatus | NodeVersionStatus>(() => {
    // Check for import failed first (highest priority for installed packages)
    if (hasImportFailed.value) {
      // Import failed doesn't have a specific status enum, so we return active
      // but the PackStatusMessage will handle it via hasImportFailed prop
      return 'NodeVersionStatusActive' as NodeVersionStatus
    }

    // Find the highest priority status from all packages
    for (const priorityStatus of STATUS_PRIORITY) {
      if (nodePacks.value.some((pack) => pack.status === priorityStatus)) {
        return priorityStatus as NodeStatus | NodeVersionStatus
      }
    }

    // Default to active if no specific status found
    return 'NodeVersionStatusActive' as NodeVersionStatus
  })

  return {
    hasImportFailed,
    overallStatus
  }
}
