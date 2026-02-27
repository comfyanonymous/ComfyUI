import { computed, unref } from 'vue'
import type { ComputedRef } from 'vue'

import { useImportFailedNodeDialog } from '@/workbench/extensions/manager/composables/useImportFailedNodeDialog'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'
import { useConflictDetectionStore } from '@/workbench/extensions/manager/stores/conflictDetectionStore'
import type {
  ConflictDetail,
  ConflictDetectionResult
} from '@/workbench/extensions/manager/types/conflictDetectionTypes'

/**
 * Extracting import failed conflicts from conflict list
 */
function extractImportFailedConflicts(conflicts?: ConflictDetail[] | null) {
  if (!conflicts) return null

  const importFailedConflicts = conflicts.filter(
    (item): item is ConflictDetail => item.type === 'import_failed'
  )

  return importFailedConflicts.length > 0 ? importFailedConflicts : null
}

/**
 * Creating import failed dialog
 */
function createImportFailedDialog() {
  const { show } = useImportFailedNodeDialog()

  return (
    conflictedPackages: ConflictDetectionResult[] | null,
    onClose?: () => void
  ) => {
    if (conflictedPackages && conflictedPackages.length > 0) {
      void show({
        conflictedPackages,
        dialogComponentProps: {
          onClose
        }
      })
    }
  }
}

/**
 * Composable for detecting and handling import failed conflicts
 * @param packageId - Package ID string or computed ref
 * @returns Object with import failed detection and dialog handler
 */
export function useImportFailedDetection(
  packageId?: string | ComputedRef<string> | null
) {
  const { isPackInstalled } = useComfyManagerStore()
  const { getConflictsForPackageByID } = useConflictDetectionStore()

  const isInstalled = computed(() =>
    packageId ? isPackInstalled(unref(packageId)) : false
  )

  const conflicts = computed(() => {
    const currentPackageId = unref(packageId)
    if (!currentPackageId || !isInstalled.value) return null
    return getConflictsForPackageByID(currentPackageId) || null
  })

  const importFailedInfo = computed(() => {
    return extractImportFailedConflicts(conflicts.value?.conflicts)
  })

  const importFailed = computed(() => {
    return importFailedInfo.value !== null
  })

  const openDialog = createImportFailedDialog()

  return {
    importFailedInfo,
    importFailed,
    showImportFailedDialog: (onClose?: () => void) => {
      if (conflicts.value) {
        openDialog([conflicts.value], onClose)
      }
    },
    isInstalled
  }
}
