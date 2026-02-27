import { until } from '@vueuse/core'
import { find } from 'es-toolkit/compat'
import { computed, getCurrentInstance, onUnmounted, readonly, ref } from 'vue'

import { useComfyRegistryService } from '@/services/comfyRegistryService'
import { useSystemStatsStore } from '@/stores/systemStatsStore'
import type { components } from '@/types/comfyRegistryTypes'
import { useInstalledPacks } from '@/workbench/extensions/manager/composables/nodePack/useInstalledPacks'
import { useConflictAcknowledgment } from '@/workbench/extensions/manager/composables/useConflictAcknowledgment'
import { useManagerState } from '@/workbench/extensions/manager/composables/useManagerState'
import { useComfyManagerService } from '@/workbench/extensions/manager/services/comfyManagerService'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'
import { useConflictDetectionStore } from '@/workbench/extensions/manager/stores/conflictDetectionStore'
import type {
  RegistryAccelerator,
  RegistryOS
} from '@/workbench/extensions/manager/types/compatibility.types'
import type {
  ConflictDetail,
  ConflictDetectionResponse,
  ConflictDetectionResult,
  ImportFailureMap,
  Node,
  NodeRequirements,
  SystemEnvironment
} from '@/workbench/extensions/manager/types/conflictDetectionTypes'
import {
  consolidateConflictsByPackage,
  createBannedConflict,
  createPendingConflict
} from '@/workbench/extensions/manager/utils/conflictUtils'
import {
  checkAcceleratorCompatibility,
  checkOSCompatibility,
  normalizeOSList
} from '@/workbench/extensions/manager/utils/systemCompatibility'
import {
  checkVersionCompatibility,
  getFrontendVersion
} from '@/workbench/extensions/manager/utils/versionUtil'

/**
 * Composable for conflict detection system.
 * Error-resilient and asynchronous to avoid affecting other components.
 */
export function useConflictDetection() {
  const managerStore = useComfyManagerStore()

  const {
    startFetchInstalled,
    installedPacks,
    installedPacksWithVersions,
    isReady: installedPacksReady
  } = useInstalledPacks()

  const isDetecting = ref(false)
  const lastDetectionTime = ref<string | null>(null)
  const detectionError = ref<string | null>(null)

  const systemEnvironment = ref<SystemEnvironment | null>(null)

  const detectionResults = ref<ConflictDetectionResult[]>([])
  // Store merged conflicts separately for testing
  const storedMergedConflicts = ref<ConflictDetectionResult[]>([])

  // Registry API request cancellation
  const abortController = ref<AbortController | null>(null)

  const acknowledgment = useConflictAcknowledgment()

  const conflictStore = useConflictDetectionStore()

  const hasConflicts = computed(() => conflictStore.hasConflicts)
  const conflictedPackages = computed(() => {
    return conflictStore.conflictedPackages
  })

  const bannedPackages = computed(() => conflictStore.bannedPackages)
  const securityPendingPackages = computed(
    () => conflictStore.securityPendingPackages
  )

  /**
   * Collects current system environment information.
   * Continues with default values even if errors occur.
   * @returns Promise that resolves to system environment information
   */
  async function collectSystemEnvironment(): Promise<SystemEnvironment> {
    try {
      // Get system stats from store (primary source of system information)
      // Wait for systemStats to be initialized if not already
      const systemStatsStore = useSystemStatsStore()
      const { systemStats } = systemStatsStore

      // Wait for initialization using the store's isInitialized property (correct reactive way)
      await until(() => systemStatsStore.isInitialized).toBe(true)

      const frontendVersion = getFrontendVersion()

      const environment: SystemEnvironment = {
        comfyui_version: systemStats?.system.comfyui_version ?? '',
        frontend_version: frontendVersion,
        os: systemStats?.system.os ?? '',
        accelerator: systemStats?.devices?.[0]?.type ?? ''
      }

      systemEnvironment.value = environment
      return environment
    } catch (error) {
      const fallbackEnvironment: SystemEnvironment = {
        comfyui_version: undefined,
        frontend_version: undefined,
        os: undefined,
        accelerator: undefined
      }
      systemEnvironment.value = fallbackEnvironment
      return fallbackEnvironment
    }
  }

  /**
   * Fetches requirement information for installed packages using Registry Store.
   *
   * This function combines local installation data with Registry API compatibility metadata
   * using the established store layer pattern with caching and batch requests.
   *
   * Process
   * 1. Get locally installed packages
   * 2. Batch fetch Registry data using store layer
   * 3. Combine local + Registry data
   * 4. Extract compatibility requirements
   *
   * @returns Promise that resolves to array of node pack requirements
   */
  async function buildNodeRequirements(): Promise<NodeRequirements[]> {
    try {
      // Step 1: Use installed packs composable instead of direct API calls
      await startFetchInstalled() // Ensure data is loaded

      if (
        !installedPacksReady.value ||
        !installedPacks.value ||
        installedPacks.value.length === 0
      ) {
        console.warn(
          '[ConflictDetection] No installed packages available from useInstalledPacks'
        )
        return []
      }

      // Step 2: Get Registry service for bulk API calls
      const registryService = useComfyRegistryService()

      // Step 3: Setup abort controller for request cancellation
      abortController.value = new AbortController()

      // Step 4: Use bulk API to fetch all version data in a single request
      const versionDataMap = new Map<
        string,
        components['schemas']['NodeVersion']
      >()

      // Prepare bulk request with actual installed versions from Manager API
      const nodeVersions = installedPacksWithVersions.value.map((pack) => ({
        node_id: pack.id,
        version: pack.version
      }))

      if (nodeVersions.length > 0) {
        try {
          const bulkResponse = await registryService.getBulkNodeVersions(
            nodeVersions,
            abortController.value?.signal
          )

          if (bulkResponse && bulkResponse.node_versions?.length > 0) {
            // Process bulk response
            bulkResponse.node_versions.forEach((result) => {
              if (result.status === 'success' && result.node_version) {
                versionDataMap.set(
                  result.identifier.node_id,
                  result.node_version
                )
              } else if (result.status === 'error') {
                console.warn(
                  `[ConflictDetection] Failed to fetch version data for ${result.identifier.node_id}@${result.identifier.version}:`,
                  result.error_message
                )
              }
            })
          }
        } catch (error) {
          console.warn(
            '[ConflictDetection] Failed to fetch bulk version data:',
            error
          )
        }
      }

      // Step 5: Combine local installation data with Registry version data
      const requirements: NodeRequirements[] = []

      // IMPORTANT: Use installedPacksWithVersions to check ALL installed packages
      // not just the ones that exist in Registry (installedPacks)
      for (const installedPackVersion of installedPacksWithVersions.value) {
        const versionData = versionDataMap.get(installedPackVersion.id)
        const isEnabled = managerStore.isPackEnabled(installedPackVersion.id)

        // Find the pack info from Registry if available
        const packInfo = find(installedPacks.value, {
          id: installedPackVersion.id
        })

        if (versionData) {
          // Combine local installation data with version-specific Registry data
          const requirement: NodeRequirements = {
            // Basic package info
            id: installedPackVersion.id,
            name: packInfo?.name || installedPackVersion.id,
            installed_version: installedPackVersion.version,
            is_enabled: isEnabled,

            // Version-specific compatibility data
            supported_comfyui_version: versionData.supported_comfyui_version,
            supported_comfyui_frontend_version:
              versionData.supported_comfyui_frontend_version,
            supported_os: normalizeOSList(
              versionData.supported_os
            ) as Node['supported_os'],
            supported_accelerators: versionData.supported_accelerators,

            // Status information
            version_status: versionData.status,
            is_banned: versionData.status === 'NodeVersionStatusBanned',
            is_pending: versionData.status === 'NodeVersionStatusPending'
          }

          requirements.push(requirement)
        } else {
          console.warn(
            `[ConflictDetection] No Registry data found for ${installedPackVersion.id}, using fallback`
          )

          // Create fallback requirement without Registry data
          const fallbackRequirement: NodeRequirements = {
            id: installedPackVersion.id,
            name: packInfo?.name || installedPackVersion.id,
            installed_version: installedPackVersion.version,
            is_enabled: isEnabled,
            is_banned: false,
            is_pending: false
          }

          requirements.push(fallbackRequirement)
        }
      }

      return requirements
    } catch (error) {
      console.warn(
        '[ConflictDetection] Failed to fetch package requirements:',
        error
      )
      return []
    }
  }

  /**
   * Detects conflicts for an individual package using Registry API data.
   *
   * @param packageReq Package requirements from Registry
   * @param sysEnv Current system environment
   * @returns Conflict detection result for the package
   */
  function analyzePackageConflicts(
    packageReq: NodeRequirements,
    systemEnvInfo: SystemEnvironment
  ): ConflictDetectionResult {
    const conflicts: ConflictDetail[] = []

    // 1. ComfyUI version conflict check
    const versionConflict = checkVersionCompatibility(
      'comfyui_version',
      systemEnvInfo.comfyui_version,
      packageReq.supported_comfyui_version
    )
    if (versionConflict) conflicts.push(versionConflict)

    // 2. Frontend version conflict check
    const frontendConflict = checkVersionCompatibility(
      'frontend_version',
      systemEnvInfo.frontend_version,
      packageReq.supported_comfyui_frontend_version
    )
    if (frontendConflict) conflicts.push(frontendConflict)

    // 3. OS compatibility check
    const osConflict = checkOSCompatibility(
      packageReq.supported_os as RegistryOS[] | undefined,
      systemEnvInfo.os
    )
    if (osConflict) conflicts.push(osConflict)

    // 4. Accelerator compatibility check
    const acceleratorConflict = checkAcceleratorCompatibility(
      packageReq.supported_accelerators as RegistryAccelerator[] | undefined,
      systemEnvInfo.accelerator
    )
    if (acceleratorConflict) conflicts.push(acceleratorConflict)

    // 5. Banned package check using shared logic
    const bannedConflict = createBannedConflict(packageReq.is_banned)
    if (bannedConflict) {
      conflicts.push(bannedConflict)
    }

    // 6. Registry data availability check using shared logic
    const pendingConflict = createPendingConflict(packageReq.is_pending)
    if (pendingConflict) {
      conflicts.push(pendingConflict)
    }

    // Generate result
    const hasConflict = conflicts.length > 0

    return {
      package_id: packageReq.id ?? '',
      package_name: packageReq.name ?? '',
      has_conflict: hasConflict,
      conflicts,
      is_compatible: !hasConflict
    }
  }

  /**
   * Fetches Python import failure information from ComfyUI Manager.
   * Gets installed packages and checks each one for import failures using bulk API.
   * @returns Promise that resolves to import failure data
   */
  async function fetchImportFailInfo(): Promise<ImportFailureMap> {
    try {
      const comfyManagerService = useComfyManagerService()

      // Use installedPacksWithVersions to match what versions bulk API uses
      // This ensures both APIs check the same set of packages
      if (
        !installedPacksWithVersions.value ||
        installedPacksWithVersions.value.length === 0
      ) {
        console.warn(
          '[ConflictDetection] No installed packages available for import failure check'
        )
        return {}
      }

      const packageIds = installedPacksWithVersions.value.map((pack) => pack.id)

      // Use bulk API to get import failure info for all packages at once
      const bulkResult = await comfyManagerService.getImportFailInfoBulk(
        { cnr_ids: packageIds },
        abortController.value?.signal
      )

      if (bulkResult) {
        // Filter out null values (packages without import failures)
        const importFailures: ImportFailureMap = {}

        Object.entries(bulkResult).forEach(([packageId, failInfo]) => {
          if (failInfo !== null) {
            importFailures[packageId] = failInfo
          }
        })

        return importFailures
      }

      return {}
    } catch (error) {
      console.warn(
        '[ConflictDetection] Failed to fetch import failure information:',
        error
      )
      return {}
    }
  }

  /**
   * Detects runtime conflicts from Python import failures.
   * @param importFailInfo Import failure data from Manager API
   * @returns Array of conflict detection results for failed imports
   */
  function detectImportFailConflicts(
    importFailInfo: ImportFailureMap
  ): ConflictDetectionResult[] {
    const results: ConflictDetectionResult[] = []
    if (!importFailInfo || typeof importFailInfo !== 'object') {
      return results
    }

    // Process import failures
    for (const [packageId, failureInfo] of Object.entries(importFailInfo)) {
      if (!failureInfo || typeof failureInfo !== 'object') continue

      const errorMsg = failureInfo.error || 'Unknown import error'
      const fullErrorInfo = failureInfo.traceback || errorMsg

      results.push({
        package_id: packageId,
        package_name: packageId,
        has_conflict: true,
        conflicts: [
          {
            type: 'import_failed',
            current_value: errorMsg,
            required_value: fullErrorInfo
          }
        ],
        is_compatible: false
      })

      console.warn(
        `[ConflictDetection] Python import failure detected for ${packageId}:`,
        errorMsg
      )
    }

    return results
  }

  /**
   * Performs complete conflict detection.
   * @returns Promise that resolves to conflict detection response
   */
  async function runFullConflictAnalysis(): Promise<ConflictDetectionResponse> {
    if (isDetecting.value) {
      return {
        success: false,
        error_message: 'Already detecting conflicts',
        results: detectionResults.value
      }
    }

    isDetecting.value = true
    detectionError.value = null

    try {
      // 1. Collect system environment information
      const systemEnvInfo = await collectSystemEnvironment()

      // 2. Collect installed node requirement information
      const installedNodeRequirements = await buildNodeRequirements()

      // 3. Detect conflicts for each package (parallel processing)
      const conflictDetectionTasks = installedNodeRequirements.map(
        async (packageReq) => {
          try {
            return analyzePackageConflicts(packageReq, systemEnvInfo)
          } catch (error) {
            console.warn(
              `[ConflictDetection] Failed to detect conflicts for package ${packageReq.name}:`,
              error
            )
            // Return null for failed packages, will be filtered out
            return null
          }
        }
      )

      const conflictResults = await Promise.allSettled(conflictDetectionTasks)
      const packageResults: ConflictDetectionResult[] = conflictResults
        .map((result) => (result.status === 'fulfilled' ? result.value : null))
        .filter((result): result is ConflictDetectionResult => result !== null)

      // 4. Detect Python import failures
      const importFailInfo = await fetchImportFailInfo()
      const importFailResults = detectImportFailConflicts(importFailInfo)

      // 5. Combine all results
      const allResults = [...packageResults, ...importFailResults]

      // 6. Update state
      detectionResults.value = allResults
      lastDetectionTime.value = new Date().toISOString()

      // Store conflict results for later UI display
      // Dialog will be shown based on specific events, not on app mount
      if (allResults.some((result) => result.has_conflict)) {
        const conflictedResults = allResults.filter(
          (result) => result.has_conflict
        )

        // Merge conflicts for packages with the same name
        const mergedConflicts = consolidateConflictsByPackage(conflictedResults)

        // Store merged conflicts in Pinia store for UI usage
        conflictStore.setConflictedPackages(mergedConflicts)

        // Also update local state for backward compatibility
        detectionResults.value = [...mergedConflicts]
        storedMergedConflicts.value = [...mergedConflicts]

        // Use merged conflicts in response as well
        const response: ConflictDetectionResponse = {
          success: true,
          results: mergedConflicts,
          detected_system_environment: systemEnvInfo
        }
        return response
      } else {
        // No conflicts detected, clear the results
        conflictStore.clearConflicts()
        detectionResults.value = []
      }

      const response: ConflictDetectionResponse = {
        success: true,
        results: allResults,
        detected_system_environment: systemEnvInfo
      }

      return response
    } catch (error) {
      console.error(
        '[ConflictDetection] Error during conflict detection:',
        error
      )
      detectionError.value =
        error instanceof Error ? error.message : String(error)

      return {
        success: false,
        error_message: detectionError.value,
        results: []
      }
    } finally {
      isDetecting.value = false
      // Clear abort controller to prevent memory leaks
      if (abortController.value) {
        abortController.value = null
      }
    }
  }

  /**
   * Error-resilient initialization (called on app mount).
   * Async function that doesn't block UI setup.
   * Ensures proper order: system_stats -> manager state -> installed -> versions bulk -> import_fail_info_bulk
   */
  async function initializeConflictDetection(): Promise<void> {
    try {
      // First, wait for systemStats to be initialized
      const systemStatsStore = useSystemStatsStore()
      await until(() => systemStatsStore.isInitialized).toBe(true)

      // Now check if manager is new Manager
      const managerState = useManagerState()

      if (!managerState.isNewManagerUI.value) {
        return
      }

      // Manager is new Manager, perform conflict detection
      // The useInstalledPacks will handle fetching installed list if needed
      await runFullConflictAnalysis()
    } catch (error) {
      console.warn(
        '[ConflictDetection] Error during initialization (ignored):',
        error
      )
      // Errors do not affect other parts of the app
    }
  }

  // Cleanup function for request cancellation
  function cancelRequests(): void {
    if (abortController.value) {
      abortController.value.abort()
      abortController.value = null
    }
  }

  // Auto-cleanup on component unmount
  // Only register lifecycle hooks if we're in a Vue component context
  const instance = getCurrentInstance()
  if (instance) {
    onUnmounted(() => {
      cancelRequests()
    })
  }

  // Helper functions (implementations at the bottom of the file)

  /**
   * Check if conflicts should trigger modal display after "What's New" dismissal
   */
  async function shouldShowConflictModalAfterUpdate(): Promise<boolean> {
    // Ensure conflict detection has run
    if (detectionResults.value.length === 0) {
      await runFullConflictAnalysis()
    }

    // Check if this is a version update scenario
    // In a real scenario, this would check actual version change
    // For now, we'll assume it's an update if we have conflicts and modal hasn't been dismissed
    const hasActualConflicts = hasConflicts.value
    const canShowModal = acknowledgment.shouldShowConflictModal.value

    return hasActualConflicts && canShowModal
  }

  /**
   * Check compatibility for a node.
   * Used by components like PackVersionSelectorPopover.
   */
  function checkNodeCompatibility(
    node: Node | components['schemas']['NodeVersion']
  ) {
    const conflicts: ConflictDetail[] = []

    // Check OS compatibility
    const osConflict = checkOSCompatibility(
      normalizeOSList(node.supported_os),
      systemEnvironment.value?.os
    )
    if (osConflict) {
      conflicts.push(osConflict)
    }

    // Check Accelerator compatibility
    const acceleratorConflict = checkAcceleratorCompatibility(
      node.supported_accelerators as RegistryAccelerator[],
      systemEnvironment.value?.accelerator
    )
    if (acceleratorConflict) {
      conflicts.push(acceleratorConflict)
    }

    // Check ComfyUI version compatibility
    const comfyUIVersionConflict = checkVersionCompatibility(
      'comfyui_version',
      systemEnvironment.value?.comfyui_version,
      node.supported_comfyui_version
    )
    if (comfyUIVersionConflict) {
      conflicts.push(comfyUIVersionConflict)
    }

    // Check ComfyUI Frontend version compatibility
    const currentFrontendVersion = getFrontendVersion()
    const frontendVersionConflict = checkVersionCompatibility(
      'frontend_version',
      currentFrontendVersion,
      node.supported_comfyui_frontend_version
    )
    if (frontendVersionConflict) {
      conflicts.push(frontendVersionConflict)
    }

    // Check banned package status using shared logic
    const bannedConflict = createBannedConflict(
      node.status === 'NodeStatusBanned' ||
        node.status === 'NodeVersionStatusBanned'
    )
    if (bannedConflict) {
      conflicts.push(bannedConflict)
    }

    // Check pending status using shared logic
    const pendingConflict = createPendingConflict(
      node.status === 'NodeVersionStatusPending'
    )
    if (pendingConflict) {
      conflicts.push(pendingConflict)
    }

    return {
      hasConflict: conflicts.length > 0,
      conflicts
    }
  }

  return {
    // State
    isDetecting: readonly(isDetecting),
    lastDetectionTime: readonly(lastDetectionTime),
    detectionError: readonly(detectionError),
    systemEnvironment: readonly(systemEnvironment),
    detectionResults: readonly(detectionResults),

    // Computed
    hasConflicts,
    conflictedPackages,
    bannedPackages,
    securityPendingPackages,

    // Methods
    runFullConflictAnalysis,
    collectSystemEnvironment,
    initializeConflictDetection,
    cancelRequests,
    shouldShowConflictModalAfterUpdate,

    // Helper functions for other components
    checkNodeCompatibility
  }
}
