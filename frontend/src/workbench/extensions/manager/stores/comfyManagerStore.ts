import { useEventListener, whenever } from '@vueuse/core'
import { defineStore } from 'pinia'
import { v4 as uuidv4 } from 'uuid'
import { ref, watch } from 'vue'

import { t } from '@/i18n'
import { useCachedRequest } from '@/composables/useCachedRequest'
import { useServerLogs } from '@/composables/useServerLogs'
import { api } from '@/scripts/api'
import { app } from '@/scripts/app'

import { normalizePackKeys } from '@/utils/packUtils'
import { useManagerQueue } from '@/workbench/extensions/manager/composables/useManagerQueue'
import { useComfyManagerService } from '@/workbench/extensions/manager/services/comfyManagerService'
import type { TaskLog } from '@/workbench/extensions/manager/types/comfyManagerTypes'
import type { components } from '@/workbench/extensions/manager/types/generatedManagerTypes'

type InstallPackParams = components['schemas']['InstallPackParams']
type InstalledPacksResponse = components['schemas']['InstalledPacksResponse']
type ManagerPackInfo = components['schemas']['ManagerPackInfo']
type ManagerPackInstalled = components['schemas']['ManagerPackInstalled']
type ManagerTaskHistory = Record<
  string,
  components['schemas']['TaskHistoryItem']
>
type ManagerTaskQueue = components['schemas']['TaskStateMessage']
type UpdateAllPacksParams = components['schemas']['UpdateAllPacksParams']

/**
 * Store for state of installed node packs
 */
export const useComfyManagerStore = defineStore('comfyManager', () => {
  const managerService = useComfyManagerService()

  const installedPacks = ref<InstalledPacksResponse>({})
  const enabledPacksIds = ref<Set<string>>(new Set())
  const disabledPacksIds = ref<Set<string>>(new Set())
  const installedPacksIds = ref<Set<string>>(new Set())
  const installingPacksIds = ref<Set<string>>(new Set())
  const isStale = ref(true)
  const taskLogs = ref<TaskLog[]>([])
  const succeededTasksLogs = ref<TaskLog[]>([])
  const failedTasksLogs = ref<TaskLog[]>([])

  const taskHistory = ref<ManagerTaskHistory>({})
  const succeededTasksIds = ref<string[]>([])
  const failedTasksIds = ref<string[]>([])
  const taskQueue = ref<ManagerTaskQueue>({
    history: {},
    running_queue: [],
    pending_queue: [],
    installed_packs: {}
  })

  // Track task ID to pack ID mapping for proper state cleanup
  const taskIdToPackId = ref(new Map<string, string>())

  const managerQueue = useManagerQueue(taskHistory, taskQueue, installedPacks)

  // Listen for task completion events to clean up installing state
  useEventListener(
    app.api,
    'cm-task-completed',
    (event: CustomEvent<{ ui_id?: string }>) => {
      const taskId = event.detail?.ui_id
      if (taskId && taskIdToPackId.value.has(taskId)) {
        const packId = taskIdToPackId.value.get(taskId)!
        installingPacksIds.value.delete(packId)
        taskIdToPackId.value.delete(taskId)
      }
    }
  )

  const setStale = () => {
    isStale.value = true
  }

  const partitionTaskLogs = () => {
    const successTaskLogs: TaskLog[] = []
    const failTaskLogs: TaskLog[] = []
    for (const log of taskLogs.value) {
      if (failedTasksIds.value.includes(log.taskId)) {
        failTaskLogs.push(log)
      } else {
        successTaskLogs.push(log)
      }
    }
    succeededTasksLogs.value = successTaskLogs
    failedTasksLogs.value = failTaskLogs
  }

  const partitionTasks = () => {
    const successTasksIds = []
    const failTasksIds = []
    for (const task of Object.values(taskHistory.value)) {
      if (task.status?.status_str === 'success') {
        successTasksIds.push(task.ui_id)
      } else {
        failTasksIds.push(task.ui_id)
      }
    }
    succeededTasksIds.value = successTasksIds
    failedTasksIds.value = failTasksIds
  }

  whenever(
    taskHistory,
    () => {
      partitionTasks()
      partitionTaskLogs()
    },
    { deep: true }
  )

  const getPackId = (pack: ManagerPackInstalled) => pack.cnr_id || pack.aux_id

  const isInstalledPackId = (packName: string | undefined): boolean =>
    !!packName && installedPacksIds.value.has(packName)

  const isEnabledPackId = (packName: string | undefined): boolean =>
    !!packName &&
    isInstalledPackId(packName) &&
    enabledPacksIds.value.has(packName)

  const isInstallingPackId = (packName: string | undefined): boolean =>
    !!packName && installingPacksIds.value.has(packName)

  const packsToIdSet = (packs: ManagerPackInstalled[]) =>
    packs.reduce((acc, pack) => {
      const id = pack.cnr_id || pack.aux_id
      if (id) acc.add(id)
      return acc
    }, new Set<string>())

  /**
   * A pack is disabled if there is a disabled entry and no corresponding
   * enabled entry. If `packname@1.0.2` is disabled, but `packname@1.0.3` is
   * enabled, then `packname` is considered enabled.
   *
   * @example
   * installedPacks = {
   *   "packname@1_0_2": { enabled: false, cnr_id: "packname" },
   *   "packname": { enabled: true, cnr_id: "packname" }
   * }
   * isDisabled("packname") // false
   *
   * installedPacks = {
   *   "packname@1_0_2": { enabled: false, cnr_id: "packname" },
   * }
   * isDisabled("packname") // true
   */
  const updateDisabledIds = (packs: ManagerPackInstalled[]) => {
    // Use temporary variables to avoid triggering reactivity
    const enabledIds = new Set<string>()
    const disabledIds = new Set<string>()

    for (const pack of packs) {
      const id = getPackId(pack)
      if (!id) continue

      const { enabled } = pack

      if (enabled === true) enabledIds.add(id)
      else if (enabled === false) disabledIds.add(id)

      // If pack in both (has a disabled and enabled version), remove from disabled
      const inBothSets = enabledIds.has(id) && disabledIds.has(id)
      if (inBothSets) disabledIds.delete(id)
    }

    enabledPacksIds.value = enabledIds
    disabledPacksIds.value = disabledIds
  }

  const updateInstalledIds = (packs: ManagerPackInstalled[]) => {
    installedPacksIds.value = packsToIdSet(packs)
  }

  const onPacksChanged = () => {
    const packs = Object.values(installedPacks.value)
    updateDisabledIds(packs)
    updateInstalledIds(packs)
  }

  watch(installedPacks, onPacksChanged, { deep: true })

  const refreshInstalledList = async () => {
    const packs = await managerService.listInstalledPacks()
    if (packs) {
      // Normalize pack keys to ensure consistent access
      installedPacks.value = normalizePackKeys(packs)
    }
    isStale.value = false
  }

  whenever(isStale, refreshInstalledList, { immediate: true })

  const enqueueTaskWithLogs = async (
    task: (taskId: string) => Promise<null>,
    taskName: string
  ) => {
    const taskId = uuidv4()
    const { logs } = useServerLogs({
      ui_id: taskId,
      immediate: true
    })

    try {
      managerQueue.isProcessing.value = true

      // Prepare logging hook
      taskLogs.value.push({ taskName, taskId, logs: logs.value })

      // Queue the task to the server
      await task(taskId)
    } catch (error) {
      // Reset processing state on error
      managerQueue.isProcessing.value = false

      // The server has authority over task history in general, but in rare
      // case of client-side error, we add that to failed tasks from the client side
      taskHistory.value[taskId] = {
        ui_id: taskId,
        client_id: api.clientId || 'unknown',
        kind: 'error',
        result: 'failed',
        status: {
          status_str: 'error',
          completed: false,
          messages: [error instanceof Error ? error.message : String(error)]
        },
        timestamp: new Date().toISOString()
      }
    }
  }

  const installPack = useCachedRequest<InstallPackParams, void>(
    async (params: InstallPackParams, signal?: AbortSignal) => {
      if (!params.id) return

      let actionDescription = t('g.installing')
      if (installedPacksIds.value.has(params.id)) {
        const installedPack = installedPacks.value[params.id]

        if (installedPack && installedPack.ver !== params.selected_version) {
          actionDescription = t('manager.changingVersion', {
            from: installedPack.ver,
            to: params.selected_version
          })
        } else {
          actionDescription = t('g.enabling')
        }
      }

      installingPacksIds.value.add(params.id)
      const task = (taskId: string) => {
        taskIdToPackId.value.set(taskId, params.id)
        return managerService.installPack(params, taskId, signal)
      }
      await enqueueTaskWithLogs(task, `${actionDescription} ${params.id}`)
    },
    { maxSize: 1 }
  )

  const uninstallPack = async (
    params: ManagerPackInfo,
    signal?: AbortSignal
  ) => {
    installPack.clear()
    installPack.cancel()

    installingPacksIds.value.add(params.id)
    const uninstallParams: components['schemas']['UninstallPackParams'] = {
      node_name: params.id,
      is_unknown: false
    }
    const task = (taskId: string) => {
      taskIdToPackId.value.set(taskId, params.id)
      return managerService.uninstallPack(uninstallParams, taskId, signal)
    }
    await enqueueTaskWithLogs(
      task,
      t('manager.uninstalling', { id: params.id })
    )
  }

  const updatePack = useCachedRequest<ManagerPackInfo, void>(
    async (params: ManagerPackInfo, signal?: AbortSignal) => {
      updateAllPacks.cancel()
      const updateParams: components['schemas']['UpdatePackParams'] = {
        node_name: params.id,
        node_ver: params.version
      }
      const task = (taskId: string) =>
        managerService.updatePack(updateParams, taskId, signal)
      await enqueueTaskWithLogs(task, t('g.updating', { id: params.id }))
    },
    { maxSize: 1 }
  )

  const updateAllPacks = useCachedRequest<UpdateAllPacksParams, void>(
    async (params: UpdateAllPacksParams, signal?: AbortSignal) => {
      const task = (taskId: string) =>
        managerService.updateAllPacks(params, taskId, signal)
      await enqueueTaskWithLogs(task, t('manager.updatingAllPacks'))
    },
    { maxSize: 1 }
  )

  const disablePack = async (params: ManagerPackInfo, signal?: AbortSignal) => {
    const disableParams: components['schemas']['DisablePackParams'] = {
      node_name: params.id,
      is_unknown: false
    }
    const task = (taskId: string) =>
      managerService.disablePack(disableParams, taskId, signal)
    await enqueueTaskWithLogs(task, t('g.disabling', { id: params.id }))
  }

  const enablePack = async (params: ManagerPackInfo, signal?: AbortSignal) => {
    const enableParams: components['schemas']['EnablePackParams'] = {
      cnr_id: params.id
    }
    const task = (taskId: string) =>
      managerService.enablePack(enableParams, taskId, signal)
    await enqueueTaskWithLogs(task, t('g.enabling', { id: params.id }))
  }

  const getInstalledPackVersion = (packId: string) => {
    const pack = installedPacks.value[packId]
    return pack?.ver
  }

  const clearLogs = () => {
    taskLogs.value = []
  }

  const resetTaskState = () => {
    // Clear all task-related reactive state for fresh start after restart
    taskLogs.value = []
    taskHistory.value = {}
    succeededTasksIds.value = []
    failedTasksIds.value = []
    succeededTasksLogs.value = []
    failedTasksLogs.value = []
    installingPacksIds.value.clear()
    taskIdToPackId.value.clear()

    // Reset task queue to initial state
    taskQueue.value = {
      history: {},
      running_queue: [],
      pending_queue: [],
      installed_packs: {}
    }
  }

  return {
    // Manager state
    isLoading: managerService.isLoading,
    error: managerService.error,
    taskLogs,
    clearLogs,
    resetTaskState,
    setStale,

    // Installed packs state
    installedPacks,
    installedPacksIds,
    isPackInstalled: isInstalledPackId,
    isPackEnabled: isEnabledPackId,
    isPackInstalling: isInstallingPackId,
    getInstalledPackVersion,
    refreshInstalledList,

    // Task queue state and actions
    taskHistory,
    taskQueue,
    isProcessingTasks: managerQueue.isProcessing,
    succeededTasksIds,
    failedTasksIds,
    succeededTasksLogs,
    failedTasksLogs,
    managerQueue,

    // Pack actions
    installPack,
    uninstallPack,
    updatePack,
    updateAllPacks,
    disablePack,
    enablePack
  }
})
