import type { AxiosError, AxiosResponse } from 'axios'
import axios from 'axios'
import { v4 as uuidv4 } from 'uuid'
import { ref } from 'vue'

import { api } from '@/scripts/api'
import { isAbortError } from '@/utils/typeGuardUtil'
import { useManagerState } from '@/workbench/extensions/manager/composables/useManagerState'
import type { components } from '@/workbench/extensions/manager/types/generatedManagerTypes'

type ManagerQueueStatus = components['schemas']['QueueStatus']
type InstallPackParams = components['schemas']['InstallPackParams']
type InstalledPacksResponse = components['schemas']['InstalledPacksResponse']
type UpdateAllPacksParams = components['schemas']['UpdateAllPacksParams']
type UpdateComfyUIParams = components['schemas']['UpdateComfyUIParams']
type ManagerTaskHistory = components['schemas']['HistoryResponse']
type QueueTaskItem = components['schemas']['QueueTaskItem']

const GENERIC_SECURITY_ERR_MSG =
  'Forbidden: A security error has occurred. Please check the terminal logs'

/**
 * API routes for ComfyUI Manager
 */
enum ManagerRoute {
  START_QUEUE = 'manager/queue/start',
  RESET_QUEUE = 'manager/queue/reset',
  QUEUE_STATUS = 'manager/queue/status',
  UPDATE_ALL = 'manager/queue/update_all',
  UPDATE_COMFYUI = 'manager/queue/update_comfyui',
  LIST_INSTALLED = 'customnode/installed',
  GET_NODES = 'customnode/getmappings',
  IMPORT_FAIL_INFO = 'customnode/import_fail_info',
  IMPORT_FAIL_INFO_BULK = 'customnode/import_fail_info_bulk',
  REBOOT = 'manager/reboot',
  IS_LEGACY_MANAGER_UI = 'manager/is_legacy_manager_ui',
  TASK_HISTORY = 'manager/queue/history',
  QUEUE_TASK = 'manager/queue/task'
}

const managerApiClient = axios.create({
  baseURL: api.apiURL('/v2/'),
  headers: {
    'Content-Type': 'application/json'
  }
})

/**
 * Service for interacting with the ComfyUI Manager API
 * Provides methods for managing packs, ComfyUI-Manager queue operations, and system functions
 * Note: This service should only be used when Manager state is NEW_UI
 */
export const useComfyManagerService = () => {
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  // Check if manager service should be available
  const isManagerServiceAvailable = () => {
    const managerState = useManagerState()
    return managerState.isNewManagerUI.value
  }

  const handleRequestError = (
    err: unknown,
    context: string,
    routeSpecificErrors?: Record<number, string>
  ) => {
    // Don't treat cancellation as an error
    if (isAbortError(err)) return

    let message: string
    if (!axios.isAxiosError(err)) {
      message = `${context} failed: ${err instanceof Error ? err.message : String(err)}`
    } else {
      const axiosError = err as AxiosError<{ message: string }>
      const status = axiosError.response?.status
      if (status && routeSpecificErrors?.[status]) {
        message = routeSpecificErrors[status]
      } else if (status === 404) {
        message = 'Could not connect to ComfyUI-Manager'
      } else {
        message =
          axiosError.response?.data?.message ??
          `${context} failed with status ${status}`
      }
    }

    error.value = message
  }

  const executeRequest = async <T>(
    requestCall: () => Promise<AxiosResponse<T>>,
    options: {
      errorContext: string
      routeSpecificErrors?: Record<number, string>
      isQueueOperation?: boolean
    }
  ): Promise<T | null> => {
    const { errorContext, routeSpecificErrors, isQueueOperation } = options

    // Block service calls if not in NEW_UI state
    if (!isManagerServiceAvailable()) {
      error.value = 'Manager service is not available in current mode'
      return null
    }

    isLoading.value = true
    error.value = null

    try {
      const response = await requestCall()
      if (isQueueOperation) await startQueue()
      return response.data
    } catch (err) {
      handleRequestError(err, errorContext, routeSpecificErrors)
      return null
    } finally {
      isLoading.value = false
    }
  }

  const startQueue = async (signal?: AbortSignal) => {
    const errorContext = 'Starting ComfyUI-Manager job queue'
    const routeSpecificErrors = {
      201: 'Created: ComfyUI-Manager job queue is already running'
    }

    return executeRequest<null>(
      () => managerApiClient.get(ManagerRoute.START_QUEUE, { signal }),
      { errorContext, routeSpecificErrors }
    )
  }

  const getQueueStatus = async (client_id?: string, signal?: AbortSignal) => {
    const errorContext = 'Getting ComfyUI-Manager queue status'

    return executeRequest<ManagerQueueStatus>(
      () =>
        managerApiClient.get(ManagerRoute.QUEUE_STATUS, {
          params: client_id ? { client_id } : undefined,
          signal
        }),
      { errorContext }
    )
  }

  const listInstalledPacks = async (signal?: AbortSignal) => {
    const errorContext = 'Fetching installed packs'

    return executeRequest<InstalledPacksResponse>(
      () => managerApiClient.get(ManagerRoute.LIST_INSTALLED, { signal }),
      { errorContext }
    )
  }

  const getImportFailInfo = async (signal?: AbortSignal) => {
    const errorContext = 'Fetching import failure information'

    return executeRequest<Record<string, unknown>>(
      () => managerApiClient.get(ManagerRoute.IMPORT_FAIL_INFO, { signal }),
      { errorContext }
    )
  }

  const getImportFailInfoBulk = async (
    params: components['schemas']['ImportFailInfoBulkRequest'] = {},
    signal?: AbortSignal
  ) => {
    const errorContext = 'Fetching bulk import failure information'

    if (!params.cnr_ids?.length && !params.urls?.length) {
      return {}
    }

    return executeRequest<components['schemas']['ImportFailInfoBulkResponse']>(
      () =>
        managerApiClient.post(ManagerRoute.IMPORT_FAIL_INFO_BULK, params, {
          signal
        }),
      { errorContext }
    )
  }

  const queueTask = async (
    kind: QueueTaskItem['kind'],
    params: QueueTaskItem['params'],
    ui_id?: string,
    signal?: AbortSignal
  ) => {
    const task: QueueTaskItem = {
      kind,
      params,
      ui_id: ui_id || uuidv4(),
      client_id: api.clientId ?? api.initialClientId ?? 'unknown'
    }

    const errorContext = `Queueing ${task.kind} task`
    const routeSpecificErrors = {
      403: GENERIC_SECURITY_ERR_MSG,
      404: `Not Found: Task could not be queued`
    }

    return executeRequest<null>(
      () => managerApiClient.post(ManagerRoute.QUEUE_TASK, task, { signal }),
      { errorContext, routeSpecificErrors, isQueueOperation: true }
    )
  }

  const installPack = async (
    params: InstallPackParams,
    ui_id?: string,
    signal?: AbortSignal
  ) => {
    return queueTask('install', params, ui_id, signal)
  }

  const uninstallPack = async (
    params: components['schemas']['UninstallPackParams'],
    ui_id?: string,
    signal?: AbortSignal
  ) => {
    return queueTask('uninstall', params, ui_id, signal)
  }

  const disablePack = async (
    params: components['schemas']['DisablePackParams'],
    ui_id?: string,
    signal?: AbortSignal
  ): Promise<null> => {
    return queueTask('disable', params, ui_id, signal)
  }

  const enablePack = async (
    params: components['schemas']['EnablePackParams'],
    ui_id?: string,
    signal?: AbortSignal
  ): Promise<null> => {
    return queueTask('enable', params, ui_id, signal)
  }

  const updatePack = async (
    params: components['schemas']['UpdatePackParams'],
    ui_id?: string,
    signal?: AbortSignal
  ): Promise<null> => {
    return queueTask('update', params, ui_id, signal)
  }

  const updateAllPacks = async (
    params: UpdateAllPacksParams = {},
    ui_id?: string,
    signal?: AbortSignal
  ) => {
    const errorContext = 'Updating all packs'
    const routeSpecificErrors = {
      403: 'Forbidden: To use this action, a security_level of `middle or below` is required',
      401: 'Unauthorized: ComfyUI-Manager job queue is busy'
    }

    const queryParams = {
      mode: params.mode,
      client_id: api.clientId ?? api.initialClientId ?? 'unknown',
      ui_id: ui_id || uuidv4()
    }

    return executeRequest<null>(
      () =>
        managerApiClient.get(ManagerRoute.UPDATE_ALL, {
          params: queryParams,
          signal
        }),
      { errorContext, routeSpecificErrors, isQueueOperation: true }
    )
  }

  const updateComfyUI = async (
    params: UpdateComfyUIParams = { is_stable: true },
    ui_id?: string,
    signal?: AbortSignal
  ) => {
    const errorContext = 'Updating ComfyUI'
    const routeSpecificErrors = {
      400: 'Bad Request: Missing required parameters',
      403: 'Forbidden: To use this action, a security_level of `middle or below` is required'
    }

    const queryParams = {
      client_id: api.clientId ?? api.initialClientId ?? 'unknown',
      ui_id: ui_id || uuidv4(),
      ...params
    }

    return executeRequest<null>(
      () =>
        managerApiClient.get(ManagerRoute.UPDATE_COMFYUI, {
          params: queryParams,
          signal
        }),
      { errorContext, routeSpecificErrors, isQueueOperation: true }
    )
  }

  const rebootComfyUI = async (signal?: AbortSignal) => {
    const errorContext = 'Rebooting ComfyUI'
    const routeSpecificErrors = {
      403: 'Forbidden: Rebooting ComfyUI requires security_level of middle or below'
    }

    return executeRequest<null>(
      () => managerApiClient.get(ManagerRoute.REBOOT, { signal }),
      { errorContext, routeSpecificErrors }
    )
  }

  const isLegacyManagerUI = async (signal?: AbortSignal) => {
    const errorContext = 'Checking if user set Manager to use the legacy UI'

    return executeRequest<{ is_legacy_manager_ui: boolean }>(
      () => managerApiClient.get(ManagerRoute.IS_LEGACY_MANAGER_UI, { signal }),
      { errorContext }
    )
  }

  const getTaskHistory = async (
    options: {
      ui_id?: string
      max_items?: number
      client_id?: string
      offset?: number
    } = {},
    signal?: AbortSignal
  ) => {
    const errorContext = 'Getting ComfyUI-Manager task history'

    return executeRequest<ManagerTaskHistory>(
      () =>
        managerApiClient.get(ManagerRoute.TASK_HISTORY, {
          params: options,
          signal
        }),
      { errorContext }
    )
  }

  return {
    // State
    isLoading,
    error,

    // Queue operations
    startQueue,
    getQueueStatus,
    getTaskHistory,

    // Pack management
    listInstalledPacks,
    getImportFailInfo,
    getImportFailInfoBulk,
    installPack,
    uninstallPack,
    enablePack,
    disablePack,
    updatePack,
    updateAllPacks,

    // System operations
    updateComfyUI,
    rebootComfyUI,
    isLegacyManagerUI
  }
}
