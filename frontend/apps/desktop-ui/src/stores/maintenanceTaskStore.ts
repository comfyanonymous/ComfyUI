import type { InstallValidation } from '@comfyorg/comfyui-electron-types'
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import { DESKTOP_MAINTENANCE_TASKS } from '@/constants/desktopMaintenanceTasks'
import type { MaintenanceTask } from '@/types/desktop/maintenanceTypes'
import { electronAPI } from '@/utils/envUtil'

/** State of a maintenance task, managed by the maintenance task store. */
type MaintenanceTaskState = 'warning' | 'error' | 'OK' | 'skipped'

// Type not exported by API
type ValidationState = InstallValidation['basePath']
// Add index to API type
type IndexedUpdate = InstallValidation & Record<string, ValidationState>

/** State of a maintenance task, managed by the maintenance task store. */
export class MaintenanceTaskRunner {
  constructor(readonly task: MaintenanceTask) {}

  private _state?: MaintenanceTaskState
  /** The current state of the task. Setter also controls {@link resolved} as a side-effect. */
  get state() {
    return this._state
  }

  /** Updates the task state and {@link resolved} status. */
  setState(value: MaintenanceTaskState) {
    // Mark resolved
    if (this._state === 'error' && value === 'OK') this.resolved = true
    // Mark unresolved (if previously resolved)
    if (value === 'error') this.resolved &&= false

    this._state = value
  }

  /** `true` if the task has been resolved (was `error`, now `OK`). This is a side-effect of the {@link state} setter. */
  resolved?: boolean

  /** Whether the task state is currently being refreshed. */
  refreshing?: boolean
  /** Whether the task is currently running. */
  executing?: boolean
  /** The error message that occurred when the task failed. */
  error?: string

  update(update: IndexedUpdate) {
    const state = update[this.task.id]

    this.refreshing = state === undefined
    if (state) this.setState(state)
  }

  finaliseUpdate(update: IndexedUpdate) {
    this.refreshing = false
    this.setState(update[this.task.id] ?? 'skipped')
  }

  /** Wraps the execution of a maintenance task, updating state and rethrowing errors. */
  async execute(task: MaintenanceTask) {
    try {
      this.executing = true
      const success = await task.execute()
      if (!success) return false

      this.error = undefined
      return true
    } catch (error) {
      this.error = (error as Error)?.message
      throw error
    } finally {
      this.executing = false
    }
  }
}

/**
 * User-initiated maintenance tasks.  Currently only used by the desktop app maintenance view.
 *
 * Includes running state, task list, and execution / refresh logic.
 * @returns The maintenance task store
 */
export const useMaintenanceTaskStore = defineStore('maintenanceTask', () => {
  /** Refresh should run for at least this long, even if it completes much faster. Ensures refresh feels like it is doing something. */
  const electron = electronAPI()

  // Reactive state
  const lastUpdate = ref<InstallValidation | null>(null)
  const isRefreshing = ref(false)
  const isRunningTerminalCommand = computed(() =>
    tasks.value
      .filter((task) => task.usesTerminal)
      .some((task) => getRunner(task)?.executing)
  )
  const isRunningInstallationFix = computed(() =>
    tasks.value
      .filter((task) => task.isInstallationFix)
      .some((task) => getRunner(task)?.executing)
  )

  const unsafeBasePath = computed(
    () => lastUpdate.value?.unsafeBasePath === true
  )
  const unsafeBasePathReason = computed(
    () => lastUpdate.value?.unsafeBasePathReason
  )

  // Task list
  const tasks = ref(DESKTOP_MAINTENANCE_TASKS)

  const taskRunners = ref(
    new Map<MaintenanceTask['id'], MaintenanceTaskRunner>(
      DESKTOP_MAINTENANCE_TASKS.map((x) => [x.id, new MaintenanceTaskRunner(x)])
    )
  )

  /** True if any tasks are in an error state. */
  const anyErrors = computed(() =>
    tasks.value.some((task) => getRunner(task).state === 'error')
  )

  /**
   * Returns the matching state object for a task.
   * @param task Task to get the matching state object for
   * @returns The state object for this task
   */
  const getRunner = (task: MaintenanceTask) => taskRunners.value.get(task.id)!

  /**
   * Updates the task list with the latest validation state.
   * @param validationUpdate Update details passed in by electron
   */
  const processUpdate = (validationUpdate: InstallValidation) => {
    lastUpdate.value = validationUpdate
    const update = validationUpdate as IndexedUpdate
    isRefreshing.value = true

    // Update each task state
    for (const task of tasks.value) {
      getRunner(task).update(update)
    }

    // Final update
    if (!update.inProgress && isRefreshing.value) {
      isRefreshing.value = false

      for (const task of tasks.value) {
        getRunner(task).finaliseUpdate(update)
      }
    }
  }

  /** Clears the resolved status of tasks (when changing filters) */
  const clearResolved = () => {
    for (const task of tasks.value) {
      getRunner(task).resolved &&= false
    }
  }

  /** @todo Refreshes Electron tasks only. */
  const refreshDesktopTasks = async () => {
    isRefreshing.value = true
    await electron.Validation.validateInstallation(processUpdate)
  }

  const execute = async (task: MaintenanceTask) => {
    const success = await getRunner(task).execute(task)
    if (success && task.isInstallationFix) {
      await refreshDesktopTasks()
    }
    return success
  }

  return {
    tasks,
    isRefreshing,
    isRunningTerminalCommand,
    isRunningInstallationFix,
    unsafeBasePath,
    unsafeBasePathReason,
    execute,
    getRunner,
    processUpdate,
    clearResolved,
    /** True if any tasks are in an error state. */
    anyErrors,
    refreshDesktopTasks
  }
})
