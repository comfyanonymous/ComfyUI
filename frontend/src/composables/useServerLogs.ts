import { useEventListener } from '@vueuse/core'
import { onUnmounted, ref } from 'vue'

import type { LogsWsMessage } from '@/schemas/apiSchema'
import { api } from '@/scripts/api'
import type { components } from '@/workbench/extensions/manager/types/generatedManagerTypes'

const LOGS_MESSAGE_TYPE = 'logs'
const MANAGER_WS_TASK_DONE_NAME = 'cm-task-completed'
const MANAGER_WS_TASK_STARTED_NAME = 'cm-task-started'

type ManagerWsTaskDoneMsg = components['schemas']['MessageTaskDone']
type ManagerWsTaskStartedMsg = components['schemas']['MessageTaskStarted']

interface UseServerLogsOptions {
  ui_id?: string
  immediate?: boolean
  messageFilter?: (message: string) => boolean
}

export const useServerLogs = (options: UseServerLogsOptions = {}) => {
  const {
    ui_id,
    immediate = false,
    messageFilter = (msg: string) => Boolean(msg.trim())
  } = options

  const logs = ref<string[]>([])
  const isTaskStarted = ref(!ui_id) // If no ui_id, capture all logs immediately
  let stopLogs: ReturnType<typeof useEventListener> | null = null
  let stopTaskDone: ReturnType<typeof useEventListener> | null = null
  let stopTaskStarted: ReturnType<typeof useEventListener> | null = null

  const isValidLogEvent = (event: CustomEvent<LogsWsMessage>) =>
    event?.type === LOGS_MESSAGE_TYPE && event.detail?.entries?.length > 0

  const parseLogMessage = (event: CustomEvent<LogsWsMessage>) =>
    event.detail.entries.map((e) => e.m).filter(messageFilter)

  const handleLogMessage = (event: CustomEvent<LogsWsMessage>) => {
    // Only capture logs if this task has started
    if (!isTaskStarted.value) return

    if (isValidLogEvent(event)) {
      const messages = parseLogMessage(event)
      if (messages.length > 0) {
        logs.value.push(...messages)
      }
    }
  }

  const handleTaskStarted = (event: CustomEvent<ManagerWsTaskStartedMsg>) => {
    if (ui_id && event?.detail?.ui_id === ui_id) {
      isTaskStarted.value = true
    }
  }

  const handleTaskDone = (event: CustomEvent<ManagerWsTaskDoneMsg>) => {
    if (ui_id && event?.detail?.ui_id === ui_id) {
      isTaskStarted.value = false
    }
  }

  const start = async () => {
    await api.subscribeLogs(true)
    stopLogs = useEventListener(api, LOGS_MESSAGE_TYPE, handleLogMessage)

    if (ui_id) {
      stopTaskStarted = useEventListener(
        api,
        MANAGER_WS_TASK_STARTED_NAME,
        handleTaskStarted
      )
      stopTaskDone = useEventListener(
        api,
        MANAGER_WS_TASK_DONE_NAME,
        handleTaskDone
      )
    }
  }

  const stopListening = async () => {
    stopLogs?.()
    stopTaskStarted?.()
    stopTaskDone?.()
    stopLogs = null
    stopTaskStarted = null
    stopTaskDone = null
    await api.subscribeLogs(false)
  }

  if (immediate) {
    void start()
  }

  onUnmounted(async () => {
    await stopListening()
    logs.value = []
  })

  return {
    logs,
    startListening: start,
    stopListening
  }
}
