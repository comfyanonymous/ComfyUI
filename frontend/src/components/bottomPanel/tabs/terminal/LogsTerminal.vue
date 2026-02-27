<template>
  <div class="h-full w-full bg-transparent">
    <p v-if="errorMessage" class="p-4 text-center">
      {{ errorMessage }}
    </p>
    <ProgressSpinner
      v-else-if="loading"
      class="relative inset-0 z-10 flex h-full items-center justify-center"
    />
    <BaseTerminal v-show="!loading" @created="terminalCreated" />
  </div>
</template>

<script setup lang="ts">
import { until } from '@vueuse/core'
import { storeToRefs } from 'pinia'
import ProgressSpinner from 'primevue/progressspinner'
import type { Ref } from 'vue'
import { onMounted, onUnmounted, ref } from 'vue'

import type { useTerminal } from '@/composables/bottomPanelTabs/useTerminal'
import type { LogEntry, LogsWsMessage } from '@/schemas/apiSchema'
import { api } from '@/scripts/api'
import { useExecutionStore } from '@/stores/executionStore'

import BaseTerminal from './BaseTerminal.vue'

const errorMessage = ref('')
const loading = ref(true)

const terminalCreated = (
  { terminal, useAutoSize }: ReturnType<typeof useTerminal>,
  root: Ref<HTMLElement | undefined>
) => {
  // Auto-size terminal to fill container width.
  // minCols: 80 ensures minimum width for colab environments.
  // See https://github.com/comfyanonymous/ComfyUI/issues/6396
  useAutoSize({ root, autoRows: true, autoCols: true, minCols: 80 })

  const update = (entries: Array<LogEntry>) => {
    terminal.write(entries.map((e) => e.m).join(''))
  }

  const logReceived = (e: CustomEvent<LogsWsMessage>) => {
    update(e.detail.entries)
  }

  const loadLogEntries = async () => {
    const logs = await api.getRawLogs()
    update(logs.entries)
  }

  const watchLogs = async () => {
    const { clientId } = storeToRefs(useExecutionStore())
    if (!clientId.value) {
      await until(clientId).not.toBeNull()
    }
    await api.subscribeLogs(true)
    api.addEventListener('logs', logReceived)
  }

  onMounted(async () => {
    try {
      await loadLogEntries()
    } catch (err) {
      console.error('Error loading logs', err)
      // On older backends the endpoints won't exist
      errorMessage.value =
        'Unable to load logs, please ensure you have updated your ComfyUI backend.'
      return
    }

    await watchLogs()
    loading.value = false
  })

  onUnmounted(async () => {
    if (api.clientId) {
      await api.subscribeLogs(false)
    }
    api.removeEventListener('logs', logReceived)
  })
}
</script>

<style scoped>
:deep(.p-terminal) .xterm {
  overflow-x: auto;
}

:deep(.p-terminal) .xterm-screen {
  overflow-y: hidden;
}
</style>
