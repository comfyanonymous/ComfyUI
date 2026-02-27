<template>
  <BaseTerminal @created="terminalCreated" />
</template>

<script setup lang="ts">
import type { IDisposable } from '@xterm/xterm'
import type { Ref } from 'vue'
import { onMounted, onUnmounted } from 'vue'

import type { useTerminal } from '@/composables/bottomPanelTabs/useTerminal'
import { electronAPI } from '@/utils/envUtil'

import BaseTerminal from './BaseTerminal.vue'

const terminalCreated = (
  { terminal, useAutoSize }: ReturnType<typeof useTerminal>,
  root: Ref<HTMLElement | undefined>
) => {
  const terminalApi = electronAPI().Terminal

  let offData: IDisposable
  let offOutput: () => void

  useAutoSize({
    root,
    autoRows: true,
    autoCols: true,
    onResize: async () => {
      // If we aren't visible, don't resize
      if (!terminal.element?.offsetParent) return

      await terminalApi.resize(terminal.cols, terminal.rows)
    }
  })

  onMounted(async () => {
    offData = terminal.onData(async (message: string) => {
      await terminalApi.write(message)
    })

    offOutput = terminalApi.onOutput((message) => {
      terminal.write(message)
    })

    const restore = await terminalApi.restore()
    setTimeout(() => {
      if (restore.buffer.length) {
        terminal.resize(restore.size.cols, restore.size.rows)
        terminal.write(restore.buffer.join(''))
      }
    }, 500)
  })

  onUnmounted(() => {
    offData?.dispose()
    offOutput?.()
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
