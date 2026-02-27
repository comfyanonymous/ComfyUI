<template>
  <Drawer
    v-model:visible="terminalVisible"
    :header
    position="bottom"
    style="height: max(50vh, 34rem)"
  >
    <BaseTerminal @created="terminalCreated" @unmounted="terminalUnmounted" />
  </Drawer>
</template>

<script setup lang="ts">
import type { Terminal } from '@xterm/xterm'
import Drawer from 'primevue/drawer'
import type { Ref } from 'vue'
import { onMounted } from 'vue'

import BaseTerminal from '@/components/bottomPanel/tabs/terminal/BaseTerminal.vue'
import type { useTerminal } from '@/composables/bottomPanelTabs/useTerminal'
import { useTerminalBuffer } from '@/composables/bottomPanelTabs/useTerminalBuffer'
import { electronAPI } from '@/utils/envUtil'

// Model
const terminalVisible = defineModel<boolean>({ required: true })
const props = defineProps<{
  header: string
  defaultMessage: string
}>()

const electron = electronAPI()

/** The actual output of all terminal commands - not rendered */
const buffer = useTerminalBuffer()
let xterm: Terminal | null = null

// Created and destroyed with the Drawer - contents copied from hidden buffer
const terminalCreated = (
  { terminal, useAutoSize }: ReturnType<typeof useTerminal>,
  root: Ref<HTMLElement | undefined>
) => {
  xterm = terminal
  useAutoSize({ root, autoRows: true, autoCols: true })
  terminal.write(props.defaultMessage)
  buffer.copyTo(terminal)

  terminal.options.cursorBlink = false
  terminal.options.cursorStyle = 'bar'
  terminal.options.cursorInactiveStyle = 'bar'
  terminal.options.disableStdin = true
}

const terminalUnmounted = () => {
  xterm = null
}

onMounted(async () => {
  electron.onLogMessage((message: string) => {
    buffer.write(message)
    xterm?.write(message)
  })
})
</script>
