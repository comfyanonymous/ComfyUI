<template>
  <div
    ref="rootEl"
    class="relative overflow-hidden h-full w-full bg-neutral-900"
  >
    <div class="p-terminal rounded-none h-full w-full p-2">
      <div ref="terminalEl" class="h-full terminal-host" />
    </div>
    <Button
      v-tooltip.left="{
        value: tooltipText,
        showDelay: 300
      }"
      icon="pi pi-copy"
      severity="secondary"
      size="small"
      :class="
        cn('absolute top-2 right-8 transition-opacity', {
          'opacity-0 pointer-events-none select-none': !isHovered
        })
      "
      :aria-label="tooltipText"
      @click="handleCopy"
    />
  </div>
</template>

<script setup lang="ts">
import { useElementHover, useEventListener } from '@vueuse/core'
import type { IDisposable } from '@xterm/xterm'
import Button from 'primevue/button'
import type { Ref } from 'vue'
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import { useTerminal } from '@/composables/bottomPanelTabs/useTerminal'
import { electronAPI, isElectron } from '@/utils/envUtil'
import { cn } from '@/utils/tailwindUtil'

const { t } = useI18n()

const emit = defineEmits<{
  created: [ReturnType<typeof useTerminal>, Ref<HTMLElement | undefined>]
  unmounted: []
}>()
const terminalEl = ref<HTMLElement | undefined>()
const rootEl = ref<HTMLElement | undefined>()
const hasSelection = ref(false)

const isHovered = useElementHover(rootEl)

const terminalData = useTerminal(terminalEl)
emit('created', terminalData, ref(rootEl))

const { terminal } = terminalData
let selectionDisposable: IDisposable | undefined

const tooltipText = computed(() => {
  return hasSelection.value
    ? t('serverStart.copySelectionTooltip')
    : t('serverStart.copyAllTooltip')
})

const handleCopy = async () => {
  const existingSelection = terminal.getSelection()
  const shouldSelectAll = !existingSelection
  if (shouldSelectAll) terminal.selectAll()

  const selectedText = shouldSelectAll
    ? terminal.getSelection()
    : existingSelection

  if (selectedText) {
    await navigator.clipboard.writeText(selectedText)

    if (shouldSelectAll) {
      terminal.clearSelection()
    }
  }
}

const showContextMenu = (event: MouseEvent) => {
  event.preventDefault()
  electronAPI()?.showContextMenu({ type: 'text' })
}

if (isElectron()) {
  useEventListener(terminalEl, 'contextmenu', showContextMenu)
}

onMounted(() => {
  selectionDisposable = terminal.onSelectionChange(() => {
    hasSelection.value = terminal.hasSelection()
  })
})

onUnmounted(() => {
  selectionDisposable?.dispose()
  emit('unmounted')
})
</script>

<style scoped>
/* xterm renders its internal DOM outside Vue templates, so :deep selectors are
 * required to style those generated nodes.
 */
:deep(.p-terminal) .xterm {
  overflow: hidden;
}

:deep(.p-terminal) .xterm-screen {
  overflow: hidden;
  background-color: var(--color-neutral-900);
}
</style>
