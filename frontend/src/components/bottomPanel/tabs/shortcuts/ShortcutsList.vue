<template>
  <div class="shortcuts-list flex justify-center">
    <div
      data-testid="shortcuts-columns"
      class="grid h-full w-[90%] grid-cols-1 gap-4 md:grid-cols-3 md:gap-24"
    >
      <div
        v-for="(subcategoryCommands, subcategory) in filteredSubcategories"
        :key="subcategory"
        class="flex flex-col"
      >
        <h3
          class="subcategory-title mb-4 text-xs font-bold tracking-wide text-text-secondary uppercase"
        >
          {{ getSubcategoryTitle(subcategory) }}
        </h3>

        <div class="flex flex-col gap-1">
          <div
            v-for="command in subcategoryCommands"
            :key="command.id"
            class="shortcut-item flex items-center justify-between rounded py-2 transition-colors duration-200"
          >
            <div class="shortcut-info grow pr-4">
              <div class="shortcut-name text-sm font-medium">
                {{ t(`commands.${normalizeI18nKey(command.id)}.label`) }}
              </div>
            </div>

            <div class="keybinding-display shrink-0">
              <div
                class="keybinding-combo flex gap-1"
                :aria-label="`Keyboard shortcut: ${command.keybinding!.combo.getKeySequences().join(' + ')}`"
              >
                <span
                  v-for="key in command.keybinding!.combo.getKeySequences()"
                  :key="key"
                  class="key-badge min-w-6 rounded bg-muted-background px-2 py-1 text-center font-mono text-xs"
                >
                  {{ formatKey(key) }}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import type { ComfyCommandImpl } from '@/stores/commandStore'
import { normalizeI18nKey } from '@/utils/formatUtil'

const { t } = useI18n()

const { subcategories } = defineProps<{
  subcategories: Record<string, ComfyCommandImpl[]>
}>()

const filteredSubcategories = computed(() => {
  const result: Record<string, ComfyCommandImpl[]> = {}

  for (const [subcategory, commands] of Object.entries(subcategories)) {
    result[subcategory] = commands.filter((cmd) => !!cmd.keybinding)
  }

  return result
})

const getSubcategoryTitle = (subcategory: string): string => {
  const titleMap: Record<string, string> = {
    workflow: t('shortcuts.subcategories.workflow'),
    node: t('shortcuts.subcategories.node'),
    queue: t('shortcuts.subcategories.queue'),
    view: t('shortcuts.subcategories.view'),
    'panel-controls': t('shortcuts.subcategories.panelControls')
  }

  return titleMap[subcategory] || subcategory
}

const formatKey = (key: string): string => {
  const keyMap: Record<string, string> = {
    Control: 'Ctrl',
    Meta: 'Cmd',
    ArrowUp: '↑',
    ArrowDown: '↓',
    ArrowLeft: '←',
    ArrowRight: '→',
    Backspace: '⌫',
    Delete: '⌦',
    Enter: '↵',
    Escape: 'Esc',
    Tab: '⇥',
    ' ': 'Space'
  }

  return keyMap[key] || key
}
</script>
