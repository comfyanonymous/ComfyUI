<template>
  <div class="flex h-full flex-col p-4">
    <div class="min-h-0 flex-1 overflow-auto">
      <ShortcutsList
        :commands="viewControlsCommands"
        :subcategories="viewControlsSubcategories"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import {
  VIEW_CONTROLS_CONFIG,
  useCommandSubcategories
} from '@/composables/bottomPanelTabs/useCommandSubcategories'
import { useCommandStore } from '@/stores/commandStore'

import ShortcutsList from './ShortcutsList.vue'

const commandStore = useCommandStore()

const viewControlsCommands = computed(() =>
  commandStore.commands.filter((cmd) => cmd.category === 'view-controls')
)

const { subcategories: viewControlsSubcategories } = useCommandSubcategories(
  viewControlsCommands,
  VIEW_CONTROLS_CONFIG
)
</script>
