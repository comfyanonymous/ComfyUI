<template>
  <SidebarIcon
    icon="pi pi-question-circle"
    class="comfy-help-center-btn"
    :label="$t('menu.help')"
    :tooltip="$t('sideToolbar.helpCenter')"
    :icon-badge="shouldShowRedDot ? '•' : ''"
    badge-class="-top-1 -right-1 min-w-2 w-2 h-2 p-0 rounded-full text-[0px] bg-[#ff3b30]"
    :is-small="isSmall"
    @click="toggleHelpCenter"
  />
</template>

<script setup lang="ts">
import { useHelpCenter } from '@/composables/useHelpCenter'

import SidebarIcon from './SidebarIcon.vue'

defineProps<{
  isSmall: boolean
}>()

const { shouldShowRedDot, toggleHelpCenter } = useHelpCenter()
</script>
