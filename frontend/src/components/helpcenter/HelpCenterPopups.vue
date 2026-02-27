<template>
  <!-- Help Center Popup positioned within canvas area -->
  <Teleport to="body">
    <div
      v-if="isHelpCenterVisible"
      class="help-center-popup"
      :class="{
        'sidebar-left':
          triggerLocation === 'sidebar' && sidebarLocation === 'left',
        'sidebar-right':
          triggerLocation === 'sidebar' && sidebarLocation === 'right',
        'topbar-right': triggerLocation === 'topbar',
        'small-sidebar': isSmall
      }"
    >
      <HelpCenterMenuContent @close="closeHelpCenter" />
    </div>
  </Teleport>

  <!-- Release Notification Toast positioned within canvas area -->
  <Teleport to="#graph-canvas-container">
    <ReleaseNotificationToast
      :class="{
        'sidebar-left': sidebarLocation === 'left',
        'sidebar-right': sidebarLocation === 'right',
        'small-sidebar': isSmall
      }"
    />
  </Teleport>

  <!-- WhatsNew Popup positioned within canvas area -->
  <Teleport to="#graph-canvas-container">
    <WhatsNewPopup
      :class="{
        'sidebar-left': sidebarLocation === 'left',
        'sidebar-right': sidebarLocation === 'right',
        'small-sidebar': isSmall
      }"
      @whats-new-dismissed="handleWhatsNewDismissed"
    />
  </Teleport>

  <!-- Backdrop to close popup when clicking outside -->
  <Teleport to="body">
    <div
      v-if="isHelpCenterVisible"
      class="help-center-backdrop"
      @click="closeHelpCenter"
    />
  </Teleport>
</template>

<script setup lang="ts">
import { useHelpCenter } from '@/composables/useHelpCenter'
import ReleaseNotificationToast from '@/platform/updates/components/ReleaseNotificationToast.vue'
import WhatsNewPopup from '@/platform/updates/components/WhatsNewPopup.vue'

import HelpCenterMenuContent from './HelpCenterMenuContent.vue'

const { isSmall = false } = defineProps<{
  isSmall?: boolean
}>()

const {
  isHelpCenterVisible,
  triggerLocation,
  sidebarLocation,
  closeHelpCenter,
  handleWhatsNewDismissed
} = useHelpCenter()
</script>

<style scoped>
.help-center-backdrop {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 9999;
  background: transparent;
}

.help-center-popup {
  position: absolute;
  bottom: 1rem;
  z-index: 10000;
  animation: slideInUp 0.2s ease-out;
  pointer-events: auto;
}

.help-center-popup.sidebar-left {
  left: 1rem;
}

.help-center-popup.sidebar-left.small-sidebar {
  left: 1rem;
}

.help-center-popup.sidebar-right {
  right: 1rem;
}

.help-center-popup.topbar-right {
  top: 2rem;
  right: 1rem;
  bottom: auto;
  animation: slideInDown 0.2s ease-out;
}

@keyframes slideInDown {
  from {
    opacity: 0;
    transform: translateY(-20px);
  }

  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes slideInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }

  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
