<!-- The main global dialog to show various things -->
<template>
  <Dialog
    v-for="item in dialogStore.dialogStack"
    :key="item.key"
    v-model:visible="item.visible"
    class="global-dialog"
    v-bind="item.dialogComponentProps"
    :pt="getDialogPt(item)"
    :aria-labelledby="item.key"
  >
    <template #header>
      <div v-if="!item.dialogComponentProps?.headless">
        <component
          :is="item.headerComponent"
          v-if="item.headerComponent"
          v-bind="item.headerProps"
          :id="item.key"
        />
        <h3 v-else :id="item.key">
          {{ item.title || ' ' }}
        </h3>
      </div>
    </template>

    <component
      :is="item.component"
      v-bind="item.contentProps"
      :maximized="item.dialogComponentProps.maximized"
    />

    <template v-if="item.footerComponent" #footer>
      <component :is="item.footerComponent" v-bind="item.footerProps" />
    </template>
  </Dialog>
</template>

<script setup lang="ts">
import { merge } from 'es-toolkit/compat'
import Dialog from 'primevue/dialog'
import type { DialogPassThroughOptions } from 'primevue/dialog'
import { computed } from 'vue'

import { useFeatureFlags } from '@/composables/useFeatureFlags'
import { isCloud } from '@/platform/distribution/types'
import type { DialogComponentProps } from '@/stores/dialogStore'
import { useDialogStore } from '@/stores/dialogStore'

const { flags } = useFeatureFlags()
const teamWorkspacesEnabled = computed(
  () => isCloud && flags.teamWorkspacesEnabled
)

const dialogStore = useDialogStore()

function getDialogPt(item: {
  key: string
  dialogComponentProps: DialogComponentProps
}): DialogPassThroughOptions {
  const isWorkspaceSettingsDialog =
    item.key === 'global-settings' && teamWorkspacesEnabled.value
  const basePt = item.dialogComponentProps.pt || {}

  if (isWorkspaceSettingsDialog) {
    return merge(basePt, {
      mask: { class: 'p-8' }
    })
  }
  return basePt
}
</script>

<style>
.global-dialog {
  max-width: calc(100vw - 1rem);
}

.global-dialog .p-dialog-header {
  padding: calc(var(--spacing) * 2);
  padding-bottom: 0;
}

.global-dialog .p-dialog-content {
  padding: calc(var(--spacing) * 2);
  padding-top: 0;
}

@media (min-width: 1536px) {
  .global-dialog .p-dialog-header {
    padding: var(--p-dialog-header-padding);
    padding-bottom: 0;
  }

  .global-dialog .p-dialog-content {
    padding: var(--p-dialog-content-padding);
    padding-top: 0;
  }
}

/* Workspace mode: wider settings dialog */
.settings-dialog-workspace {
  width: 100%;
  max-width: 1440px;
  height: 100%;
}

.settings-dialog-workspace .p-dialog-content {
  width: 100%;
  height: 100%;
  overflow-y: auto;
}

.manager-dialog {
  height: 80vh;
  max-width: 1724px;
  max-height: 1026px;
}

@media (min-width: 3000px) {
  .manager-dialog {
    max-width: 2200px;
    max-height: 1320px;
  }
}
</style>
