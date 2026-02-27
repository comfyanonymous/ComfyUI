<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import WorkflowActionsDropdown from '@/components/common/WorkflowActionsDropdown.vue'
import Button from '@/components/ui/button/Button.vue'
import { useWorkflowTemplateSelectorDialog } from '@/composables/useWorkflowTemplateSelectorDialog'
import { useCommandStore } from '@/stores/commandStore'
import { useWorkspaceStore } from '@/stores/workspaceStore'
import { cn } from '@/utils/tailwindUtil'
import { useAppMode } from '@/composables/useAppMode'

const { t } = useI18n()
const commandStore = useCommandStore()
const workspaceStore = useWorkspaceStore()
const { enableAppBuilder, setMode } = useAppMode()
const tooltipOptions = { showDelay: 300, hideDelay: 300 }

const isAssetsActive = computed(
  () => workspaceStore.sidebarTab.activeSidebarTab?.id === 'assets'
)
const isWorkflowsActive = computed(
  () => workspaceStore.sidebarTab.activeSidebarTab?.id === 'workflows'
)

function enterBuilderMode() {
  setMode('builder:select')
}

function openAssets() {
  void commandStore.execute('Workspace.ToggleSidebarTab.assets')
}

function showApps() {
  void commandStore.execute('Workspace.ToggleSidebarTab.workflows')
}

function openTemplates() {
  useWorkflowTemplateSelectorDialog().show('sidebar')
}
</script>

<template>
  <div class="flex flex-col gap-2 pointer-events-auto">
    <WorkflowActionsDropdown source="app_mode_toolbar">
      <template #button>
        <Button
          v-tooltip.right="{
            value: t('sideToolbar.labels.menu'),
            ...tooltipOptions
          }"
          variant="secondary"
          size="unset"
          :aria-label="t('sideToolbar.labels.menu')"
          class="h-10 rounded-lg pl-3 pr-2 gap-1 data-[state=open]:bg-secondary-background-hover data-[state=open]:shadow-interface"
        >
          <i class="icon-[lucide--panels-top-left] size-4" />
          <i class="icon-[lucide--chevron-down] size-4 text-muted-foreground" />
        </Button>
      </template>
    </WorkflowActionsDropdown>

    <Button
      v-if="enableAppBuilder"
      v-tooltip.right="{
        value: t('linearMode.appModeToolbar.appBuilder'),
        ...tooltipOptions
      }"
      variant="secondary"
      size="unset"
      :aria-label="t('linearMode.appModeToolbar.appBuilder')"
      class="size-10 rounded-lg"
      @click="enterBuilderMode"
    >
      <i class="icon-[lucide--hammer] size-4" />
    </Button>

    <div
      class="flex flex-col w-10 rounded-lg bg-secondary-background overflow-hidden"
    >
      <Button
        v-tooltip.right="{
          value: t('sideToolbar.mediaAssets.title'),
          ...tooltipOptions
        }"
        variant="textonly"
        size="unset"
        :aria-label="t('sideToolbar.mediaAssets.title')"
        :class="
          cn('size-10', isAssetsActive && 'bg-secondary-background-hover')
        "
        @click="openAssets"
      >
        <i class="icon-[comfy--image-ai-edit] size-4" />
      </Button>
      <Button
        v-tooltip.right="{
          value: t('linearMode.appModeToolbar.apps'),
          ...tooltipOptions
        }"
        variant="textonly"
        size="unset"
        :aria-label="t('linearMode.appModeToolbar.apps')"
        :class="
          cn('size-10', isWorkflowsActive && 'bg-secondary-background-hover')
        "
        @click="showApps"
      >
        <i class="icon-[lucide--panels-top-left] size-4" />
      </Button>
      <Button
        v-tooltip.right="{
          value: t('sideToolbar.templates'),
          ...tooltipOptions
        }"
        variant="textonly"
        size="unset"
        :aria-label="t('sideToolbar.templates')"
        class="size-10"
        @click="openTemplates"
      >
        <i class="icon-[comfy--template] size-4" />
      </Button>
    </div>
  </div>
</template>
