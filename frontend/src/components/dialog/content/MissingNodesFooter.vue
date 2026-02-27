<template>
  <div class="flex w-full flex-col gap-2 py-2 px-4">
    <div class="flex flex-col gap-1 text-sm text-muted-foreground">
      <div class="flex items-center gap-1">
        <input
          id="doNotAskAgainNodes"
          v-model="doNotAskAgain"
          type="checkbox"
          class="h-4 w-4 cursor-pointer"
        />
        <label for="doNotAskAgainNodes">{{
          $t('missingModelsDialog.doNotAskAgain')
        }}</label>
      </div>
      <i18n-t
        v-if="doNotAskAgain"
        keypath="missingModelsDialog.reEnableInSettings"
        tag="span"
        class="text-sm text-muted-foreground ml-6"
      >
        <template #link>
          <Button
            variant="textonly"
            class="underline cursor-pointer p-0 text-sm text-muted-foreground hover:bg-transparent"
            @click="openShowMissingNodesSetting"
          >
            {{ $t('missingModelsDialog.reEnableInSettingsLink') }}
          </Button>
        </template>
      </i18n-t>
    </div>

    <!-- All nodes replaceable: Skip button (cloud + OSS) -->
    <div v-if="!hasNonReplaceableNodes" class="flex justify-end gap-1">
      <Button variant="secondary" size="md" @click="handleGotItClick">
        {{ $t('nodeReplacement.skipForNow') }}
      </Button>
    </div>

    <!-- Cloud mode: Learn More + Got It buttons -->
    <div
      v-else-if="isCloud"
      class="flex w-full items-center justify-between gap-2"
    >
      <Button
        variant="textonly"
        size="sm"
        as="a"
        href="https://www.comfy.org/cloud"
        target="_blank"
        rel="noopener noreferrer"
      >
        <i class="icon-[lucide--info]"></i>
        <span>{{ $t('missingNodes.cloud.learnMore') }}</span>
      </Button>
      <Button variant="secondary" size="md" @click="handleGotItClick">{{
        $t('missingNodes.cloud.gotIt')
      }}</Button>
    </div>

    <!-- OSS mode: Manager buttons -->
    <div v-else-if="showManagerButtons" class="flex justify-end gap-1">
      <Button variant="textonly" @click="handleOpenManager">{{
        $t('g.openManager')
      }}</Button>
      <PackInstallButton
        v-if="showInstallAllButton"
        type="secondary"
        size="md"
        :disabled="
          isLoading || !!error || missingNodePacks.length === 0 || isInstalling
        "
        :is-loading="isLoading"
        :node-packs="missingNodePacks"
        :label="
          isLoading
            ? $t('manager.gettingInfo')
            : $t('manager.installAllMissingNodes')
        "
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { isCloud } from '@/platform/distribution/types'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { useSettingsDialog } from '@/platform/settings/composables/useSettingsDialog'
import { useDialogStore } from '@/stores/dialogStore'
import type { MissingNodeType } from '@/types/comfy'
import PackInstallButton from '@/workbench/extensions/manager/components/manager/button/PackInstallButton.vue'
import { useMissingNodes } from '@/workbench/extensions/manager/composables/nodePack/useMissingNodes'
import { useManagerState } from '@/workbench/extensions/manager/composables/useManagerState'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'
import { ManagerTab } from '@/workbench/extensions/manager/types/comfyManagerTypes'

const { missingNodeTypes } = defineProps<{
  missingNodeTypes?: MissingNodeType[]
}>()

const dialogStore = useDialogStore()
const { t } = useI18n()

const doNotAskAgain = ref(false)

watch(doNotAskAgain, (value) => {
  void useSettingStore().set('Comfy.Workflow.ShowMissingNodesWarning', !value)
})

const handleGotItClick = () => {
  dialogStore.closeDialog({ key: 'global-missing-nodes' })
}

function openShowMissingNodesSetting() {
  dialogStore.closeDialog({ key: 'global-missing-nodes' })
  useSettingsDialog().show(undefined, 'Comfy.Workflow.ShowMissingNodesWarning')
}

const { missingNodePacks, isLoading, error } = useMissingNodes()
const comfyManagerStore = useComfyManagerStore()
const managerState = useManagerState()
function handleOpenManager() {
  managerState.openManager({
    initialTab: ManagerTab.Missing,
    showToastOnLegacyError: true
  })
}

// Check if any of the missing packs are currently being installed
const isInstalling = computed(() => {
  if (!missingNodePacks.value?.length) return false
  return missingNodePacks.value.some((pack) =>
    comfyManagerStore.isPackInstalling(pack.id)
  )
})

// Show manager buttons unless manager is disabled
const showManagerButtons = computed(() => {
  return managerState.shouldShowManagerButtons.value
})

// Only show Install All button for NEW_UI (new manager with v4 support)
const showInstallAllButton = computed(() => {
  return managerState.shouldShowInstallButton.value
})

const hasNonReplaceableNodes = computed(
  () =>
    missingNodeTypes?.some(
      (n) =>
        typeof n === 'string' || (typeof n === 'object' && !n.isReplaceable)
    ) ?? false
)

// Track whether missingNodePacks was ever non-empty (i.e. there were packs to install)
const hadMissingPacks = ref(false)

watch(
  missingNodePacks,
  (packs) => {
    if (packs && packs.length > 0) hadMissingPacks.value = true
  },
  { immediate: true }
)

// Only consider "all installed" when packs transitioned from non-empty to empty
// (actual installation happened). Replaceable-only case is handled by Content auto-close.
const allMissingNodesInstalled = computed(() => {
  if (!hadMissingPacks.value) return false
  return (
    !isLoading.value &&
    !isInstalling.value &&
    missingNodePacks.value?.length === 0
  )
})

// Watch for completion and close dialog (OSS mode only)
watch(allMissingNodesInstalled, async (allInstalled) => {
  if (!isCloud && allInstalled && showInstallAllButton.value) {
    // Use nextTick to ensure state updates are complete
    await nextTick()

    dialogStore.closeDialog({ key: 'global-missing-nodes' })

    // Show success toast
    useToastStore().add({
      severity: 'success',
      summary: t('g.success'),
      detail: t('manager.allMissingNodesInstalled'),
      life: 3000
    })
  }
})
</script>
