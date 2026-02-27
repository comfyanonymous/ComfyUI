<template>
  <template v-if="nodePack">
    <div
      ref="scrollContainer"
      class="flex h-full flex-col overflow-y-auto scrollbar-custom"
    >
      <PropertiesAccordionItem v-if="!importFailed" :class="accordionClass">
        <template #label>
          <span class="text-xs uppercase font-inter">
            {{ t('manager.actions') }}
          </span>
        </template>
        <div class="flex flex-col gap-1 px-4">
          <template v-if="canTryNightlyUpdate">
            <PackTryUpdateButton :node-pack="nodePack" size="md" />
            <PackUninstallButton :node-packs="[nodePack]" size="md" />
          </template>
          <template v-else-if="isUpdateAvailable">
            <PackUpdateButton :node-packs="[nodePack]" size="md" />
            <PackUninstallButton :node-packs="[nodePack]" size="md" />
          </template>
          <template v-else-if="isAllInstalled">
            <PackUninstallButton :node-packs="[nodePack]" size="md" />
          </template>
          <template v-else>
            <PackInstallButton
              :node-packs="[nodePack]"
              size="md"
              :has-conflict="hasCompatibilityIssues || hasConflictInfo"
              :conflict-info="conflictInfo"
            />
          </template>
        </div>
      </PropertiesAccordionItem>

      <PropertiesAccordionItem :class="accordionClass">
        <template #label>
          <span class="text-xs uppercase font-inter">
            {{ t('manager.basicInfo') }}
          </span>
        </template>
        <ModelInfoField :label="t('g.name')">
          <span class="text-muted-foreground">{{ nodePack.name }}</span>
        </ModelInfoField>
        <ModelInfoField
          v-if="!importFailed && isPackInstalled(nodePack.id)"
          :label="t('manager.filter.enabled')"
        >
          <PackEnableToggle
            :node-pack="nodePack"
            :has-conflict="hasCompatibilityIssues"
          />
        </ModelInfoField>
        <ModelInfoField
          v-for="item in infoItems"
          v-show="item.value !== undefined && item.value !== null"
          :key="item.key"
          :label="item.label"
        >
          <span class="text-muted-foreground">{{ item.value }}</span>
        </ModelInfoField>
        <ModelInfoField :label="t('g.status')">
          <PackStatusMessage
            :status-type="
              nodePack.status as components['schemas']['NodeVersionStatus']
            "
            :has-compatibility-issues="hasCompatibilityIssues"
          />
        </ModelInfoField>
        <ModelInfoField :label="t('manager.version')">
          <PackVersionBadge :node-pack="nodePack" :is-selected="true" />
        </ModelInfoField>
      </PropertiesAccordionItem>

      <PropertiesAccordionItem :class="accordionClass">
        <template #label>
          <span class="text-xs uppercase font-inter">
            {{ t('g.description') }}
          </span>
        </template>
        <DescriptionTabPanel :node-pack="nodePack" />
      </PropertiesAccordionItem>

      <PropertiesAccordionItem
        v-if="hasCompatibilityIssues"
        :class="accordionClass"
      >
        <template #label>
          <span class="text-xs uppercase font-inter">
            ⚠️ {{ importFailed ? t('g.error') : t('g.warning') }}
          </span>
        </template>
        <div class="px-4 py-2">
          <WarningTabPanel :conflict-result="conflictResult" />
        </div>
      </PropertiesAccordionItem>

      <PropertiesAccordionItem :class="accordionClass">
        <template #label>
          <span class="text-xs uppercase font-inter">
            {{ t('g.nodes') }}
          </span>
        </template>
        <div class="px-4 py-2">
          <NodesTabPanel :node-pack="nodePack" :node-names="nodeNames" />
        </div>
      </PropertiesAccordionItem>
    </div>
  </template>
  <template v-else>
    <div class="flex-1 overflow-hidden px-8 pt-4 text-sm">
      {{ $t('manager.infoPanelEmpty') }}
    </div>
  </template>
</template>

<script setup lang="ts">
import { whenever } from '@vueuse/core'
import { computed, provide, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import PropertiesAccordionItem from '@/components/rightSidePanel/layout/PropertiesAccordionItem.vue'
import ModelInfoField from '@/platform/assets/components/modelInfo/ModelInfoField.vue'
import type { components } from '@/types/comfyRegistryTypes'
import { cn } from '@/utils/tailwindUtil'
import PackStatusMessage from '@/workbench/extensions/manager/components/manager/PackStatusMessage.vue'
import PackVersionBadge from '@/workbench/extensions/manager/components/manager/PackVersionBadge.vue'
import PackEnableToggle from '@/workbench/extensions/manager/components/manager/button/PackEnableToggle.vue'
import PackInstallButton from '@/workbench/extensions/manager/components/manager/button/PackInstallButton.vue'
import PackTryUpdateButton from '@/workbench/extensions/manager/components/manager/button/PackTryUpdateButton.vue'
import PackUninstallButton from '@/workbench/extensions/manager/components/manager/button/PackUninstallButton.vue'
import PackUpdateButton from '@/workbench/extensions/manager/components/manager/button/PackUpdateButton.vue'
import DescriptionTabPanel from '@/workbench/extensions/manager/components/manager/infoPanel/tabs/DescriptionTabPanel.vue'
import NodesTabPanel from '@/workbench/extensions/manager/components/manager/infoPanel/tabs/NodesTabPanel.vue'
import WarningTabPanel from '@/workbench/extensions/manager/components/manager/infoPanel/tabs/WarningTabPanel.vue'
import { usePackUpdateStatus } from '@/workbench/extensions/manager/composables/nodePack/usePackUpdateStatus'
import { useConflictDetection } from '@/workbench/extensions/manager/composables/useConflictDetection'
import { useImportFailedDetection } from '@/workbench/extensions/manager/composables/useImportFailedDetection'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'
import { useConflictDetectionStore } from '@/workbench/extensions/manager/stores/conflictDetectionStore'
import { IsInstallingKey } from '@/workbench/extensions/manager/types/comfyManagerTypes'
import type {
  ConflictDetail,
  ConflictDetectionResult
} from '@/workbench/extensions/manager/types/conflictDetectionTypes'
import { ImportFailedKey } from '@/workbench/extensions/manager/types/importFailedTypes'

interface InfoItem {
  key: string
  label: string
  value: string | number | undefined
}

const { nodePack } = defineProps<{
  nodePack: components['schemas']['Node']
}>()

const scrollContainer = ref<HTMLElement | null>(null)

const accordionClass = cn(
  'bg-modal-panel-background border-t border-border-default'
)

const managerStore = useComfyManagerStore()
const { isPackInstalled } = managerStore
const isInstalled = computed(() => isPackInstalled(nodePack.id))
const isInstalling = ref(false)
provide(IsInstallingKey, isInstalling)
whenever(isInstalled, () => {
  isInstalling.value = false
})

const { canTryNightlyUpdate, isUpdateAvailable } = usePackUpdateStatus(
  () => nodePack
)

const isAllInstalled = computed(() => isPackInstalled(nodePack.id))

const { checkNodeCompatibility } = useConflictDetection()

const conflictInfo = computed<ConflictDetail[]>(() => {
  const compatibility = checkNodeCompatibility(nodePack)
  return compatibility.conflicts ?? []
})

const hasConflictInfo = computed(() => conflictInfo.value.length > 0)
const { getConflictsForPackageByID } = useConflictDetectionStore()

const { t, d, n } = useI18n()

// Check compatibility once and pass to children
const conflictResult = computed((): ConflictDetectionResult | null => {
  // For installed packages, use stored conflict data
  if (isInstalled.value && nodePack.id) {
    return getConflictsForPackageByID(nodePack.id) || null
  }

  // For non-installed packages, perform compatibility check
  const compatibility = checkNodeCompatibility(nodePack)

  if (compatibility.hasConflict) {
    return {
      package_id: nodePack.id || '',
      package_name: nodePack.name || '',
      has_conflict: true,
      conflicts: compatibility.conflicts,
      is_compatible: false
    }
  }

  return null
})

const hasCompatibilityIssues = computed(() => {
  return conflictResult.value?.has_conflict
})

const packageId = computed(() => nodePack.id || '')
const { importFailed, showImportFailedDialog } =
  useImportFailedDetection(packageId)

provide(ImportFailedKey, {
  importFailed,
  showImportFailedDialog
})

const nodeNames = computed(() => {
  // @ts-expect-error comfy_nodes is an Algolia-specific field
  const { comfy_nodes } = nodePack
  return comfy_nodes ?? []
})

const infoItems = computed<InfoItem[]>(() => [
  {
    key: 'publisher',
    label: t('manager.createdBy'),
    value: nodePack.publisher?.name ?? nodePack.publisher?.id
  },
  {
    key: 'downloads',
    label: t('manager.downloads'),
    value: nodePack.downloads ? n(nodePack.downloads) : undefined
  },
  {
    key: 'lastUpdated',
    label: t('manager.lastUpdated'),
    value: nodePack.latest_version?.createdAt
      ? d(nodePack.latest_version.createdAt, {
          dateStyle: 'medium'
        })
      : undefined
  }
])

whenever(
  () => nodePack.id,
  (nodePackId, oldNodePackId) => {
    if (nodePackId !== oldNodePackId && scrollContainer.value) {
      scrollContainer.value.scrollTop = 0
    }
  },
  { immediate: true }
)
</script>
