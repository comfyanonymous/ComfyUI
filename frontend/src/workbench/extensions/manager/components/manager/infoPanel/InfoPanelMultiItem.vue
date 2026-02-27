<template>
  <div
    v-if="nodePacks?.length"
    class="flex h-full flex-col overflow-y-auto scrollbar-custom"
  >
    <PropertiesAccordionItem :class="accordionClass">
      <template #label>
        <span class="text-xs uppercase font-inter">
          {{ t('manager.actions') }}
        </span>
      </template>
      <div class="flex flex-col gap-1 px-4">
        <!-- Mixed: Don't show any button -->
        <div v-if="isMixed" class="text-sm text-neutral-500">
          {{ $t('manager.mixedSelectionMessage') }}
        </div>
        <!-- All installed: Show update (if nightly) and uninstall buttons -->
        <template v-else-if="isAllInstalled">
          <Button
            v-if="hasNightlyPacks"
            v-tooltip.top="$t('manager.tryUpdateTooltip')"
            variant="textonly"
            size="md"
            :disabled="isUpdatingSelected"
            @click="updateSelectedNightlyPacks"
          >
            <DotSpinner v-if="isUpdatingSelected" duration="1s" :size="16" />
            <span>{{ updateSelectedLabel }}</span>
          </Button>
          <PackUninstallButton size="md" :node-packs="installedPacks" />
        </template>
        <!-- None installed: Show install button -->
        <PackInstallButton
          v-else-if="isNoneInstalled"
          size="md"
          :node-packs="notInstalledPacks"
          :has-conflict="hasConflicts"
          :conflict-info="conflictInfo"
        />
      </div>
    </PropertiesAccordionItem>

    <PropertiesAccordionItem :class="accordionClass">
      <template #label>
        <span class="text-xs uppercase font-inter">
          {{ t('manager.basicInfo') }}
        </span>
      </template>
      <ModelInfoField :label="t('manager.selected')">
        <span>
          <span class="font-bold text-blue-500">{{ nodePacks.length }}</span>
          {{ t('manager.packsSelected') }}
        </span>
      </ModelInfoField>
      <ModelInfoField :label="t('g.status')">
        <PackStatusMessage
          :status-type="overallStatus"
          :has-compatibility-issues="hasConflicts"
        />
      </ModelInfoField>
      <ModelInfoField :label="t('manager.totalNodes')">
        <span class="text-muted-foreground">{{ totalNodesCount }}</span>
      </ModelInfoField>
    </PropertiesAccordionItem>
  </div>
  <div v-else class="mx-8 mt-4 flex-1 overflow-hidden text-sm">
    {{ $t('manager.infoPanelEmpty') }}
  </div>
</template>

<script setup lang="ts">
import { useAsyncState } from '@vueuse/core'
import { computed, onUnmounted, provide, ref, toRef } from 'vue'
import { useI18n } from 'vue-i18n'

import DotSpinner from '@/components/common/DotSpinner.vue'
import PropertiesAccordionItem from '@/components/rightSidePanel/layout/PropertiesAccordionItem.vue'
import Button from '@/components/ui/button/Button.vue'
import ModelInfoField from '@/platform/assets/components/modelInfo/ModelInfoField.vue'
import { useComfyRegistryStore } from '@/stores/comfyRegistryStore'
import type { components } from '@/types/comfyRegistryTypes'
import { cn } from '@/utils/tailwindUtil'
import PackStatusMessage from '@/workbench/extensions/manager/components/manager/PackStatusMessage.vue'
import PackInstallButton from '@/workbench/extensions/manager/components/manager/button/PackInstallButton.vue'
import PackUninstallButton from '@/workbench/extensions/manager/components/manager/button/PackUninstallButton.vue'
import { usePacksSelection } from '@/workbench/extensions/manager/composables/nodePack/usePacksSelection'
import { usePacksStatus } from '@/workbench/extensions/manager/composables/nodePack/usePacksStatus'
import { useConflictDetection } from '@/workbench/extensions/manager/composables/useConflictDetection'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'
import type { ConflictDetail } from '@/workbench/extensions/manager/types/conflictDetectionTypes'
import { ImportFailedKey } from '@/workbench/extensions/manager/types/importFailedTypes'

const { nodePacks } = defineProps<{
  nodePacks: components['schemas']['Node'][]
}>()

const { t } = useI18n()

const accordionClass = cn(
  'bg-modal-panel-background border-t border-border-default'
)

const managerStore = useComfyManagerStore()
const nodePacksRef = toRef(() => nodePacks)

// Use new composables for cleaner code
const {
  installedPacks,
  notInstalledPacks,
  isAllInstalled,
  isNoneInstalled,
  isMixed,
  nightlyPacks,
  hasNightlyPacks
} = usePacksSelection(nodePacksRef)

const { hasImportFailed, overallStatus } = usePacksStatus(nodePacksRef)

// Batch update state for nightly packs
const isUpdatingSelected = ref(false)

async function updateSelectedNightlyPacks() {
  if (nightlyPacks.value.length === 0) return

  isUpdatingSelected.value = true
  try {
    for (const pack of nightlyPacks.value) {
      if (!pack.id) continue
      await managerStore.updatePack.call({
        id: pack.id,
        version: 'nightly'
      })
    }
    managerStore.updatePack.clear()
  } catch (error) {
    console.error('Batch nightly update failed:', error)
  } finally {
    isUpdatingSelected.value = false
  }
}

const updateSelectedLabel = computed(() =>
  isUpdatingSelected.value ? t('g.updating') : t('manager.updateSelected')
)

const { checkNodeCompatibility } = useConflictDetection()
const { getNodeDefs } = useComfyRegistryStore()

// Provide import failed context for PackStatusMessage
provide(ImportFailedKey, {
  importFailed: hasImportFailed,
  showImportFailedDialog: () => {} // No-op for multi-selection
})

// Check for conflicts in not-installed packages - keep original logic but simplified
const packageConflicts = computed(() => {
  const conflictsByPackage = new Map<string, ConflictDetail[]>()

  for (const pack of notInstalledPacks.value) {
    const compatibilityCheck = checkNodeCompatibility(pack)
    if (compatibilityCheck.hasConflict && pack.id) {
      conflictsByPackage.set(pack.id, compatibilityCheck.conflicts)
    }
  }

  return conflictsByPackage
})

// Aggregate all unique conflicts for display
const conflictInfo = computed<ConflictDetail[]>(() => {
  const conflictMap = new Map<string, ConflictDetail>()

  packageConflicts.value.forEach((conflicts) => {
    conflicts.forEach((conflict) => {
      const key = `${conflict.type}-${conflict.current_value}-${conflict.required_value}`
      if (!conflictMap.has(key)) {
        conflictMap.set(key, conflict)
      }
    })
  })

  return Array.from(conflictMap.values())
})

const hasConflicts = computed(() => conflictInfo.value.length > 0)

const getPackNodes = async (pack: components['schemas']['Node']) => {
  if (!pack.latest_version?.version) return []
  const nodeDefs = await getNodeDefs.call({
    packId: pack.id,
    version: pack.latest_version?.version,
    // Fetch all nodes.
    // TODO: Render all nodes previews and handle pagination.
    // For determining length, use the `totalNumberOfPages` field of response
    limit: 8192
  })
  return nodeDefs?.comfy_nodes ?? []
}

const { state: allNodeDefs } = useAsyncState(
  () => Promise.all(nodePacks.map(getPackNodes)),
  [],
  {
    immediate: true
  }
)

const totalNodesCount = computed(() =>
  allNodeDefs.value.reduce(
    (total, nodeDefs) => total + (nodeDefs?.length || 0),
    0
  )
)

onUnmounted(() => {
  getNodeDefs.cancel()
})
</script>
