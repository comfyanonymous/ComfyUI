<template>
  <div class="flex flex-col w-full mb-2">
    <!-- Pack header row: pack name + info + chevron -->
    <div class="flex h-8 items-center w-full">
      <!-- Warning icon for unknown packs -->
      <i
        v-if="group.packId === null && !group.isResolving"
        class="icon-[lucide--triangle-alert] size-4 text-warning-background shrink-0 mr-1.5"
      />
      <p
        class="flex-1 min-w-0 text-sm font-medium overflow-hidden text-ellipsis whitespace-nowrap"
        :class="
          group.packId === null && !group.isResolving
            ? 'text-warning-background'
            : 'text-foreground'
        "
      >
        <span v-if="group.isResolving" class="text-muted-foreground italic">
          {{ t('g.loading') }}...
        </span>
        <span v-else>
          {{
            `${group.packId ?? t('rightSidePanel.missingNodePacks.unknownPack')} (${group.nodeTypes.length})`
          }}
        </span>
      </p>
      <Button
        v-if="showInfoButton && group.packId !== null"
        variant="textonly"
        size="icon-sm"
        class="size-8 text-muted-foreground hover:text-base-foreground shrink-0"
        :aria-label="t('rightSidePanel.missingNodePacks.viewInManager')"
        @click="emit('openManagerInfo', group.packId ?? '')"
      >
        <i class="icon-[lucide--info] size-4" />
      </Button>
      <Button
        variant="textonly"
        size="icon-sm"
        :class="
          cn(
            'size-8 shrink-0 transition-transform duration-200 hover:bg-transparent',
            { 'rotate-180': expanded }
          )
        "
        :aria-label="
          expanded
            ? t('rightSidePanel.missingNodePacks.collapse')
            : t('rightSidePanel.missingNodePacks.expand')
        "
        @click="toggleExpand"
      >
        <i
          class="icon-[lucide--chevron-down] size-4 text-muted-foreground group-hover:text-base-foreground"
        />
      </Button>
    </div>

    <!-- Sub-labels: individual node instances, each with their own Locate button -->
    <TransitionCollapse>
      <div
        v-if="expanded"
        class="flex flex-col gap-0.5 pl-2 mb-1 overflow-hidden"
      >
        <div
          v-for="nodeType in group.nodeTypes"
          :key="getKey(nodeType)"
          class="flex h-7 items-center"
        >
          <span
            v-if="
              showNodeIdBadge &&
              typeof nodeType !== 'string' &&
              nodeType.nodeId != null
            "
            class="shrink-0 rounded-md bg-secondary-background-selected px-2 py-0.5 text-xs font-mono text-muted-foreground font-bold mr-1"
          >
            #{{ nodeType.nodeId }}
          </span>
          <p class="flex-1 min-w-0 text-xs text-muted-foreground truncate">
            {{ getLabel(nodeType) }}
          </p>
          <Button
            v-if="typeof nodeType !== 'string' && nodeType.nodeId != null"
            variant="textonly"
            size="icon-sm"
            class="size-6 text-muted-foreground hover:text-base-foreground shrink-0 mr-1"
            :aria-label="t('rightSidePanel.locateNode')"
            @click="handleLocateNode(nodeType)"
          >
            <i class="icon-[lucide--locate] size-3" />
          </Button>
        </div>
      </div>
    </TransitionCollapse>

    <!-- Install button: shown when manager enabled, registry knows the pack or it's already installed -->
    <div
      v-if="
        shouldShowManagerButtons &&
        group.packId !== null &&
        (nodePack || comfyManagerStore.isPackInstalled(group.packId))
      "
      class="flex items-start w-full pt-1 pb-1"
    >
      <Button
        variant="secondary"
        size="md"
        class="flex flex-1 w-full"
        :disabled="
          comfyManagerStore.isPackInstalled(group.packId) || isInstalling
        "
        @click="handlePackInstallClick"
      >
        <DotSpinner
          v-if="isInstalling"
          duration="1s"
          :size="12"
          class="mr-1.5 shrink-0"
        />
        <i
          v-else-if="comfyManagerStore.isPackInstalled(group.packId)"
          class="icon-[lucide--check] size-4 text-foreground shrink-0 mr-1"
        />
        <i
          v-else
          class="icon-[lucide--download] size-4 text-foreground shrink-0 mr-1"
        />
        <span class="text-sm text-foreground truncate min-w-0">
          {{
            isInstalling
              ? t('rightSidePanel.missingNodePacks.installing')
              : comfyManagerStore.isPackInstalled(group.packId)
                ? t('rightSidePanel.missingNodePacks.installed')
                : t('rightSidePanel.missingNodePacks.installNodePack')
          }}
        </span>
      </Button>
    </div>

    <!-- Registry still loading: packId known but result not yet available -->
    <div
      v-else-if="group.packId !== null && shouldShowManagerButtons && isLoading"
      class="flex items-start w-full pt-1 pb-1"
    >
      <div
        class="flex flex-1 h-8 items-center justify-center overflow-hidden p-2 rounded-lg min-w-0 bg-secondary-background opacity-60 cursor-not-allowed select-none"
      >
        <DotSpinner duration="1s" :size="12" class="mr-1.5 shrink-0" />
        <span class="text-sm text-foreground truncate min-w-0">
          {{ t('g.loading') }}
        </span>
      </div>
    </div>

    <!-- Search in Manager: fetch done but pack not found in registry -->
    <div
      v-else-if="group.packId !== null && shouldShowManagerButtons"
      class="flex items-start w-full pt-1 pb-1"
    >
      <Button
        variant="secondary"
        size="md"
        class="flex flex-1 w-full"
        @click="
          openManager({
            initialTab: ManagerTab.All,
            initialPackId: group.packId!
          })
        "
      >
        <i class="icon-[lucide--search] size-4 text-foreground shrink-0 mr-1" />
        <span class="text-sm text-foreground truncate min-w-0">
          {{ t('rightSidePanel.missingNodePacks.searchInManager') }}
        </span>
      </Button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { cn } from '@/utils/tailwindUtil'
import { useI18n } from 'vue-i18n'
import Button from '@/components/ui/button/Button.vue'
import DotSpinner from '@/components/common/DotSpinner.vue'
import TransitionCollapse from '@/components/rightSidePanel/layout/TransitionCollapse.vue'
import { useMissingNodes } from '@/workbench/extensions/manager/composables/nodePack/useMissingNodes'
import { usePackInstall } from '@/workbench/extensions/manager/composables/nodePack/usePackInstall'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'
import { useManagerState } from '@/workbench/extensions/manager/composables/useManagerState'
import { ManagerTab } from '@/workbench/extensions/manager/types/comfyManagerTypes'
import type { MissingNodeType } from '@/types/comfy'
import type { MissingPackGroup } from '@/components/rightSidePanel/errors/useErrorGroups'

const props = defineProps<{
  group: MissingPackGroup
  showInfoButton: boolean
  showNodeIdBadge: boolean
}>()

const emit = defineEmits<{
  locateNode: [nodeId: string]
  openManagerInfo: [packId: string]
}>()

const { t } = useI18n()

const { missingNodePacks, isLoading } = useMissingNodes()
const comfyManagerStore = useComfyManagerStore()
const { shouldShowManagerButtons, openManager } = useManagerState()

const nodePack = computed(() => {
  if (!props.group.packId) return null
  return missingNodePacks.value.find((p) => p.id === props.group.packId) ?? null
})

const { isInstalling, installAllPacks } = usePackInstall(() =>
  nodePack.value ? [nodePack.value] : []
)

function handlePackInstallClick() {
  if (!props.group.packId) return
  if (!comfyManagerStore.isPackInstalled(props.group.packId)) {
    void installAllPacks()
  }
}

const expanded = ref(false)

function toggleExpand() {
  expanded.value = !expanded.value
}

function getKey(nodeType: MissingNodeType): string {
  if (typeof nodeType === 'string') return nodeType
  return nodeType.nodeId != null ? String(nodeType.nodeId) : nodeType.type
}

function getLabel(nodeType: MissingNodeType): string {
  return typeof nodeType === 'string' ? nodeType : nodeType.type
}

function handleLocateNode(nodeType: MissingNodeType) {
  if (typeof nodeType === 'string') return
  if (nodeType.nodeId != null) {
    emit('locateNode', String(nodeType.nodeId))
  }
}
</script>
