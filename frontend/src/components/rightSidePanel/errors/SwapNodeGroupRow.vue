<template>
  <div class="flex flex-col w-full mb-4">
    <!-- Type header row: type name + chevron -->
    <div class="flex h-8 items-center w-full">
      <p
        class="flex-1 min-w-0 text-sm font-medium overflow-hidden text-ellipsis whitespace-nowrap text-foreground"
      >
        {{ `${group.type} (${group.nodeTypes.length})` }}
      </p>

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
            ? t('rightSidePanel.missingNodePacks.collapse', 'Collapse')
            : t('rightSidePanel.missingNodePacks.expand', 'Expand')
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
        class="flex flex-col gap-0.5 pl-2 mb-2 overflow-hidden"
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
            :aria-label="t('rightSidePanel.locateNode', 'Locate Node')"
            @click="handleLocateNode(nodeType)"
          >
            <i class="icon-[lucide--locate] size-3" />
          </Button>
        </div>
      </div>
    </TransitionCollapse>

    <!-- Description rows: what it is replaced by -->
    <div class="flex flex-col text-[13px] mb-2 mt-1 px-1 gap-0.5">
      <span class="text-muted-foreground">{{
        t('nodeReplacement.willBeReplacedBy', 'This node will be replaced by:')
      }}</span>
      <span class="font-bold text-foreground">{{
        group.newNodeId ?? t('nodeReplacement.unknownNode', 'Unknown')
      }}</span>
    </div>

    <!-- Replace Action Button -->
    <div class="flex items-start w-full pt-1 pb-1">
      <Button
        variant="secondary"
        size="md"
        class="flex flex-1 w-full"
        @click="handleReplaceNode"
      >
        <i class="icon-[lucide--repeat] size-4 text-foreground shrink-0 mr-1" />
        <span class="text-sm text-foreground truncate min-w-0">
          {{ t('nodeReplacement.replaceNode', 'Replace Node') }}
        </span>
      </Button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { cn } from '@/utils/tailwindUtil'
import { useI18n } from 'vue-i18n'
import Button from '@/components/ui/button/Button.vue'
import TransitionCollapse from '@/components/rightSidePanel/layout/TransitionCollapse.vue'
import type { MissingNodeType } from '@/types/comfy'
import type { SwapNodeGroup } from './useErrorGroups'
import { useNodeReplacement } from '@/platform/nodeReplacement/useNodeReplacement'
import { useExecutionErrorStore } from '@/stores/executionErrorStore'

const props = defineProps<{
  group: SwapNodeGroup
  showNodeIdBadge: boolean
}>()

const emit = defineEmits<{
  'locate-node': [nodeId: string]
}>()

const { t } = useI18n()
const { replaceNodesInPlace } = useNodeReplacement()
const executionErrorStore = useExecutionErrorStore()

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
    emit('locate-node', String(nodeType.nodeId))
  }
}

function handleReplaceNode() {
  const replaced = replaceNodesInPlace(props.group.nodeTypes)
  if (replaced.length > 0) {
    executionErrorStore.removeMissingNodesByType([props.group.type])
  }
}
</script>
