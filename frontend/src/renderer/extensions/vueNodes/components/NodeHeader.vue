<template>
  <div v-if="renderError" class="node-error p-4 text-red-500">
    {{ st('nodeErrors.header', 'Node Header Error') }}
  </div>
  <div
    v-else
    :class="
      cn(
        'lg-node-header text-sm py-2 pl-2 pr-3 w-full min-w-0',
        'text-node-component-header',
        headerShapeClass
      )
    "
    :data-testid="`node-header-${nodeData?.id || ''}`"
    @dblclick="handleDoubleClick"
  >
    <div class="flex items-center justify-between gap-2.5 min-w-0">
      <!-- Collapse/Expand Button -->
      <div class="relative flex items-center gap-2.5 min-w-0 shrink-1 mr-auto">
        <div class="flex shrink-0 items-center px-0.5">
          <Button
            size="icon-sm"
            variant="textonly"
            class="hover:bg-transparent"
            data-testid="node-collapse-button"
            @click.stop="handleCollapse"
            @dblclick.stop
          >
            <i
              :class="
                cn(
                  'icon-[lucide--chevron-down] size-5 transition-transform',
                  collapsed && '-rotate-90'
                )
              "
              class="text-node-component-header-icon"
            />
          </Button>
        </div>
        <!-- Node Title -->
        <div
          v-tooltip.top="tooltipConfig"
          class="flex min-w-0 flex-1 items-center gap-2"
          data-testid="node-title"
        >
          <div class="truncate flex-1">
            <EditableText
              :model-value="displayTitle"
              :is-editing="isEditing"
              :input-attrs="{ 'data-testid': 'node-title-input' }"
              @edit="handleTitleEdit"
              @cancel="handleTitleCancel"
            />
          </div>
        </div>
      </div>

      <template v-for="badge in priceBadges ?? []" :key="badge.required">
        <span
          :class="
            cn(
              'flex h-5 bg-component-node-widget-background p-1 items-center text-xs shrink-0',
              badge.rest ? 'rounded-l-full pr-1' : 'rounded-full'
            )
          "
        >
          <i class="h-full icon-[lucide--component] bg-amber-400" />
          <span class="truncate" v-text="badge.required" />
        </span>
        <span
          v-if="badge.rest"
          class="truncate -ml-2.5 grow-1 basis-0 bg-component-node-widget-background rounded-r-full max-w-max min-w-0"
        >
          <span class="pr-2" v-text="badge.rest" />
        </span>
      </template>
      <NodeBadge v-if="statusBadge" v-bind="statusBadge" />
      <i
        v-if="isPinned"
        class="size-5 icon-[comfy--pin]"
        data-testid="node-pin-indicator"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onErrorCaptured, ref, watch } from 'vue'

import EditableText from '@/components/common/EditableText.vue'
import Button from '@/components/ui/button/Button.vue'
import type { VueNodeData } from '@/composables/graph/useGraphNodeManager'
import { useErrorHandling } from '@/composables/useErrorHandling'
import { st } from '@/i18n'
import { LGraphEventMode, RenderShape } from '@/lib/litegraph/src/litegraph'
import NodeBadge from '@/renderer/extensions/vueNodes/components/NodeBadge.vue'
import { useNodeTooltips } from '@/renderer/extensions/vueNodes/composables/useNodeTooltips'
import { resolveNodeDisplayName } from '@/utils/nodeTitleUtil'
import { cn } from '@/utils/tailwindUtil'

import type { NodeBadgeProps } from './NodeBadge.vue'

interface NodeHeaderProps {
  nodeData?: VueNodeData
  collapsed?: boolean
  priceBadges?: { required: string; rest?: string }[]
}

const { nodeData, collapsed } = defineProps<NodeHeaderProps>()

const emit = defineEmits<{
  collapse: []
  'update:title': [newTitle: string]
}>()

// Error boundary implementation
const renderError = ref<string | null>(null)
const { toastErrorHandler } = useErrorHandling()

onErrorCaptured((error) => {
  renderError.value = error.message
  toastErrorHandler(error)
  return false
})

// Editing state
const isEditing = ref(false)

const { getNodeDescription, createTooltipConfig } = useNodeTooltips(
  nodeData?.type || ''
)

const tooltipConfig = computed(() => {
  if (isEditing.value) {
    return { value: '', disabled: true }
  }
  const description = getNodeDescription.value
  return createTooltipConfig(description)
})

const resolveTitle = (info: VueNodeData | undefined) => {
  const untitledLabel = st('g.untitled', 'Untitled')
  return resolveNodeDisplayName(info ?? null, {
    emptyLabel: untitledLabel,
    untitledLabel,
    st
  })
}

// Local state for title to provide immediate feedback
const displayTitle = ref(resolveTitle(nodeData))

const bypassed = computed(
  (): boolean => nodeData?.mode === LGraphEventMode.BYPASS
)
const muted = computed((): boolean => nodeData?.mode === LGraphEventMode.NEVER)

const statusBadge = computed((): NodeBadgeProps | undefined =>
  muted.value
    ? { text: 'Muted', cssIcon: 'icon-[lucide--ban]' }
    : bypassed.value
      ? { text: 'Bypassed', cssIcon: 'icon-[lucide--redo-dot]' }
      : undefined
)

const isPinned = computed(() => Boolean(nodeData?.flags?.pinned))

const headerShapeClass = computed(() => {
  if (collapsed) {
    switch (nodeData?.shape) {
      case RenderShape.BOX:
        return 'rounded-none'
      case RenderShape.CARD:
        return 'rounded-tl-2xl rounded-br-2xl rounded-tr-none rounded-bl-none'
      default:
        return 'rounded-2xl'
    }
  }
  switch (nodeData?.shape) {
    case RenderShape.BOX:
      return 'rounded-t-none'
    case RenderShape.CARD:
      return 'rounded-tl-2xl rounded-tr-none'
    default:
      return 'rounded-t-2xl'
  }
})

// Watch for external changes to the node title or type
watch(
  () => [nodeData?.title, nodeData?.type] as const,
  () => {
    const next = resolveTitle(nodeData)
    if (next !== displayTitle.value) {
      displayTitle.value = next
    }
  }
)

// Event handlers
const handleCollapse = () => {
  emit('collapse')
}

const handleDoubleClick = () => {
  isEditing.value = true
}

const handleTitleEdit = (newTitle: string) => {
  isEditing.value = false
  const trimmedTitle = newTitle.trim()
  if (trimmedTitle && trimmedTitle !== displayTitle.value) {
    // Emit for litegraph sync
    emit('update:title', trimmedTitle)
  }
}

const handleTitleCancel = () => {
  isEditing.value = false
}
</script>
