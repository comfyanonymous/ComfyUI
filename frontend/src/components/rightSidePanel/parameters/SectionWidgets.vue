<script setup lang="ts">
import { computed, inject, provide, ref, shallowRef, watchEffect } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { isPromotedWidgetView } from '@/core/graph/subgraph/promotedWidgetTypes'
import type { LGraphGroup, LGraphNode } from '@/lib/litegraph/src/litegraph'
import { SubgraphNode } from '@/lib/litegraph/src/litegraph'
import { usePromotionStore } from '@/stores/promotionStore'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useExecutionErrorStore } from '@/stores/executionErrorStore'
import { useRightSidePanelStore } from '@/stores/workspace/rightSidePanelStore'
import { useSettingStore } from '@/platform/settings/settingStore'
import { cn } from '@/utils/tailwindUtil'
import { isGroupNode } from '@/utils/executableGroupNodeDto'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import { getWidgetDefaultValue } from '@/utils/widgetUtil'
import type { WidgetValue } from '@/utils/widgetUtil'

import PropertiesAccordionItem from '../layout/PropertiesAccordionItem.vue'
import { HideLayoutFieldKey } from '@/types/widgetTypes'

import { GetNodeParentGroupKey } from '../shared'
import WidgetItem from './WidgetItem.vue'

const {
  label,
  node,
  widgets: widgetsProp,
  showLocateButton = false,
  isDraggable = false,
  hiddenFavoriteIndicator = false,
  showNodeName = false,
  parents = [],
  enableEmptyState = false,
  tooltip
} = defineProps<{
  label?: string
  parents?: SubgraphNode[]
  node?: LGraphNode
  widgets: { widget: IBaseWidget; node: LGraphNode }[]
  showLocateButton?: boolean
  isDraggable?: boolean
  hiddenFavoriteIndicator?: boolean
  showNodeName?: boolean
  /**
   * Whether to show the empty state slot when there are no widgets.
   */
  enableEmptyState?: boolean
  tooltip?: string
}>()

const collapse = defineModel<boolean>('collapse', { default: false })

const widgetsContainer = ref<HTMLElement>()
const rootElement = ref<HTMLElement>()

const widgets = shallowRef(widgetsProp)
watchEffect(() => (widgets.value = widgetsProp))

provide(HideLayoutFieldKey, true)

const canvasStore = useCanvasStore()
const executionErrorStore = useExecutionErrorStore()
const rightSidePanelStore = useRightSidePanelStore()
const nodeDefStore = useNodeDefStore()
const { t } = useI18n()

const getNodeParentGroup = inject(GetNodeParentGroupKey, null)

const promotionStore = usePromotionStore()

function isWidgetShownOnParents(
  widgetNode: LGraphNode,
  widget: IBaseWidget
): boolean {
  return parents.some((parent) => {
    if (isPromotedWidgetView(widget)) {
      return promotionStore.isPromoted(
        parent.rootGraph.id,
        parent.id,
        widget.sourceNodeId,
        widget.sourceWidgetName
      )
    }
    return promotionStore.isPromoted(
      parent.rootGraph.id,
      parent.id,
      String(widgetNode.id),
      widget.name
    )
  })
}

const isEmpty = computed(() => widgets.value.length === 0)

const displayLabel = computed(
  () => label ?? (node ? node.title : t('rightSidePanel.inputs'))
)

const targetNode = computed<LGraphNode | null>(() => {
  if (node) return node
  if (isEmpty.value) return null

  const firstNodeId = widgets.value[0].node.id
  const allSameNode = widgets.value.every(({ node }) => node.id === firstNodeId)

  return allSameNode ? widgets.value[0].node : null
})

const hasDirectError = computed(() => {
  if (!targetNode.value) return false
  return executionErrorStore.activeGraphErrorNodeIds.has(
    String(targetNode.value.id)
  )
})

const hasContainerInternalError = computed(() => {
  if (!targetNode.value) return false
  const isContainer =
    targetNode.value instanceof SubgraphNode || isGroupNode(targetNode.value)
  if (!isContainer) return false

  return executionErrorStore.isContainerWithInternalError(targetNode.value)
})

const nodeHasError = computed(() => {
  if (!targetNode.value) return false
  if (canvasStore.selectedItems.length === 1) return false
  return hasDirectError.value || hasContainerInternalError.value
})

const showSeeError = computed(
  () =>
    nodeHasError.value &&
    useSettingStore().get('Comfy.RightSidePanel.ShowErrorsTab')
)

const parentGroup = computed<LGraphGroup | null>(() => {
  if (!targetNode.value || !getNodeParentGroup) return null
  return getNodeParentGroup(targetNode.value)
})

const canShowLocateButton = computed(
  () => showLocateButton && targetNode.value !== null
)

function handleLocateNode() {
  if (!targetNode.value || !canvasStore.canvas) return

  const graphNode = canvasStore.canvas.graph?.getNodeById(targetNode.value.id)
  if (graphNode) {
    canvasStore.canvas.animateToBounds(graphNode.boundingRect)
  }
}

function navigateToErrorTab() {
  if (!targetNode.value) return
  if (!useSettingStore().get('Comfy.RightSidePanel.ShowErrorsTab')) return
  rightSidePanelStore.focusedErrorNodeId = String(targetNode.value.id)
  rightSidePanelStore.openPanel('errors')
}

function writeWidgetValue(widget: IBaseWidget, value: WidgetValue) {
  widget.value = value
  widget.callback?.(value)
  canvasStore.canvas?.setDirty(true, true)
}

function handleResetAllWidgets() {
  for (const { widget, node: widgetNode } of widgetsProp) {
    const spec = nodeDefStore.getInputSpecForWidget(widgetNode, widget.name)
    const defaultValue = getWidgetDefaultValue(spec)
    if (defaultValue !== undefined) {
      writeWidgetValue(widget, defaultValue)
    }
  }
}

function handleWidgetValueUpdate(widget: IBaseWidget, newValue: WidgetValue) {
  if (newValue === undefined) return
  writeWidgetValue(widget, newValue)
}

function handleWidgetReset(widget: IBaseWidget, newValue: WidgetValue) {
  writeWidgetValue(widget, newValue)
}

defineExpose({
  widgetsContainer,
  rootElement
})
</script>

<template>
  <div ref="rootElement">
    <PropertiesAccordionItem
      v-model:collapse="collapse"
      :enable-empty-state
      :disabled="isEmpty"
      :tooltip
      :size="showSeeError ? 'lg' : 'default'"
    >
      <template #label>
        <div class="flex flex-wrap items-center gap-2 flex-1 min-w-0">
          <span class="flex-1 flex items-center gap-2 min-w-0">
            <i
              v-if="nodeHasError"
              class="icon-[lucide--octagon-alert] size-4 shrink-0 text-destructive-background-hover"
            />
            <span
              :class="
                cn(
                  'truncate',
                  nodeHasError && 'text-destructive-background-hover'
                )
              "
            >
              <slot name="label">
                {{ displayLabel }}
              </slot>
            </span>
            <span
              v-if="parentGroup"
              class="text-xs text-muted-foreground truncate flex-1 text-right min-w-11"
              :title="parentGroup.title"
            >
              {{ parentGroup.title }}
            </span>
          </span>
          <Button
            v-if="showSeeError"
            variant="secondary"
            size="sm"
            class="shrink-0 rounded-lg text-sm h-8"
            @click.stop="navigateToErrorTab"
          >
            {{ t('rightSidePanel.seeError') }}
          </Button>
          <Button
            v-if="!isEmpty"
            variant="muted-textonly"
            size="icon-sm"
            class="subbutton shrink-0 size-8 hover:text-base-foreground"
            :title="t('rightSidePanel.resetAllParameters')"
            :aria-label="t('rightSidePanel.resetAllParameters')"
            @click.stop="handleResetAllWidgets"
          >
            <i class="icon-[lucide--rotate-ccw] size-4" />
          </Button>
          <Button
            v-if="canShowLocateButton"
            variant="muted-textonly"
            size="icon-sm"
            class="subbutton shrink-0 mr-3 size-8 hover:text-base-foreground"
            :title="t('rightSidePanel.locateNode')"
            :aria-label="t('rightSidePanel.locateNode')"
            @click.stop="handleLocateNode"
          >
            <i class="icon-[lucide--locate] size-4" />
          </Button>
        </div>
      </template>

      <template #empty><slot name="empty" /></template>

      <div
        ref="widgetsContainer"
        class="space-y-2 rounded-lg px-4 pt-1 relative"
      >
        <TransitionGroup name="list-scale">
          <WidgetItem
            v-for="{ widget, node } in widgets"
            :key="`${node.id}-${widget.name}-${widget.type}`"
            :widget="widget"
            :node="node"
            :is-draggable="isDraggable"
            :hidden-favorite-indicator="hiddenFavoriteIndicator"
            :show-node-name="showNodeName"
            :parents="parents"
            :is-shown-on-parents="isWidgetShownOnParents(node, widget)"
            @update:widget-value="handleWidgetValueUpdate(widget, $event)"
            @reset-to-default="handleWidgetReset(widget, $event)"
          />
        </TransitionGroup>
      </div>
    </PropertiesAccordionItem>
  </div>
</template>
