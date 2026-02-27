<template>
  <div
    data-testid="subgraph-breadcrumb"
    class="subgraph-breadcrumb flex w-auto drop-shadow-(--interface-panel-drop-shadow) items-center"
    :class="{
      'subgraph-breadcrumb-collapse': collapseTabs,
      'subgraph-breadcrumb-overflow': overflowingTabs
    }"
    :style="{
      '--p-breadcrumb-gap': `0px`,
      '--p-breadcrumb-item-margin': `${ITEM_GAP / 2}px`,
      '--p-breadcrumb-item-min-width': `${MIN_WIDTH}px`,
      '--p-breadcrumb-item-padding': `${ITEM_PADDING}px`,
      '--p-breadcrumb-icon-width': `${ICON_WIDTH}px`
    }"
  >
    <WorkflowActionsDropdown source="breadcrumb_subgraph_menu_selected" />
    <Button
      v-if="isInSubgraph"
      class="back-button pointer-events-auto h-8 w-8 shrink-0 border border-transparent bg-transparent p-0 ml-1.5 transition-all hover:rounded-lg hover:border-interface-stroke hover:bg-comfy-menu-bg"
      text
      severity="secondary"
      size="small"
      @click="handleBackClick"
    >
      <i class="icon-[lucide--undo-2]" />
    </Button>
    <Breadcrumb
      ref="breadcrumbRef"
      class="w-fit rounded-lg p-0"
      :class="{ hidden: !isInSubgraph }"
      :model="items"
      :pt="{ item: { class: 'pointer-events-auto' } }"
      :aria-label="$t('g.graphNavigation')"
    >
      <template #item="{ item }">
        <SubgraphBreadcrumbItem
          :item="item"
          :is-active="item.key === activeItemKey"
        />
      </template>
      <template #separator
        ><span style="transform: scale(1.5)"> / </span></template
      >
    </Breadcrumb>
  </div>
</template>

<script setup lang="ts">
import Breadcrumb from 'primevue/breadcrumb'
import Button from 'primevue/button'
import type { MenuItem } from 'primevue/menuitem'
import { computed, onUpdated, ref, watch } from 'vue'

import SubgraphBreadcrumbItem from '@/components/breadcrumb/SubgraphBreadcrumbItem.vue'
import WorkflowActionsDropdown from '@/components/common/WorkflowActionsDropdown.vue'
import { useOverflowObserver } from '@/composables/element/useOverflowObserver'
import { useTelemetry } from '@/platform/telemetry'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useCommandStore } from '@/stores/commandStore'
import { useSubgraphNavigationStore } from '@/stores/subgraphNavigationStore'
import { useSubgraphStore } from '@/stores/subgraphStore'
import { forEachSubgraphNode } from '@/utils/graphTraversalUtil'

const MIN_WIDTH = 28
const ITEM_GAP = 8
const ITEM_PADDING = 8
const ICON_WIDTH = 20

const workflowStore = useWorkflowStore()
const navigationStore = useSubgraphNavigationStore()
const breadcrumbRef = ref<InstanceType<typeof Breadcrumb>>()
const workflowName = computed(() => workflowStore.activeWorkflow?.filename)
const isBlueprint = computed(() =>
  useSubgraphStore().isSubgraphBlueprint(workflowStore.activeWorkflow)
)
const collapseTabs = ref(false)
const overflowingTabs = ref(false)

const isInSubgraph = computed(() => navigationStore.navigationStack.length > 0)

const home = computed(() => ({
  label: workflowName.value,
  icon: 'pi pi-home',
  key: 'root',
  isBlueprint: isBlueprint.value,
  command: () => {
    useTelemetry()?.trackUiButtonClicked({
      button_id: 'breadcrumb_subgraph_root_selected'
    })
    const canvas = useCanvasStore().getCanvas()
    if (!canvas.graph) throw new TypeError('Canvas has no graph')

    canvas.setGraph(canvas.graph.rootGraph)
  }
}))

const items = computed(() => {
  const items = navigationStore.navigationStack.map<MenuItem>((subgraph) => ({
    label: subgraph.name,
    key: `subgraph-${subgraph.id}`,
    command: () => {
      useTelemetry()?.trackUiButtonClicked({
        button_id: 'breadcrumb_subgraph_item_selected'
      })
      const canvas = useCanvasStore().getCanvas()
      if (!canvas.graph) throw new TypeError('Canvas has no graph')

      canvas.setGraph(subgraph)
    },
    updateTitle: (title: string) => {
      const rootGraph = useCanvasStore().getCanvas().graph?.rootGraph
      if (!rootGraph) return

      forEachSubgraphNode(rootGraph, subgraph.id, (node) => {
        node.title = title
      })
    }
  }))

  return [home.value, ...items]
})

const activeItemKey = computed(() => items.value.at(-1)?.key)

const handleBackClick = () => {
  void useCommandStore().execute('Comfy.Graph.ExitSubgraph')
}

const breadcrumbElement = computed(() => {
  if (!breadcrumbRef.value) return null

  const el = (breadcrumbRef.value as unknown as { $el: HTMLElement }).$el
  const list = el?.querySelector('.p-breadcrumb-list') as HTMLElement
  return list
})

// Check for overflow on breadcrumb items and collapse/expand the breadcrumb to fit
let overflowObserver: ReturnType<typeof useOverflowObserver> | undefined
watch(breadcrumbElement, (el) => {
  overflowObserver?.dispose()
  overflowObserver = undefined

  if (!el) return

  overflowObserver = useOverflowObserver(el, {
    onCheck: (isOverflowing) => {
      overflowingTabs.value = isOverflowing

      if (collapseTabs.value) {
        // Items are currently hidden, check if we can show them
        if (!isOverflowing) {
          const items = [
            ...el.querySelectorAll('.p-breadcrumb-item')
          ] as HTMLElement[]

          if (items.length < 3) return

          const itemsWithIcon = items.filter((item) =>
            item.querySelector('.p-breadcrumb-item-link-icon-visible')
          ).length
          const separators = el.querySelectorAll(
            '.p-breadcrumb-separator'
          ) as NodeListOf<HTMLElement>
          const separator = separators[separators.length - 1] as HTMLElement
          const separatorWidth = separator.offsetWidth

          // items + separators + gaps + icons
          const itemsWidth =
            (MIN_WIDTH + ITEM_PADDING + ITEM_PADDING) * items.length +
            itemsWithIcon * ICON_WIDTH
          const separatorsWidth = (items.length - 1) * separatorWidth
          const gapsWidth = (items.length - 1) * (ITEM_GAP * 2)
          const totalWidth = itemsWidth + separatorsWidth + gapsWidth
          const containerWidth = el.clientWidth

          if (totalWidth <= containerWidth) {
            collapseTabs.value = false
          }
        }
      } else if (isOverflowing) {
        collapseTabs.value = true
      }
    }
  })
})

// If e.g. the workflow name changes, we need to check the overflow again
onUpdated(() => {
  if (!overflowObserver?.disposed.value) {
    overflowObserver?.checkOverflow()
  }
})
</script>

<style scoped>
.subgraph-breadcrumb:not(:empty) {
  flex: auto;
  flex-shrink: 10000;
  min-width: 120px;
}

.subgraph-breadcrumb,
:deep(.p-breadcrumb) {
  overflow: hidden;
}

:deep(.p-breadcrumb) {
  width: 100%;
  background-color: transparent;
}

:deep(.p-breadcrumb-item) {
  display: flex;
  align-items: center;
  overflow: hidden;
  height: calc(var(--spacing) * 8);
  min-width: calc(var(--p-breadcrumb-item-min-width) + 1rem);
  border: 1px solid transparent;
  background-color: transparent;
  transition: all 0.2s;
  /* Collapse middle items first */
  flex-shrink: 10000;
}

:deep(.p-breadcrumb-separator) {
  border: 1px solid transparent;
  background-color: transparent;
  display: flex;
  padding: 0 var(--p-breadcrumb-item-margin);
}

:deep(.p-breadcrumb-item-link) {
  padding: 0
    calc(var(--p-breadcrumb-item-margin) + var(--p-breadcrumb-item-padding));
}

:deep(.p-breadcrumb-item:hover) {
  border-radius: var(--radius-lg);
  border-color: var(--interface-stroke);
  background-color: var(--comfy-menu-bg);
}

:deep(.p-breadcrumb-item:has(.p-breadcrumb-item-link-icon-visible)) {
  min-width: calc(var(--p-breadcrumb-item-min-width) + 1rem + 20px);
}

:deep(.p-breadcrumb-item:first-child) {
  /* Then collapse the root workflow */
  flex-shrink: 5000;

  .p-breadcrumb-item-link {
    padding-left: var(--p-breadcrumb-item-padding);
  }
}

:deep(.p-breadcrumb-item:last-child) {
  /* Then collapse the active item */
  flex-shrink: 1;
}

:deep(.p-breadcrumb-item-link-menu-visible) {
  background-color: color-mix(
    in srgb,
    var(--fg-color) 10%,
    var(--comfy-menu-bg)
  ) !important;
  color: var(--fg-color);
}
</style>

<style>
.subgraph-breadcrumb-collapse .p-breadcrumb-list {
  .p-breadcrumb-item,
  .p-breadcrumb-separator {
    display: none;
  }

  .p-breadcrumb-item:nth-last-child(3),
  .p-breadcrumb-separator:nth-last-child(2),
  .p-breadcrumb-item:nth-last-child(1) {
    display: flex;
  }
}
</style>
