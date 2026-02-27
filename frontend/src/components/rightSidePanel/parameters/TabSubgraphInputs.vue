<script setup lang="ts">
import { useMounted, watchDebounced } from '@vueuse/core'
import { storeToRefs } from 'pinia'
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  shallowRef,
  useTemplateRef,
  watch
} from 'vue'
import { useI18n } from 'vue-i18n'

import { isPromotedWidgetView } from '@/core/graph/subgraph/promotedWidgetTypes'
import type { SubgraphNode } from '@/lib/litegraph/src/subgraph/SubgraphNode'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import FormSearchInput from '@/renderer/extensions/vueNodes/widgets/components/form/FormSearchInput.vue'
import { DraggableList } from '@/scripts/ui/draggableList'
import { usePromotionStore } from '@/stores/promotionStore'
import { useRightSidePanelStore } from '@/stores/workspace/rightSidePanelStore'

import { searchWidgets } from '../shared'
import type { NodeWidgetsList } from '../shared'
import SectionWidgets from './SectionWidgets.vue'

const { node } = defineProps<{
  node: SubgraphNode
}>()

const { t } = useI18n()
const canvasStore = useCanvasStore()
const promotionStore = usePromotionStore()
const rightSidePanelStore = useRightSidePanelStore()
const { focusedSection, searchQuery } = storeToRefs(rightSidePanelStore)

const advancedInputsCollapsed = ref(true)
const draggableList = ref<DraggableList | undefined>(undefined)
const sectionWidgetsRef = useTemplateRef('sectionWidgetsRef')
const advancedInputsSectionRef = useTemplateRef('advancedInputsSectionRef')

const promotionEntries = computed(() =>
  promotionStore.getPromotions(node.rootGraph.id, node.id)
)

watch(
  focusedSection,
  async (section) => {
    if (section === 'advanced-inputs') {
      advancedInputsCollapsed.value = false
      rightSidePanelStore.clearFocusedSection()

      await nextTick()

      await new Promise((resolve) => setTimeout(resolve, 300))

      const sectionComponent = advancedInputsSectionRef.value
      const sectionElement = sectionComponent?.rootElement
      if (sectionElement) {
        sectionElement.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }
    }
  },
  { immediate: true }
)

const widgetsList = computed((): NodeWidgetsList => {
  const entries = promotionEntries.value
  const { widgets = [] } = node

  const result: NodeWidgetsList = []
  for (const { interiorNodeId, widgetName } of entries) {
    const widget = widgets.find((w) => {
      if (isPromotedWidgetView(w)) {
        return (
          String(w.sourceNodeId) === interiorNodeId &&
          w.sourceWidgetName === widgetName
        )
      }
      return w.name === widgetName
    })
    if (widget) {
      result.push({ node, widget })
    }
  }
  return result
})

const advancedInputsWidgets = computed((): NodeWidgetsList => {
  const interiorNodes = node.subgraph.nodes

  const allInteriorWidgets = interiorNodes.flatMap((interiorNode) => {
    const { widgets = [] } = interiorNode
    return widgets
      .filter((w) => !w.computedDisabled)
      .map((widget) => ({ node: interiorNode, widget }))
  })

  return allInteriorWidgets.filter(
    ({ node: interiorNode, widget }) =>
      !promotionStore.isPromoted(
        node.rootGraph.id,
        node.id,
        String(interiorNode.id),
        widget.name
      )
  )
})

const parents = computed<SubgraphNode[]>(() => [node])

const searchedWidgetsList = shallowRef<NodeWidgetsList>(widgetsList.value)
const isSearching = ref(false)

async function searcher(query: string) {
  isSearching.value = query.trim() !== ''
  searchedWidgetsList.value = searchWidgets(widgetsList.value, query)
}

const isMounted = useMounted()

function setDraggableState() {
  if (!isMounted.value) return

  draggableList.value?.dispose()
  const container = sectionWidgetsRef.value?.widgetsContainer
  if (isSearching.value || !container?.children?.length) return

  draggableList.value = new DraggableList(container, '.draggable-item')

  draggableList.value.applyNewItemsOrder = function () {
    const reorderedItems: HTMLElement[] = []

    let oldPosition = -1
    this.getAllItems().forEach((item, index) => {
      if (item === this.draggableItem) {
        oldPosition = index
        return
      }
      if (!this.isItemToggled(item)) {
        reorderedItems[index] = item
        return
      }
      const newIndex = this.isItemAbove(item) ? index + 1 : index - 1
      reorderedItems[newIndex] = item
    })

    if (oldPosition === -1) {
      console.error('[TabSubgraphInputs] draggableItem not found in items')
      return
    }

    for (let index = 0; index < this.getAllItems().length; index++) {
      const item = reorderedItems[index]
      if (typeof item === 'undefined') {
        reorderedItems[index] = this.draggableItem as HTMLElement
      }
    }

    const newPosition = reorderedItems.indexOf(
      this.draggableItem as HTMLElement
    )

    promotionStore.movePromotion(
      node.rootGraph.id,
      node.id,
      oldPosition,
      newPosition
    )
    canvasStore.canvas?.setDirty(true, true)
  }
}

watchDebounced(searchedWidgetsList, () => setDraggableState(), {
  debounce: 100
})
onMounted(() => setDraggableState())
onBeforeUnmount(() => draggableList.value?.dispose())

const label = computed(() => {
  return searchedWidgetsList.value.length !== 0
    ? t('rightSidePanel.inputs')
    : t('rightSidePanel.inputsNone')
})
</script>

<template>
  <div class="px-4 pt-1 pb-4 flex gap-2 border-b border-interface-stroke">
    <FormSearchInput
      v-model="searchQuery"
      :searcher
      :update-key="widgetsList"
    />
  </div>
  <SectionWidgets
    ref="sectionWidgetsRef"
    :node
    :label
    :parents
    :widgets="searchedWidgetsList"
    :is-draggable="!isSearching"
    :enable-empty-state="isSearching"
    :tooltip="
      isSearching || searchedWidgetsList.length
        ? ''
        : t('rightSidePanel.inputsNoneTooltip')
    "
    class="border-b border-interface-stroke"
    @update:collapse="nextTick(setDraggableState)"
  >
    <template #empty>
      <div class="text-sm text-muted-foreground px-4 text-center pt-5 pb-15">
        {{ t('rightSidePanel.noneSearchDesc') }}
      </div>
    </template>
  </SectionWidgets>
  <SectionWidgets
    v-if="advancedInputsWidgets.length > 0"
    ref="advancedInputsSectionRef"
    v-model:collapse="advancedInputsCollapsed"
    :label="t('rightSidePanel.advancedInputs')"
    :parents="parents"
    :widgets="advancedInputsWidgets"
    show-node-name
    class="border-b border-interface-stroke"
  />
</template>
