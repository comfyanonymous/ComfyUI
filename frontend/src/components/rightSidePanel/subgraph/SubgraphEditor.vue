<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'

import DraggableList from '@/components/common/DraggableList.vue'
import Button from '@/components/ui/button/Button.vue'
import {
  demoteWidget,
  getPromotableWidgets,
  getWidgetName,
  isRecommendedWidget,
  promoteWidget,
  pruneDisconnected
} from '@/core/graph/subgraph/promotionUtils'
import type { WidgetItem } from '@/core/graph/subgraph/promotionUtils'
import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import { SubgraphNode } from '@/lib/litegraph/src/subgraph/SubgraphNode'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import FormSearchInput from '@/renderer/extensions/vueNodes/widgets/components/form/FormSearchInput.vue'
import { useLitegraphService } from '@/services/litegraphService'
import { usePromotionStore } from '@/stores/promotionStore'
import { useRightSidePanelStore } from '@/stores/workspace/rightSidePanelStore'
import { cn } from '@/utils/tailwindUtil'

import SubgraphNodeWidget from './SubgraphNodeWidget.vue'

const { t } = useI18n()
const canvasStore = useCanvasStore()
const promotionStore = usePromotionStore()
const rightSidePanelStore = useRightSidePanelStore()
const { searchQuery } = storeToRefs(rightSidePanelStore)

const promotionEntries = computed(() => {
  const node = activeNode.value
  if (!node) return []
  return promotionStore.getPromotions(node.rootGraph.id, node.id)
})

const activeNode = computed(() => {
  const node = canvasStore.selectedItems[0]
  if (node instanceof SubgraphNode) return node
  return undefined
})

const activeWidgets = computed<WidgetItem[]>({
  get() {
    const node = activeNode.value
    if (!node) return []

    return promotionEntries.value.flatMap(
      ({ interiorNodeId, widgetName }): WidgetItem[] => {
        if (interiorNodeId === '-1') {
          const widget = node.widgets.find((w) => w.name === widgetName)
          if (!widget) return []
          return [
            [{ id: -1, title: t('subgraphStore.linked'), type: '' }, widget]
          ]
        }
        const wNode = node.subgraph._nodes_by_id[interiorNodeId]
        if (!wNode) return []
        const widget = getPromotableWidgets(wNode).find(
          (w) => w.name === widgetName
        )
        if (!widget) return []
        return [[wNode, widget]]
      }
    )
  },
  set(value: WidgetItem[]) {
    const node = activeNode.value
    if (!node) {
      console.error('Attempted to toggle widgets with no node selected')
      return
    }
    promotionStore.setPromotions(
      node.rootGraph.id,
      node.id,
      value.map(([n, w]) => ({
        interiorNodeId: String(n.id),
        widgetName: getWidgetName(w)
      }))
    )
  }
})

const interiorWidgets = computed<WidgetItem[]>(() => {
  const node = activeNode.value
  if (!node) return []
  const { updatePreviews } = useLitegraphService()
  const interiorNodes = node.subgraph.nodes
  for (const node of interiorNodes) {
    node.updateComputedDisabled()
    updatePreviews(node)
  }
  return interiorNodes
    .flatMap(nodeWidgets)
    .filter(([_, w]: WidgetItem) => !w.computedDisabled)
})

const candidateWidgets = computed<WidgetItem[]>(() => {
  const node = activeNode.value
  if (!node) return []
  return interiorWidgets.value.filter(
    ([n, w]: WidgetItem) =>
      !promotionStore.isPromoted(
        node.rootGraph.id,
        node.id,
        String(n.id),
        w.name
      )
  )
})
const filteredCandidates = computed<WidgetItem[]>(() => {
  const query = searchQuery.value.toLowerCase()
  if (!query) return candidateWidgets.value
  return candidateWidgets.value.filter(
    ([n, w]: WidgetItem) =>
      n.title.toLowerCase().includes(query) ||
      w.name.toLowerCase().includes(query)
  )
})

const recommendedWidgets = computed(() => {
  const node = activeNode.value
  if (!node) return []
  return filteredCandidates.value.filter(isRecommendedWidget)
})

const filteredActive = computed<WidgetItem[]>(() => {
  const query = searchQuery.value.toLowerCase()
  if (!query) return activeWidgets.value
  return activeWidgets.value.filter(
    ([n, w]: WidgetItem) =>
      n.title.toLowerCase().includes(query) ||
      w.name.toLowerCase().includes(query)
  )
})

function toKey(item: WidgetItem) {
  return `${item[0].id}: ${item[1].name}`
}
function nodeWidgets(n: LGraphNode): WidgetItem[] {
  return getPromotableWidgets(n).map((w) => [n, w])
}
function demote([node, widget]: WidgetItem) {
  const subgraphNode = activeNode.value
  if (!subgraphNode) return
  demoteWidget(node, widget, [subgraphNode])
  promotionStore.demote(
    subgraphNode.rootGraph.id,
    subgraphNode.id,
    String(node.id),
    getWidgetName(widget)
  )
}
function promote([node, widget]: WidgetItem) {
  const subgraphNode = activeNode.value
  if (!subgraphNode) return
  promoteWidget(node, widget, [subgraphNode])
  promotionStore.promote(
    subgraphNode.rootGraph.id,
    subgraphNode.id,
    String(node.id),
    widget.name
  )
}
function showAll() {
  const node = activeNode.value
  if (!node) return
  for (const [n, w] of filteredCandidates.value) {
    promotionStore.promote(node.rootGraph.id, node.id, String(n.id), w.name)
  }
}
function hideAll() {
  const node = activeNode.value
  if (!node) return
  for (const [n, w] of filteredActive.value) {
    if (String(n.id) === '-1') continue
    promotionStore.demote(
      node.rootGraph.id,
      node.id,
      String(n.id),
      getWidgetName(w)
    )
  }
}
function showRecommended() {
  const node = activeNode.value
  if (!node) return
  for (const [n, w] of recommendedWidgets.value) {
    promotionStore.promote(node.rootGraph.id, node.id, String(n.id), w.name)
  }
}

onMounted(() => {
  if (activeNode.value) pruneDisconnected(activeNode.value)
})
</script>

<template>
  <div v-if="activeNode" class="subgraph-edit-section flex h-full flex-col">
    <div class="px-4 pb-4 pt-1 flex gap-2 border-b border-interface-stroke">
      <FormSearchInput v-model="searchQuery" />
    </div>

    <div class="flex-1">
      <div
        v-if="
          searchQuery &&
          filteredActive.length === 0 &&
          filteredCandidates.length === 0
        "
        class="text-sm text-muted-foreground px-4 py-10 text-center"
      >
        {{ $t('rightSidePanel.noneSearchDesc') }}
      </div>

      <div
        v-if="filteredActive.length"
        class="flex flex-col border-b border-interface-stroke"
      >
        <div
          class="sticky top-0 z-10 flex items-center justify-between backdrop-blur-xl min-h-12 px-4"
        >
          <div class="text-sm font-semibold uppercase line-clamp-1">
            {{ $t('subgraphStore.shown') }}
          </div>
          <a
            class="cursor-pointer text-right text-xs font-normal text-text-secondary hover:text-azure-600 whitespace-nowrap"
            @click.stop="hideAll"
          >
            {{ $t('subgraphStore.hideAll') }}</a
          >
        </div>
        <DraggableList v-slot="{ dragClass }" v-model="activeWidgets">
          <SubgraphNodeWidget
            v-for="[node, widget] in filteredActive"
            :key="toKey([node, widget])"
            :class="cn(!searchQuery && dragClass, 'bg-comfy-menu-bg')"
            :node-title="node.title"
            :widget-name="widget.name"
            :is-physical="node.id === -1"
            :is-draggable="!searchQuery"
            @toggle-visibility="demote([node, widget])"
          />
        </DraggableList>
      </div>

      <div
        v-if="filteredCandidates.length"
        class="flex flex-col border-b border-interface-stroke"
      >
        <div
          class="sticky top-0 z-10 flex items-center justify-between backdrop-blur-xl min-h-12 px-4"
        >
          <div class="text-sm font-semibold uppercase line-clamp-1">
            {{ $t('subgraphStore.hidden') }}
          </div>
          <a
            class="cursor-pointer text-right text-xs font-normal text-text-secondary hover:text-azure-600 whitespace-nowrap"
            @click.stop="showAll"
          >
            {{ $t('subgraphStore.showAll') }}</a
          >
        </div>
        <div class="pb-2 px-2 space-y-0.5 mt-0.5">
          <SubgraphNodeWidget
            v-for="[node, widget] in filteredCandidates"
            :key="toKey([node, widget])"
            class="bg-comfy-menu-bg"
            :node-title="node.title"
            :widget-name="widget.name"
            @toggle-visibility="promote([node, widget])"
          />
        </div>
      </div>

      <div
        v-if="recommendedWidgets.length"
        class="flex justify-center border-b border-interface-stroke py-4"
      >
        <Button
          size="sm"
          class="rounded border-none px-3 py-0.5"
          @click.stop="showRecommended"
        >
          {{ $t('subgraphStore.showRecommended') }}
        </Button>
      </div>
    </div>
  </div>
</template>
